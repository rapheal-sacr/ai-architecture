"""Bounded, causal archive of learned E2E fast states.

One adaptive final MLP only. Frozen prefix work and all attention caches are
shared; fast and following static MLPs are reevaluated for each candidate.
Selection/admission/gating rules are fixed human-written rules, not a learned
router or recursive procedure improvement. Observations, never context IDs,
drive them. Current predictions are scored before any target-based selection.
"""
import copy
import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from e2e_core import E2ECore, StreamState

POLICY_VERSION = "suffix-archive-v1"


def copy_fast(fast, *, grad=False):
    return {k: v.detach().clone().requires_grad_(grad) for k, v in fast.items()}


def model_fingerprint(model):
    h = hashlib.sha256(json.dumps(asdict(model.cfg), sort_keys=True).encode())
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SharedSuffix:
    def __init__(self, model: E2ECore):
        if model.cfg.suffix != 1:
            raise ValueError("Shared prefix/cache requires exactly one adaptive final block")
        self.model = model

    def features(self, tokens, state):
        """Exactly the core's operation order through the last attention residual."""
        m, cfg, p = self.model, self.model.cfg, self.model.p
        assert tokens.ndim == 1 and 0 < len(tokens) <= cfg.chunk
        with torch.no_grad():
            x = F.embedding(tokens, p["embed"])
            n = len(tokens)
            qp = torch.arange(state.position, state.position + n, device=x.device)
            caches, elements = [], 0
            for layer in range(cfg.layers):
                name = f"b{layer}_"
                z = m.norm(x, p[name + "attn_norm"])
                q, k, v = [(z @ p[name + a]).view(n, cfg.heads, -1) for a in ("q", "k", "v")]
                q, k = m.norm(q, p[name + "q_norm"]), m.norm(k, p[name + "k_norm"])
                old_k, old_v = state.cache[layer]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
                kp = torch.arange(state.position - len(old_k), state.position + n, device=x.device)
                qr, kr = m.rope(q, qp), m.rope(k, kp)
                scores = torch.einsum("thd,shd->hts", qr, kr) / (cfg.width // cfg.heads) ** .5
                mask = (qp[:, None] >= kp) & (qp[:, None] - kp < cfg.window)
                scores = scores.masked_fill(~mask[None], -torch.inf)
                attn = torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(n, cfg.width)
                x = x + m.norm(attn @ p[name + "o"], p[name + "attn_post"])
                if layer < cfg.layers - 1:
                    x = x + m.norm(m.mlp(m.norm(x, p[name + "static_norm"]), p, name + "static"), p[name + "static_post"])
                caches.append((k[-cfg.window:].clone(), v[-cfg.window:].clone()))
                elements += cfg.heads * n * len(k)
        return x, StreamState(state.fast, caches, state.position + n), elements

    def logits(self, features, fast):
        m, cfg = self.model, self.model.cfg
        p = {**m.p, **fast}
        name = f"b{cfg.layers-1}_"
        x = features + m.norm(m.mlp(m.norm(features, p[name + "fast_norm"]), p, name + "fast"), p[name + "fast_post"])
        # The static residual depends on the fast residual and cannot be shared.
        x = x + m.norm(m.mlp(m.norm(x, p[name + "static_norm"]), p, name + "static"), p[name + "static_post"])
        return m.norm(x, p["final_norm"]) @ p["embed"].T

    def updated(self, loss, fast):
        grads = torch.autograd.grad(loss, tuple(fast.values()))
        norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt()
        scale = (self.model.cfg.clip / norm.clamp_min(1e-30)).clamp(max=1.0)
        updated = {k: (v - self.model.cfg.inner_lr * (scale * g)).detach().clone().requires_grad_(True)
                   for (k, v), g in zip(fast.items(), grads)}
        return updated, float(norm.detach())


@dataclass(frozen=True)
class ArchiveConfig:
    capacity: int = 0
    select: bool = True
    update_nll_threshold: float | None = None
    switch_margin: float = .1
    probation_chunks: int = 4
    admit_accuracy: float = .75
    admit_nll: float = 1.0

    def validate(self):
        assert self.capacity >= 0 and self.probation_chunks >= 1
        assert self.switch_margin >= 0 and 0 <= self.admit_accuracy <= 1
        assert self.admit_nll >= 0
        assert self.update_nll_threshold is None or self.update_nll_threshold >= 0


class SuffixArchive:
    def __init__(self, model, config: ArchiveConfig):
        config.validate()
        self.model, self.config = model, config
        self.kernel = SharedSuffix(model)
        self.identity = model_fingerprint(model)
        self.state = model.initial_state().detached()
        self.archives = []
        self.probe = None
        self.clock = 0
        self.next_id = 0
        self.totals = {k: 0 for k in ("prefix_tokens", "attention_score_elements", "suffix_tokens",
            "suffix_forwards", "gradient_updates", "gradient_tokens", "switches", "admissions",
            "duplicates", "rejections", "evictions")}
        if config.capacity:
            self._new_probe()

    def _new_probe(self):
        self.probe = {"fast": copy_fast(self.state.fast), "born": self.clock, "chunks": 0,
                      "n": 0, "loss_sum": 0., "correct": 0, "comparisons": {}}

    def _finish_probe(self):
        p, c = self.probe, self.config
        assert p is not None and p["chunks"] == c.probation_chunks
        event = {"born": p["born"], "validated_through": self.clock, "n": p["n"],
                 "correct": p["correct"], "mean_nll": p["loss_sum"] / p["n"]}
        if p["correct"] / p["n"] < c.admit_accuracy or event["mean_nll"] > c.admit_nll:
            self.totals["rejections"] += 1
            event["outcome"] = "reject"
            return event
        # Exact prediction agreement on the fresh probation sample only. This
        # does not certify equivalence on arbitrary future inputs/queries.
        eligible = []
        for entry in self.archives:
            compare = p["comparisons"].get(entry["id"])
            if compare and compare["n"] == p["n"] and compare["agree"] == p["n"] and compare["correct"] / p["n"] >= c.admit_accuracy:
                eligible.append((compare["loss_sum"] / compare["n"], entry["id"], entry))
        if eligible:
            _, _, entry = min(eligible, key=lambda x: (x[0], x[1]))
            entry["last_used"] = self.clock
            self.totals["duplicates"] += 1
            event.update(outcome="duplicate", matched_id=entry["id"])
            return event
        evicted = None
        if len(self.archives) == c.capacity:
            old = min(self.archives, key=lambda e: (e["last_used"], e["admitted"], e["id"]))
            self.archives = [e for e in self.archives if e["id"] != old["id"]]
            evicted = old["id"]
            self.totals["evictions"] += 1
        entry = {"id": self.next_id, "fast": copy_fast(p["fast"]),
                 "admitted": self.clock, "last_used": self.clock}
        self.next_id += 1
        self.archives.append(entry)
        self.totals["admissions"] += 1
        event.update(outcome="admit", admitted_id=entry["id"], evicted_id=evicted)
        return event

    def step(self, inputs, targets):
        """No context/boundary argument. Only the returned operational logits score.

        Candidate losses use labels that have just become observed. They choose
        the branch used for subsequent predictions and its one current-data
        update; they never retrospectively replace operational predictions.
        """
        assert inputs.shape == targets.shape
        n = len(inputs)
        features, next_state, attention_elements = self.kernel.features(inputs, self.state)
        actual_logits = self.kernel.logits(features, self.state.fast)
        actual_token_loss = F.cross_entropy(actual_logits, targets, reduction="none")
        actual_loss = actual_token_loss.mean()
        actual_correct = int(actual_logits.argmax(-1).eq(targets).sum())
        suffix_calls = 1
        candidates = []
        for entry in self.archives:
            with torch.no_grad():
                logits = self.kernel.logits(features, entry["fast"])
                losses = F.cross_entropy(logits, targets, reduction="none")
            candidates.append({"entry": entry, "logits": logits, "losses": losses,
                               "loss": float(losses.mean()), "correct": int(logits.argmax(-1).eq(targets).sum())})
            suffix_calls += 1
        probe_logits = None
        if self.probe is not None:
            with torch.no_grad():
                probe_logits = self.kernel.logits(features, self.probe["fast"])
                probe_losses = F.cross_entropy(probe_logits, targets, reduction="none")
            suffix_calls += 1
            p = self.probe
            p["chunks"] += 1
            p["n"] += n
            p["loss_sum"] += float(probe_losses.sum())
            p["correct"] += int(probe_logits.argmax(-1).eq(targets).sum())
            for candidate in candidates:
                record = p["comparisons"].setdefault(candidate["entry"]["id"],
                    {"n": 0, "loss_sum": 0., "agree": 0, "correct": 0})
                record["n"] += n
                record["loss_sum"] += float(candidate["losses"].sum())
                record["correct"] += candidate["correct"]
                record["agree"] += int(candidate["logits"].argmax(-1).eq(probe_logits.argmax(-1)).sum())
        choice = None
        if self.config.select and candidates:
            best = min(candidates, key=lambda x: (x["loss"], x["entry"]["id"]))
            if best["loss"] + self.config.switch_margin < float(actual_loss.detach()):
                choice = best
        selected_fast = self.state.fast
        selected_loss = actual_loss
        selected_correct = actual_correct
        selected_nll = float(actual_loss.detach())
        if choice is not None:
            selected_fast = copy_fast(choice["entry"]["fast"], grad=True)
            selected_nll, selected_correct = choice["loss"], choice["correct"]
            choice["entry"]["last_used"] = self.clock + 1
            self.totals["switches"] += 1
        threshold = self.config.update_nll_threshold
        should_update = threshold is None or selected_correct < n or selected_nll > threshold
        grad_norm = 0.
        if should_update:
            if choice is not None:
                selected_logits = self.kernel.logits(features, selected_fast)
                selected_loss = F.cross_entropy(selected_logits, targets, reduction="none").mean()
                suffix_calls += 1
            selected_fast, grad_norm = self.kernel.updated(selected_loss, selected_fast)
        else:
            selected_fast = copy_fast(selected_fast, grad=True)
        next_state.fast = selected_fast
        self.state = next_state
        self.clock += 1
        self.totals["prefix_tokens"] += n
        self.totals["attention_score_elements"] += attention_elements
        self.totals["suffix_forwards"] += suffix_calls
        self.totals["suffix_tokens"] += suffix_calls * n
        self.totals["gradient_updates"] += int(should_update)
        self.totals["gradient_tokens"] += int(should_update) * n
        probe_event = None
        if self.probe is not None and self.probe["chunks"] == self.config.probation_chunks:
            probe_event = self._finish_probe()
            self._new_probe()
        assert len(self.archives) <= self.config.capacity
        return {"logits": actual_logits.detach(), "losses": actual_token_loss.detach(),
                "correct": actual_correct, "selected_archive": None if choice is None else choice["entry"]["id"],
                "selected_current_data_nll": selected_nll, "updated": should_update, "grad_norm": grad_norm,
                "candidate_scores": [{"id": x["entry"]["id"], "nll": x["loss"], "correct": x["correct"]} for x in candidates],
                "probe_event": probe_event, "suffix_forwards": suffix_calls, "attention_score_elements": attention_elements}

    def payload(self):
        return {"version": POLICY_VERSION, "identity": self.identity, "config": asdict(self.config), "state": self.state.payload(),
                "archives": copy.deepcopy(self.archives), "probe": copy.deepcopy(self.probe),
                "clock": self.clock, "next_id": self.next_id, "totals": dict(self.totals)}

    def restore(self, payload):
        if payload["version"] != POLICY_VERSION or payload["identity"] != self.identity or payload["config"] != asdict(self.config):
            raise ValueError("Incompatible model/representation or archive policy version")
        self.state = StreamState.restore(payload["state"])
        self.archives = copy.deepcopy(payload["archives"])
        self.probe = copy.deepcopy(payload["probe"])
        self.clock, self.next_id = payload["clock"], payload["next_id"]
        self.totals = dict(payload["totals"])
        assert len(self.archives) <= self.config.capacity

    def tensor_bytes(self):
        fast = lambda f: sum(v.numel() * v.element_size() for v in f.values())
        return {"active_and_cache": self.state.bytes(),
                "archives": sum(fast(e["fast"]) for e in self.archives),
                "probe": 0 if self.probe is None else fast(self.probe["fast"])}
