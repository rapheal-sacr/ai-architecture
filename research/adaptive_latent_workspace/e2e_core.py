"""Small functional E2E-TTT mechanism port, not a released-model reproduction.

Source: test-time-training/e2e a4fc4788ace38e29b5067916d4f4be33da894085.
Preserves causal SWA, sequential fast/static SwiGLU, late-MLP clipped SGD,
pre-update CE and differentiation through updates. All layers stream by chunk;
the upstream implementation computes the fixed prefix once for the sequence.
No fused kernels, dropout, mixed precision or gradient checkpointing here.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class Config:
    vocab: int = 32
    width: int = 32
    hidden: int = 64
    layers: int = 4
    suffix: int = 1
    heads: int = 4
    window: int = 8
    chunk: int = 4
    inner_lr: float = 1.0
    clip: float = 1.0
    eps: float = 1e-6
    rope_theta: float = 500000.0
    init_std: float = 0.15

    def validate(self):
        assert self.width % self.heads == 0
        assert (self.width // self.heads) % 2 == 0
        assert 0 < self.suffix <= self.layers
        assert 0 < self.chunk <= self.window
        assert self.inner_lr >= 0 and self.clip > 0


@dataclass
class StreamState:
    fast: dict[str, Tensor]
    cache: list[tuple[Tensor, Tensor]]
    position: int

    def detached(self):
        """Evaluation boundary: retain values, release history, permit next update."""
        return StreamState(
            {k: v.detach().clone().requires_grad_(True) for k, v in self.fast.items()},
            [(k.detach().clone(), v.detach().clone()) for k, v in self.cache],
            self.position,
        )

    def clear_context(self):
        """A new document/session with persistent weights, no previous KV context."""
        return StreamState(self.fast, [(k.new_empty((0, *k.shape[1:])),
                                       v.new_empty((0, *v.shape[1:]))) for k, v in self.cache], 0)

    def payload(self):
        return {
            "fast": {k: v.detach().clone() for k, v in self.fast.items()},
            "cache": [(k.detach().clone(), v.detach().clone()) for k, v in self.cache],
            "position": self.position,
        }

    @classmethod
    def restore(cls, payload):
        return cls(
            {k: v.clone().requires_grad_(True) for k, v in payload["fast"].items()},
            [(k.clone(), v.clone()) for k, v in payload["cache"]],
            int(payload["position"]),
        )

    def bytes(self):
        # Tensor storage only. Python containers/scalar position reported separately.
        return sum(t.numel() * t.element_size() for t in self.fast.values()) + sum(
            t.numel() * t.element_size() for pair in self.cache for t in pair
        )


class E2ECore(nn.Module):
    def __init__(self, cfg: Config, *, seed: int, dtype=torch.float32):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        rng = torch.Generator(device="cpu").manual_seed(seed)
        p = {}

        def weight(name, shape, norm=False):
            value = torch.ones(shape, dtype=dtype) if norm else (
                torch.randn(shape, generator=rng, dtype=dtype) * cfg.init_std
            )
            p[name] = nn.Parameter(value)

        weight("embed", (cfg.vocab, cfg.width))
        weight("final_norm", (cfg.width,), True)
        for layer in range(cfg.layers):
            pre = f"b{layer}_"
            for name in ("attn_norm", "static_norm", "attn_post", "static_post"):
                weight(pre + name, (cfg.width,), True)
            for name in ("q_norm", "k_norm"):
                weight(pre + name, (cfg.width // cfg.heads,), True)
            for name in ("q", "k", "v", "o"):
                weight(pre + name, (cfg.width, cfg.width))
            mlps = ["static"]
            if layer >= cfg.layers - cfg.suffix:
                mlps.append("fast")
                weight(pre + "fast_norm", (cfg.width,), True)
                weight(pre + "fast_post", (cfg.width,), True)
            for name in mlps:
                weight(pre + name + "_w1", (cfg.width, cfg.hidden))
                weight(pre + name + "_w3", (cfg.width, cfg.hidden))
                weight(pre + name + "_w2", (cfg.hidden, cfg.width))
        self.p = nn.ParameterDict(p)
        # Like upstream spec_inner, norms are outer-only, even the prime norm.
        self.fast_names = tuple(k for k in self.p if "_fast_w" in k)

    def initial_state(self, params: Mapping[str, Tensor] | None = None):
        p = self.p if params is None else params
        exemplar = p["embed"]
        cfg = self.cfg
        return StreamState(
            {k: p[k] for k in self.fast_names},
            [(exemplar.new_empty((0, cfg.heads, cfg.width // cfg.heads)),
              exemplar.new_empty((0, cfg.heads, cfg.width // cfg.heads)))
             for _ in range(cfg.layers)],
            0,
        )

    def norm(self, x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.cfg.eps) * w

    @staticmethod
    def mlp(x, p, name):
        return (F.silu(x @ p[name + "_w1"]) * (x @ p[name + "_w3"])) @ p[name + "_w2"]

    def rope(self, x, positions):
        dim = x.shape[-1]
        inv = self.cfg.rope_theta ** (
            -torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim
        )
        angle = positions.to(x.dtype)[:, None, None] * inv[None, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * angle.cos() - b * angle.sin(),
                            a * angle.sin() + b * angle.cos()), dim=-1).flatten(-2)

    def chunk_logits(self, tokens, state: StreamState, params=None):
        """No target tokens, update, or input-state mutation in this operation."""
        cfg = self.cfg
        assert tokens.ndim == 1 and 0 < len(tokens) <= cfg.chunk
        outer = self.p if params is None else params
        p = {**outer, **state.fast}
        x = F.embedding(tokens, p["embed"])
        caches, elements = [], 0
        n = len(tokens)
        qp = torch.arange(state.position, state.position + n, device=x.device)
        for layer in range(cfg.layers):
            name = f"b{layer}_"
            z = self.norm(x, p[name + "attn_norm"])
            q, k, v = [(z @ p[name + a]).view(n, cfg.heads, -1) for a in ("q", "k", "v")]
            q, k = self.norm(q, p[name + "q_norm"]), self.norm(k, p[name + "k_norm"])
            old_k, old_v = state.cache[layer]
            k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            kp = torch.arange(state.position - len(old_k), state.position + n, device=x.device)
            qr, kr = self.rope(q, qp), self.rope(k, kp)
            scores = torch.einsum("thd,shd->hts", qr, kr) / (cfg.width // cfg.heads) ** 0.5
            mask = (qp[:, None] >= kp) & (qp[:, None] - kp < cfg.window)
            scores = scores.masked_fill(~mask[None], -torch.inf)
            attn = torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(n, cfg.width)
            x = x + self.norm(attn @ p[name + "o"], p[name + "attn_post"])
            # IMPORTANT: fast residual changes the input of the frozen static MLP.
            if name + "fast_w1" in p:
                x = x + self.norm(self.mlp(self.norm(x, p[name + "fast_norm"]), p, name + "fast"),
                                  p[name + "fast_post"])
            x = x + self.norm(self.mlp(self.norm(x, p[name + "static_norm"]), p, name + "static"),
                              p[name + "static_post"])
            caches.append((k[-cfg.window:], v[-cfg.window:]))
            elements += cfg.heads * n * len(k)
        logits = self.norm(x, p["final_norm"]) @ p["embed"].T
        return logits, StreamState(state.fast, caches, state.position + n), elements

    def step(self, inputs, targets, state, *, params=None, mode="second_order"):
        """Score the chunk first, update after labels are observed.

        second_order: full E2E meta-gradient; first_order: stop only the update
        gradient's derivative; frozen: no inner update. In evaluation, detach
        the returned state between steps, after observing/recording the loss.
        """
        assert mode in ("second_order", "first_order", "frozen")
        assert inputs.shape == targets.shape
        logits, next_state, elements = self.chunk_logits(inputs, state, params)
        token_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = token_loss.mean()
        grad_norm = loss.new_zeros(())
        if mode != "frozen":
            grad = torch.autograd.grad(loss, tuple(state.fast.values()),
                                       create_graph=(mode == "second_order"), retain_graph=True)
            grad_norm = torch.stack([g.square().sum() for g in grad]).sum().sqrt()
            scale = (self.cfg.clip / grad_norm.clamp_min(1e-30)).clamp(max=1.0)
            updates = [scale * g for g in grad]
            if mode == "first_order":
                updates = [g.detach() for g in updates]
            next_state.fast = {
                k: value - self.cfg.inner_lr * g
                for (k, value), g in zip(state.fast.items(), updates)
            }
        return loss, token_loss, logits, next_state, {
            "forward_tokens": len(inputs), "attention_score_elements": elements,
            "inner_updates": int(mode != "frozen"),
            "inner_grad_norm": float(grad_norm.detach()),
        }

    def sequence(self, tokens, *, state=None, params=None, mode="second_order",
                 detach_between=False):
        """tokens[0] is context; token t is scored before it is learned as a label.

        Returns final state explicitly. Unlike upstream's evaluator, callers
        may carry it to another sequence. That extension is not a retention proof.
        """
        assert tokens.ndim == 1 and (len(tokens) - 1) % self.cfg.chunk == 0
        if state is None:
            state = self.initial_state(params)
        if detach_between:
            state = state.detached()
        losses, logits, work = [], [], []
        for start in range(0, len(tokens) - 1, self.cfg.chunk):
            end = start + self.cfg.chunk
            _, tok_loss, out, state, w = self.step(tokens[start:end], tokens[start+1:end+1],
                                                   state, params=params, mode=mode)
            losses.append(tok_loss.detach() if detach_between else tok_loss)
            logits.append(out.detach() if detach_between else out)
            work.append(w)
            if detach_between:
                state = state.detached()
        per_token = torch.cat(losses)
        return per_token.mean(), per_token, torch.cat(logits), state, work

    def specification(self):
        return {
            "config": asdict(self.cfg), "parameters": sum(p.numel() for p in self.p.values()),
            "fast_parameters": sum(self.p[k].numel() for k in self.fast_names),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.p.values()),
            "dtype": str(self.p["embed"].dtype), "fast_names": list(self.fast_names),
        }
