"""CPU mechanism checks for the small port; not a task-competence benchmark."""
import argparse
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from e2e_core import Config, E2ECore, StreamState


def dense_reference(model, tokens):
    """Frozen full-sequence reference; no streaming/cache or production mask."""
    p, c = model.p, model.cfg
    t = len(tokens)
    x = p["embed"][tokens]
    position = torch.arange(t, dtype=x.dtype)
    dim = c.width // c.heads
    phase = position[:, None] / c.rope_theta ** (torch.arange(0, dim, 2, dtype=x.dtype) / dim)
    rotation = torch.polar(torch.ones_like(phase), phase)[:, None, :]
    for layer in range(c.layers):
        prefix = f"b{layer}_"
        z = model.norm(x, p[prefix + "attn_norm"])
        q, k, v = [(z @ p[prefix + key]).reshape(t, c.heads, dim) for key in ("q", "k", "v")]
        q = model.norm(q, p[prefix + "q_norm"])
        k = model.norm(k, p[prefix + "k_norm"])
        q = torch.view_as_real(torch.view_as_complex(q.reshape(t, c.heads, -1, 2)) * rotation).flatten(-2)
        k = torch.view_as_real(torch.view_as_complex(k.reshape(t, c.heads, -1, 2)) * rotation).flatten(-2)
        outputs = []
        for i in range(t):
            begin = max(0, i - c.window + 1)
            # Individually attend only to known, in-window positions.
            score = (q[i][None] * k[begin:i+1]).sum(-1) / dim ** 0.5
            outputs.append((score.softmax(0)[..., None] * v[begin:i+1]).sum(0))
        x = x + model.norm(torch.stack(outputs).reshape(t, c.width) @ p[prefix + "o"], p[prefix + "attn_post"])
        if prefix + "fast_w1" in p:
            x = x + model.norm(model.mlp(model.norm(x, p[prefix + "fast_norm"]), p, prefix + "fast"), p[prefix + "fast_post"])
        x = x + model.norm(model.mlp(model.norm(x, p[prefix + "static_norm"]), p, prefix + "static"), p[prefix + "static_post"])
    return model.norm(x, p["final_norm"]) @ p["embed"].T


def assert_state_equal(a, b):
    assert a.position == b.position
    assert a.fast.keys() == b.fast.keys()
    for key in a.fast:
        assert torch.equal(a.fast[key], b.fast[key]), key
    for aa, bb in zip(a.cache, b.cache):
        for x, y in zip(aa, bb):
            assert torch.equal(x, y)


def main(out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(vocab=12, width=16, hidden=24, layers=4, suffix=1, heads=2,
                 window=4, chunk=2, inner_lr=0.7, init_std=0.2)
    model = E2ECore(cfg, seed=331, dtype=torch.float64)
    saved = {k: p.detach().clone() for k, p in model.p.items()}
    rng = torch.Generator().manual_seed(332)
    tokens = torch.randint(cfg.vocab, (17,), generator=rng)
    checks = {}

    # Independent attention/reference calculation, including cache wraparound.
    frozen = model.sequence(tokens, mode="frozen")
    dense = dense_reference(model, tokens[:-1])
    torch.testing.assert_close(frozen[2], dense, atol=1e-12, rtol=1e-12)
    checks["dense_reference_max_difference"] = float((frozen[2] - dense).abs().max().detach())

    # Future-token interventions at every boundary; token j may affect predicting
    # j+1, never logits whose targets are <= j (logit indices < j).
    full = model.sequence(tokens, mode="second_order")
    causal_max = 0.0
    for j in range(1, len(tokens)):
        changed = tokens.clone()
        changed[j] = (changed[j] + 1) % cfg.vocab
        altered = model.sequence(changed, mode="second_order")
        delta = float((full[2][:j] - altered[2][:j]).abs().max().detach())
        causal_max = max(causal_max, delta)
        assert delta == 0.0, (j, delta)
    checks["causality_all_16_future_interventions_max_difference"] = causal_max

    # Pre-update loss does not depend on learning rate. Post-update state does.
    no_lr = E2ECore(replace(cfg, inner_lr=0), seed=331, dtype=torch.float64)
    first = model.step(tokens[:2], tokens[1:3], model.initial_state())
    zero = no_lr.step(tokens[:2], tokens[1:3], no_lr.initial_state())
    assert torch.equal(first[2], zero[2]) and torch.equal(first[0], zero[0])
    assert any(not torch.equal(first[3].fast[k], zero[3].fast[k]) for k in model.fast_names)
    checks["pre_update_scoring"] = True

    # Evaluate uninterrupted vs actual save/load at a chunk boundary. Serialize
    # fast state AND attention caches; no hidden process state should be needed.
    operational = model.sequence(tokens, detach_between=True)
    prefix = model.sequence(tokens[:9], detach_between=True)
    buf = io.BytesIO()
    torch.save(prefix[3].payload(), buf)
    buf.seek(0)
    restored = StreamState.restore(torch.load(buf, weights_only=True))
    continuation = model.sequence(tokens[8:], state=restored, detach_between=True)
    assert torch.equal(operational[2], torch.cat((prefix[2], continuation[2])))
    assert_state_equal(operational[3], continuation[3])
    checks["serialized_continuation_exact"] = True

    # Storage does not grow with the number of processed chunks at inference.
    long = model.sequence(tokens.repeat(4)[:65], detach_between=True)
    assert long[3].bytes() == operational[3].bytes()
    expected = (sum(model.p[k].numel() for k in model.fast_names) +
                cfg.layers * 2 * cfg.window * cfg.width) * 8
    assert expected == long[3].bytes()
    checks["bounded_inference_state_tensor_bytes"] = expected
    checks["attention_elements_16_tokens"] = sum(w["attention_score_elements"] for w in operational[4])
    checks["attention_elements_64_tokens"] = sum(w["attention_score_elements"] for w in long[4])

    # Model parameters, including fast initialization, are functionally immutable
    # in the inner loop. This does NOT imply immutable output functions.
    assert all(torch.equal(saved[k], p) for k, p in model.p.items())
    query = tokens[:2]
    old = model.chunk_logits(query, model.initial_state())[0]
    carried = model.chunk_logits(query, operational[3].clear_context())[0]
    changed_logits = float((old - carried).abs().max().detach())
    assert changed_logits > 1e-6
    checks["all_initialization_and_static_weights_unchanged"] = True
    checks["same_query_after_persistent_fast_updates_max_logit_change"] = changed_logits
    # This is a functional counterexample, not a measurement of forgetting a
    # previously acquired task: the model here is randomly initialized.

    # Central differences check the complete outer objective, both inactive and
    # active clipping. Perturb ALL parameters, including outer-only prefix ones.
    fd = []
    short = tokens[:7]
    for clip in (100.0, 0.001):
        checked = E2ECore(replace(cfg, clip=clip), seed=331, dtype=torch.float64)
        p = dict(checked.p)
        loss, _, _, _, work = checked.sequence(short, params=p)
        assert all((w["inner_grad_norm"] < clip) == (clip == 100.0) for w in work)
        grads = torch.autograd.grad(loss, tuple(p.values()))
        direction = {k: torch.randn(v.shape, generator=rng, dtype=v.dtype) for k, v in p.items()}
        norm = sum(v.square().sum() for v in direction.values()).sqrt()
        direction = {k: v / norm for k, v in direction.items()}
        analytic = sum((g * direction[k]).sum() for k, g in zip(p, grads)).item()
        eps = 1e-4
        estimates = []
        for delta in (eps, eps / 2):
            losses = []
            for sign in (-1, 1):
                params = {k: (v.detach() + sign * delta * direction[k]).requires_grad_(True) for k, v in p.items()}
                losses.append(checked.sequence(short, params=params)[0].item())
            estimates.append((losses[1] - losses[0]) / (2 * delta))
        error = abs(estimates[-1] - analytic)
        assert error < 2e-7, (clip, analytic, estimates)
        grad_fo = torch.autograd.grad(checked.sequence(short, mode="first_order")[0], tuple(checked.p.values()))
        fo_delta = sum((a - b).square().sum() for a, b in zip(grads, grad_fo)).sqrt().item()
        assert fo_delta > 1e-7
        fd.append({"clip": clip, "inner_norms": [w["inner_grad_norm"] for w in work],
                   "analytic_directional_derivative": analytic, "central_differences": estimates,
                   "absolute_error": error, "full_vs_first_order_gradient_l2": fo_delta})
    checks["finite_difference_meta_gradient"] = fd

    # Two adaptive suffix blocks exercise fast-dependent KV from earlier chunks.
    # Also differentiate across a real document boundary while clearing only KV.
    multi = E2ECore(replace(cfg, suffix=2, clip=100.0), seed=333, dtype=torch.float64)
    def continued_objective(params):
        a = multi.sequence(tokens[:5], params=params)
        b = multi.sequence(tokens[8:13], params=params, state=a[3].clear_context())
        return (a[0] + b[0]) / 2
    p = dict(multi.p)
    grads = torch.autograd.grad(continued_objective(p), tuple(p.values()))
    direction = {k: torch.randn(v.shape, generator=rng, dtype=v.dtype) for k, v in p.items()}
    norm = sum(v.square().sum() for v in direction.values()).sqrt()
    direction = {k: v / norm for k, v in direction.items()}
    analytic = sum((g * direction[k]).sum() for k, g in zip(p, grads)).item()
    eps = 5e-5
    minus_plus = [continued_objective({k: (v.detach() + sign * eps * direction[k]).requires_grad_(True)
                                      for k, v in p.items()}).item() for sign in (-1, 1)]
    estimate = (minus_plus[1] - minus_plus[0]) / (2 * eps)
    assert abs(analytic - estimate) < 2e-7, (analytic, estimate)
    multi_run = multi.sequence(tokens, detach_between=True)
    multi_prefix = multi.sequence(tokens[:9], detach_between=True)
    multi_cont = multi.sequence(tokens[8:], state=StreamState.restore(multi_prefix[3].payload()), detach_between=True)
    assert torch.equal(multi_run[2], torch.cat((multi_prefix[2], multi_cont[2])))
    assert_state_equal(multi_run[3], multi_cont[3])
    checks["two_suffix_blocks_persistent_meta_gradient"] = {
        "analytic": analytic, "central_difference": estimate, "absolute_error": abs(analytic - estimate),
        "fast_dependent_cache_serialization_exact": True,
    }

    # A real outer optimizer update reaches both static prefix and initialization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    model.sequence(tokens[:9])[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.p.values())
    optimizer.step()
    checks["outer_changes_prefix_and_fast_initialization"] = all(
        not torch.equal(saved[k], model.p[k]) for k in ("b0_q", model.fast_names[0])
    )
    assert checks["outer_changes_prefix_and_fast_initialization"]
    report = {
        "event": "E2E_CORE_PREFLIGHT_PASS", "upstream_commit": "a4fc4788ace38e29b5067916d4f4be33da894085",
        "scope": "Mechanism checks only; no upstream numerical replication, acquired-task retention or RSI result.",
        "model": model.specification(), "checks": checks, "seconds": time.perf_counter() - start,
        "runtime": {"torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()},
        "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("e2e_core.py", "preflight_e2e_core.py")},
    }
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    main(parser.parse_args().out)
