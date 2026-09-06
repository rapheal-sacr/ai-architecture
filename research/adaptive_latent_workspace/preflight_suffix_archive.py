"""Mechanism checks only: random fixtures are not acquired task competence."""
import argparse
import copy
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from e2e_core import Config, E2ECore, StreamState
from record_e33 import exact_tree
from suffix_archive import ArchiveConfig, SharedSuffix, SuffixArchive, copy_fast, model_fingerprint


def main(out):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(vocab=12, width=16, hidden=24, heads=2, window=4, chunk=4,
                 inner_lr=.3, init_std=.2)
    checks = {}
    for dtype in (torch.float32, torch.float64):
        model = E2ECore(cfg, seed=350, dtype=dtype)
        identity = model_fingerprint(model)
        rng = torch.Generator().manual_seed(351)
        tokens = torch.randint(cfg.vocab, (129,), generator=rng)
        learner = SuffixArchive(model, ArchiveConfig())
        standard = model.initial_state().detached()
        for i in range(0, 128, 4):
            inputs, targets = tokens[i:i+4], tokens[i+1:i+5]
            actual = learner.step(inputs, targets)
            _, loss, logits, standard, work = model.step(inputs, targets, standard, mode="first_order")
            standard = standard.detached()
            assert torch.equal(actual["logits"], logits) and torch.equal(actual["losses"], loss)
            exact_tree(learner.state.payload(), standard.payload())
            assert actual["grad_norm"] == work["inner_grad_norm"]
            assert actual["attention_score_elements"] == work["attention_score_elements"]
        checks[str(dtype) + "_32_updates_exact_standard_core"] = True
        # Arbitrary candidate fast states cannot change any attention cache for
        # this one-final-suffix architecture. Compare complete candidate paths.
        kernel = SharedSuffix(model)
        features, shared_next, _ = kernel.features(tokens[:4], standard)
        for k in range(4):
            fast = {key: (value.detach() + torch.randn(value.shape, generator=rng, dtype=dtype) * .2).requires_grad_(True)
                    for key, value in standard.fast.items()}
            logits = kernel.logits(features, fast)
            full, full_next, _ = model.chunk_logits(tokens[:4], StreamState(fast, standard.cache, standard.position))
            assert torch.equal(logits, full)
            exact_tree(shared_next.payload()["cache"], full_next.payload()["cache"])
            updated, norm = kernel.updated(F.cross_entropy(logits, tokens[1:5], reduction="none").mean(), fast)
            _, _, _, next_full, work = model.step(tokens[:4], tokens[1:5], StreamState(fast, standard.cache, standard.position), mode="first_order")
            exact_tree(updated, next_full.fast)
            assert norm == work["inner_grad_norm"]
        checks[str(dtype) + "_4_arbitrary_candidates_logits_gradients_caches_exact"] = True
        assert model_fingerprint(model) == identity

    model = E2ECore(cfg, seed=352)
    identity = model_fingerprint(model)
    # Lenient fixture admission is deliberately not the scored task threshold.
    config = ArchiveConfig(capacity=2, update_nll_threshold=.5, probation_chunks=2,
                           admit_accuracy=0, admit_nll=100)
    a = SuffixArchive(model, config)
    chunks, admissions, evictions = [], 0, 0
    for i in range(0, 64, 4):
        before = {e["id"]: copy_fast(e["fast"]) for e in a.archives}
        result = a.step(tokens[i:i+4], tokens[i+1:i+5])
        chunks.append(result)
        for e in a.archives:
            if e["id"] in before:
                exact_tree(before[e["id"]], e["fast"])
        if result["probe_event"]:
            event = result["probe_event"]
            assert event["validated_through"] - event["born"] == 2
            admissions += event["outcome"] == "admit"
            evictions += event.get("evicted_id") is not None
        assert len(a.archives) <= 2
    assert admissions > 2 and evictions > 0
    checks["fresh_probation_immutable_snapshots_capacity_eviction"] = {"admissions": admissions, "evictions": evictions}
    buf = io.BytesIO()
    torch.save(a.payload(), buf)
    buf.seek(0)
    b = SuffixArchive(model, config)
    b.restore(torch.load(buf, weights_only=True))
    for i in range(64, 128, 4):
        exact_tree(a.step(tokens[i:i+4], tokens[i+1:i+5]), b.step(tokens[i:i+4], tokens[i+1:i+5]))
        exact_tree(a.payload(), b.payload())
    checks["serialized_archive_probe_cache_continuation_exact_16_chunks"] = True
    assert a.totals["switches"] > 0
    checks["actual_archive_switches_exercised"] = a.totals["switches"]

    # Target changes must not change current operational predictions. They may
    # change archive choice and updates that will affect subsequent chunks.
    b.restore(a.payload())
    x, y = tokens[:4], tokens[1:5]
    original = a.step(x, y)
    altered = b.step(x, (y + 1) % cfg.vocab)
    assert torch.equal(original["logits"], altered["logits"])
    checks["target_intervention_no_retrospective_prediction_selection"] = True

    # A populated but disabled archive must be exactly the active gated learner.
    baseline = SuffixArchive(model, ArchiveConfig(update_nll_threshold=.5))
    disabled = SuffixArchive(model, replace(config, select=False))
    for i in range(0, 128, 4):
        left = baseline.step(tokens[i:i+4], tokens[i+1:i+5])
        right = disabled.step(tokens[i:i+4], tokens[i+1:i+5])
        assert torch.equal(left["logits"], right["logits"])
        exact_tree(baseline.state.payload(), disabled.state.payload())
    assert disabled.totals["admissions"] > 0 and disabled.totals["switches"] == 0
    assert disabled.totals["suffix_tokens"] > baseline.totals["suffix_tokens"]
    checks["disabled_selection_exact_gated_active_with_extra_charged_work"] = True
    skip = SuffixArchive(model, ArchiveConfig(update_nll_threshold=100))
    logits = model.chunk_logits(tokens[:4], skip.state)[0]
    old_fast = copy_fast(skip.state.fast)
    result = skip.step(tokens[:4], logits.argmax(-1))
    assert not result["updated"] and skip.totals["gradient_updates"] == 0
    exact_tree(old_fast, skip.state.fast)
    checks["correct_low_loss_chunk_skips_update_exact"] = True

    # Exercise the independent event auditor on populated/evicting fixture banks;
    # untrained scored-threshold preflight streams alone mostly reject probes.
    from record_e35 import FullCoreKernel, independent_events
    shared = SuffixArchive(model, config)
    reference = SuffixArchive(model, config)
    reference.kernel = FullCoreKernel(model)
    for i in range(0, 128, 4):
        old = reference.payload()
        expected = reference.step(tokens[i:i+4], tokens[i+1:i+5])
        independent_events(old, reference, expected, reference.kernel.calls, tokens[i+1:i+5])
        actual = shared.step(tokens[i:i+4], tokens[i+1:i+5])
        exact_tree(actual, expected)
        exact_tree(shared.payload(), reference.payload())
    assert reference.totals["switches"] > 0 and reference.totals["evictions"] > 0
    checks["full_core_and_independent_policy_audit_populated_fixture"] = reference.totals

    mismatch = E2ECore(cfg, seed=353)
    try:
        SuffixArchive(mismatch, config).restore(a.payload())
    except ValueError:
        checks["changed_encoder_or_initialization_rejected"] = True
    else:
        raise AssertionError("Changed model accepted")
    try:
        SuffixArchive(model, replace(config, capacity=3)).restore(a.payload())
    except ValueError:
        checks["changed_policy_rejected"] = True
    else:
        raise AssertionError("Changed policy accepted")
    try:
        SharedSuffix(E2ECore(replace(cfg, suffix=2), seed=354))
    except ValueError:
        checks["multiple_adaptive_layers_rejected"] = True
    else:
        raise AssertionError("Invalid sharing accepted")
    assert model_fingerprint(model) == identity
    checks["static_and_initialization_weights_unchanged"] = True
    checks["tensor_bytes_at_capacity"] = a.tensor_bytes()
    result = {"event": "SUFFIX_ARCHIVE_PREFLIGHT_PASSED", "checks": checks,
              "seconds": time.perf_counter() - started, "scope": "Mechanism fixtures, no competence or efficiency claim"}
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    main(parser.parse_args().out)
