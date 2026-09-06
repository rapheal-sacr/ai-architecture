"""Audit saved E34 interventions; independent dense reference for frozen arms."""
import argparse
import collections
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from e2e_core import Config, E2ECore, StreamState
from preflight_e2e_core import dense_reference
from record_e33 import attention_work, exact_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(source, out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    source, out = Path(source), Path(out)
    manifest = json.loads((source / "manifest.json").read_text())
    full = json.loads((source / "complete.json").read_text())
    root = Path(__file__).resolve().parent
    assert full["event"] == ("E34_PREFLIGHT_COMPLETE" if manifest["preflight"] else "E34_COMPLETE")
    for name, expected in manifest["files"].items():
        assert sha(root / name) == expected
        if not manifest["preflight"]:
            relative = (root / name).relative_to(root.parent.parent)
            blob = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + str(relative)])
            assert hashlib.sha256(blob).hexdigest() == expected
    for path, expected in full["source_files_unchanged"].items():
        assert sha(path) == expected
    parent = Path(manifest["source"])
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    data = json.loads((parent / "data.json").read_text())
    cfg = Config(**parent_manifest["protocol"]["config"])
    groups, seen = collections.defaultdict(list), set()
    max_dense_diff = 0.0
    neural_replays = dense_replays = 0
    model = E2ECore(cfg, seed=0)
    dense_model = E2ECore(cfg, seed=0, dtype=torch.float64)
    cached = {}
    for row in full["rows"]:
        seed, arm, idx, treatment = row["seed"], row["arm"], row["stream_index"], row["treatment"]
        key = seed, arm, idx, treatment
        assert key not in seen
        seen.add(key)
        if (seed, arm) not in cached:
            d = parent / f"{seed}_{arm}"
            trained = torch.load(d / "trained.pt", weights_only=True)["model"]
            result = json.loads((d / "result.json").read_text())
            sources = {x["stream_index"]: x for x in result["evaluation"] if x["policy"] == "carry_ttt"}
            cached[seed, arm] = trained, sources
        trained, sources = cached[seed, arm]
        original = sources[idx]
        stream = data["evaluation"][str(seed)][idx]
        assert row["regime"] == stream["regime"]
        assert row["source_initial_A_acquired"] == original["sessions"][0]["last_cycle_acquired"]
        assert row["source_archive_checkpoint"] == original["sessions"][0]["checkpoint"]
        assert row["source_current_checkpoint"] == original["sessions"][-2]["checkpoint"]
        tokens = torch.tensor(stream["sessions"][-1]["tokens"])
        model.load_state_dict(trained)
        if treatment.startswith("current"):
            state = StreamState.restore(torch.load(original["sessions"][-2]["checkpoint"], weights_only=True)["state"]).clear_context()
        elif treatment.startswith("archive"):
            state = StreamState.restore(torch.load(original["sessions"][0]["checkpoint"], weights_only=True)["state"]).clear_context()
        else:
            assert treatment == "reset_frozen"
            state = model.initial_state().detached()
        snapshot = torch.load(row["checkpoint"], weights_only=True)
        assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
        logits, losses = snapshot["logits"], snapshot["losses"]
        frozen = treatment.endswith("frozen")
        if frozen:
            exact_tree({k: v.detach() for k, v in state.fast.items()}, snapshot["state"]["fast"])
            dense_model.load_state_dict(trained)
            with torch.no_grad():
                for k, v in state.fast.items():
                    dense_model.p[k].copy_(v.double())
                reference = dense_reference(dense_model, tokens[:-1])
            delta = (reference - logits.double()).abs().max().item()
            max_dense_diff = max(max_dense_diff, delta)
            torch.testing.assert_close(logits.double(), reference, atol=2e-4, rtol=1e-5)
            dense_replays += 1
        else:
            _, reference_losses, reference, final, _ = model.sequence(tokens, state=state, mode="first_order", detach_between=True)
            assert torch.equal(reference, logits) and torch.equal(reference_losses, losses)
            exact_tree(final.payload(), snapshot["state"])
            neural_replays += 1
        ce = logits.double().logsumexp(-1) - logits.double().gather(1, tokens[1:, None]).squeeze(1)
        torch.testing.assert_close(ce, losses.double(), atol=1e-6, rtol=1e-6)
        correct = logits.argmax(-1).eq(tokens[1:])
        for name, n in (("first_chunk", cfg.chunk), ("first_cycle", cfg.vocab), ("complete_return", len(tokens) - 1)):
            assert row[name]["n"] == n and row[name]["correct"] == int(correct[:n].sum())
            assert row[name]["mean_nll"] == float(losses[:n].mean())
        assert row["forward_tokens"] == len(tokens) - 1
        assert row["attention_score_elements"] == attention_work(cfg, len(tokens) - 1)
        assert row["inner_updates"] == (0 if frozen else (len(tokens) - 1) // cfg.chunk)
        groups[arm, stream["regime"], treatment].append(row)
    expected = {(int(seed), arm, idx, t) for seed, streams in data["evaluation"].items()
                for arm in parent_manifest["protocol"]["training_arms"] for idx in range(len(streams))
                for t in manifest["protocol"]["treatments"]}
    assert seen == expected
    summary = []
    for (arm, regime, treatment), rows in sorted(groups.items()):
        summary.append({"arm": arm, "regime": regime, "treatment": treatment, "streams": len(rows),
            "source_initial_A_acquired": sum(r["source_initial_A_acquired"] for r in rows),
            **{part + "_accuracy": sum(r[part]["correct"] for r in rows) / sum(r[part]["n"] for r in rows)
               for part in ("first_chunk", "first_cycle", "complete_return")},
            **{part + "_nll": sum(r[part]["mean_nll"] * r[part]["n"] for r in rows) / sum(r[part]["n"] for r in rows)
               for part in ("first_chunk", "first_cycle", "complete_return")},
            "seconds": sum(r["seconds"] for r in rows), "inner_updates": sum(r["inner_updates"] for r in rows)})
    report = {"event": "E34_PREFLIGHT_AUDITED" if manifest["preflight"] else "E34_AUDITED",
        "scored_commit": manifest["source_commit"], "rows_audited": len(seen), "adaptive_exact_replays": neural_replays,
        "frozen_independent_dense_replays": dense_replays, "dense_max_logit_difference": max_dense_diff,
        "summary": summary, "audit_seconds": time.perf_counter() - start,
        "scope": "Reused E33 streams, oracle archive choice. Frozen references use independent dense float64 attention against float32 streaming; not a new-task generalization or autonomous archive result."}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    main(a.source, a.out)
