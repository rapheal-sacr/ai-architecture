"""Post hoc, source-frozen intervention on all completed E33 return states."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from e2e_core import Config, E2ECore, StreamState
from record_e33 import attention_work, exact_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(logits, losses, tokens):
    correct = logits.argmax(-1).eq(tokens[1:])
    alternate = logits.double().logsumexp(-1) - logits.double().gather(1, tokens[1:, None]).squeeze(1)
    torch.testing.assert_close(losses.double(), alternate, atol=1e-6, rtol=1e-6)
    return {"n": len(correct), "correct": int(correct.sum()), "mean_nll": float(losses.mean())}


def main(source, audit_path, out, preflight):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    source, audit_path, out = Path(source), Path(audit_path), Path(out)
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol_e34.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    assert preflight == source_manifest["preflight"]
    assert audit["event"] == ("E33_PREFLIGHT_AUDITED" if preflight else "E33_AUDITED")
    if not preflight:
        assert source_manifest["source_commit"] == protocol["source_commit"]
        assert audit["outer_updates_replayed_exact"] == 1152 and audit["sessions_replayed_exact"] == 1008
    files = ("e2e_core.py", "e34_frozen_return.py", "protocol_e34.json", "record_e33.py")
    if not preflight:
        for name in files:
            subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", str(root / name)], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "ls-files", "--error-unmatch", str(root / name)], check=True, stdout=subprocess.PIPE)
    assert not out.exists() or not any(out.iterdir()), "Never overwrite a previous or running diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((source / "data.json").read_text())
    cfg = Config(**source_manifest["protocol"]["config"])
    manifest = {"source": str(source), "source_audit_sha256": sha(audit_path), "source_data_sha256": sha(source / "data.json"),
        "preflight": preflight, "protocol": protocol, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": {name: sha(root / name) for name in files}, "runtime": {"device": "cpu", "threads": 1, "torch": torch.__version__}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    tracked_sources = {str(audit_path): sha(audit_path), str(source / "data.json"): sha(source / "data.json")}
    for case in audit["cases"]:
        seed, arm = case["seed"], case["arm"]
        case_dir = source / f"{seed}_{arm}"
        trained_path = case_dir / "trained.pt"
        result_path = case_dir / "result.json"
        tracked_sources.update({str(p): sha(p) for p in (trained_path, result_path)})
        trained = torch.load(trained_path, weights_only=True)
        model = E2ECore(cfg, seed=0)
        model.load_state_dict(trained["model"])
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        source_result = json.loads(result_path.read_text())
        selected = [r for r in source_result["evaluation"] if r["policy"] == "carry_ttt"]
        assert len(selected) == len(data["evaluation"][str(seed)])
        for entry in selected:
            index = entry["stream_index"]
            stream = data["evaluation"][str(seed)][index]
            tokens = torch.tensor(stream["sessions"][-1]["tokens"])
            assert stream["schedule"][0] == stream["schedule"][-1] == "A"
            current_path = Path(entry["sessions"][-2]["checkpoint"])
            archive_path = Path(entry["sessions"][0]["checkpoint"])
            reference_path = Path(entry["sessions"][-1]["checkpoint"])
            tracked_sources.update({str(p): sha(p) for p in (current_path, archive_path, reference_path)})
            current = torch.load(current_path, weights_only=True)["state"]
            archive = torch.load(archive_path, weights_only=True)["state"]
            reference = torch.load(reference_path, weights_only=True)
            outputs = {}
            for treatment in protocol["treatments"]:
                if treatment.startswith("current"):
                    state = StreamState.restore(current).clear_context()
                elif treatment.startswith("archive"):
                    state = StreamState.restore(archive).clear_context()
                else:
                    state = model.initial_state().detached()
                initial_fast = {k: v.detach().clone() for k, v in state.fast.items()}
                frozen = treatment.endswith("frozen")
                tick = time.perf_counter()
                _, losses, logits, final, work = model.sequence(tokens, state=state,
                            mode="frozen" if frozen else "first_order", detach_between=True)
                seconds = time.perf_counter() - tick
                if frozen:
                    exact_tree(initial_fast, {k: v.detach() for k, v in final.fast.items()})
                if treatment == "current_adaptive":
                    assert torch.equal(logits, reference["logits"])
                    assert torch.equal(losses, reference["losses"])
                    exact_tree(final.payload(), reference["state"])
                assert sum(w["forward_tokens"] for w in work) == len(tokens) - 1
                assert sum(w["attention_score_elements"] for w in work) == attention_work(cfg, len(tokens) - 1)
                path = out / f"{seed}_{arm}_{index}_{treatment}.pt"
                torch.save({"logits": logits, "losses": losses, "state": final.payload()}, path)
                row = {"seed": seed, "arm": arm, "stream_index": index, "regime": stream["regime"], "treatment": treatment,
                    "source_initial_A_acquired": entry["sessions"][0]["last_cycle_acquired"],
                    "first_chunk": metric(logits[:cfg.chunk], losses[:cfg.chunk], tokens[:cfg.chunk+1]),
                    "first_cycle": metric(logits[:cfg.vocab], losses[:cfg.vocab], tokens[:cfg.vocab+1]),
                    "complete_return": metric(logits, losses, tokens), "seconds": seconds,
                    "forward_tokens": sum(w["forward_tokens"] for w in work),
                    "attention_score_elements": sum(w["attention_score_elements"] for w in work),
                    "inner_updates": sum(w["inner_updates"] for w in work),
                    "fast_state_tensor_bytes": sum(v.numel() * v.element_size() for v in initial_fast.values()),
                    "source_current_checkpoint": str(current_path), "source_archive_checkpoint": str(archive_path),
                    "checkpoint": str(path), "checkpoint_sha256": sha(path)}
                rows.append(row)
                outputs[treatment] = logits
            assert torch.equal(outputs["current_adaptive"][:cfg.chunk], outputs["current_frozen"][:cfg.chunk])
            assert torch.equal(outputs["archive_adaptive"][:cfg.chunk], outputs["archive_frozen"][:cfg.chunk])
        exact_tree(before, model.state_dict())
        print(json.dumps({"event": "E34_CASE", "seed": seed, "arm": arm}), flush=True)
    assert all(sha(path) == expected for path, expected in tracked_sources.items())
    expected = sum(len(v) for v in data["evaluation"].values()) * len(source_manifest["protocol"]["training_arms"]) * len(protocol["treatments"])
    assert len(rows) == expected
    result = {"event": "E34_PREFLIGHT_COMPLETE" if preflight else "E34_COMPLETE", "rows": rows,
        "source_files_unchanged": tracked_sources, "current_adaptive_exact_source_replay": True,
        "frozen_states_unchanged": True, "pre_update_chunk_pairs_exact": True,
        "seconds": time.perf_counter() - start, "scope": "Post hoc counterfactual source-state diagnosis; archive selection is oracle scaffolding; no new generalization or autonomous memory claim."}
    (out / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "source_files_unchanged")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.source, args.audit, args.out, args.preflight)
