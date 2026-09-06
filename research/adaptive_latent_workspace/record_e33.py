"""Reconstruct E33 data, every outer update and every operational checkpoint."""
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from e2e_core import Config, E2ECore, StreamState
from e33_persistent_e2e import digest, domain_seed, prepare_data, train_example


def exact_tree(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact_tree(x, y)
    else:
        assert a == b


def attention_work(c, n):
    old, result = 0, 0
    for _ in range(0, n, c.chunk):
        result += c.layers * c.heads * c.chunk * (old + c.chunk)
        old = min(c.window, old + c.chunk)
    return result


def main(source, out):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    completed = json.loads((source / "complete.json").read_text())
    protocol = manifest["protocol"]
    assert manifest["protocol_digest"] == digest(protocol)
    root = Path(__file__).resolve().parent
    for name, expected in manifest["sources"].items():
        current = (root / name).read_bytes()
        assert hashlib.sha256(current).hexdigest() == expected, name
        if not manifest["preflight"]:
            relative = (root / name).relative_to(root.parent.parent)
            blob = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + str(relative)])
            assert blob == current
    train, evaluation, data_audit = prepare_data(protocol)
    data = json.loads((source / "data.json").read_text())
    assert digest(data) == digest({"train": train, "evaluation": evaluation, "audit": data_audit})
    c = Config(**protocol["config"])
    # Independent transition validation. The label is used only in this auditor.
    for streams in list(train.values()) + list(evaluation.values()):
        for stream in streams:
            for mapping in stream["maps_for_evaluator_only"].values():
                assert sorted(mapping) == list(range(c.vocab))
                visited, node = set(), 0
                while node not in visited:
                    visited.add(node)
                    node = mapping[node]
                assert node == 0 and len(visited) == c.vocab
            for session in stream["sessions"]:
                mapping = stream["maps_for_evaluator_only"][session["label_for_evaluator_only"]]
                tokens = session["tokens"]
                assert len(tokens) == stream["cycles"] * c.vocab + 1
                assert all(mapping[x] == y for x, y in zip(tokens[:-1], tokens[1:]))
    rows, case_rows, replayed_updates, replayed_sessions = [], [], 0, 0
    expected_cases = {(s, a) for s in protocol["seeds"] for a in protocol["training_arms"]}
    assert {(r["seed"], r["arm"]) for r in completed["cases"]} == expected_cases
    for seed, arm in sorted(expected_cases):
        case_dir = source / f"{seed}_{arm}"
        result = json.loads((case_dir / "result.json").read_text())
        cp = torch.load(case_dir / "trained.pt", weights_only=True)
        model = E2ECore(c, seed=domain_seed(seed, "initialization"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["outer_lr"], weight_decay=protocol["outer_weight_decay"])
        assert len(result["training"]) == protocol["outer_steps"]
        for i, row in enumerate(result["training"]):
            stream = train[seed][i]
            assert row["stream_digest"] == digest(stream)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = train_example(model, stream, arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["outer_clip"])
            optimizer.step()
            assert loss.item() == row["loss"] and float(norm) == row["outer_grad_norm"]
            n = (len(stream["sessions"][0]["tokens"]) - 1)
            count = len(stream["sessions"])
            assert row["forward_tokens"] == count * n
            assert row["attention_score_elements"] == count * attention_work(c, n)
            assert row["inner_updates"] == (0 if arm == "static" else count * n // c.chunk)
            assert row["outer_updates"] == 1 and row["step"] == i + 1
            replayed_updates += 1
        exact_tree(model.state_dict(), cp["model"])
        exact_tree(optimizer.state_dict(), cp["optimizer"])
        state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        seen = set()
        for row in result["evaluation"]:
            policy, idx = row["policy"], row["stream_index"]
            assert (policy, idx) not in seen
            seen.add((policy, idx))
            stream = evaluation[seed][idx]
            assert row["stream_digest"] == digest(stream)
            state = None
            reconstructed = []
            for session, recorded in zip(stream["sessions"], row["sessions"], strict=True):
                state = model.initial_state() if state is None or policy != "carry_ttt" else state.clear_context()
                tokens = torch.tensor(session["tokens"])
                _, losses, logits, state, _ = model.sequence(tokens, state=state,
                    mode="frozen" if policy == "frozen" else "first_order", detach_between=True)
                snapshot = torch.load(recorded["checkpoint"], weights_only=True)
                assert torch.equal(logits, snapshot["logits"]) and torch.equal(losses, snapshot["losses"])
                exact_tree(state.payload(), snapshot["state"])
                # Independent log-probability calculation and deterministic ranking.
                alternate_ce = logits.double().logsumexp(-1) - logits.double().gather(1, tokens[1:, None]).squeeze(1)
                torch.testing.assert_close(losses.double(), alternate_ce, atol=1e-6, rtol=1e-6)
                preds = logits.argmax(-1)
                correct = preds.eq(tokens[1:])
                assert recorded["predictions"] == preds.tolist()
                assert recorded["correct"] == int(correct.sum()) and recorded["n"] == len(correct)
                assert math.isclose(recorded["mean_nll"], float(losses.mean()), abs_tol=1e-12)
                for j, cycle in enumerate(recorded["cycles"]):
                    part = slice(j * c.vocab, (j + 1) * c.vocab)
                    assert cycle["correct"] == int(correct[part].sum()) and cycle["n"] == c.vocab
                    assert cycle["mean_nll"] == float(losses[part].mean())
                acquired = float(correct[-c.vocab:].sum()) / c.vocab >= protocol["acquisition_cycle_accuracy"]
                assert acquired == recorded["last_cycle_acquired"]
                reconstructed.append(recorded)
                replayed_sessions += 1
            n = len(stream["sessions"][0]["tokens"]) - 1
            count = len(stream["sessions"])
            assert row["forward_tokens"] == n * count
            assert row["attention_score_elements"] == count * attention_work(c, n)
            assert row["inner_updates"] == (0 if policy == "frozen" else count * n // c.chunk)
            assert row["final_state_tensor_bytes"] == state.bytes()
            rows.append({"seed": seed, "arm": arm, "policy": policy, "regime": stream["regime"], "world": stream["index"],
                "first_context_acquired": reconstructed[0]["last_cycle_acquired"],
                "all_pre_return_contexts_acquired": all(r["last_cycle_acquired"] for r in reconstructed[:-1]),
                "initial_last_cycle_correct": reconstructed[0]["cycles"][-1]["correct"],
                "return_first_cycle_correct": reconstructed[-1]["cycles"][0]["correct"],
                "return_first_cycle_nll": reconstructed[-1]["cycles"][0]["mean_nll"],
                "return_last_cycle_correct": reconstructed[-1]["cycles"][-1]["correct"],
                "return_last_cycle_acquired": reconstructed[-1]["last_cycle_acquired"],
                "total_correct": sum(r["correct"] for r in reconstructed),
                "total_tokens": sum(r["n"] for r in reconstructed),
                "mean_nll": sum(r["mean_nll"] * r["n"] for r in reconstructed) / sum(r["n"] for r in reconstructed),
                "evaluation_seconds": row["seconds"]})
        assert seen == {(p, i) for p in protocol["evaluation_policies"][arm] for i in range(len(evaluation[seed]))}
        exact_tree(state_before, model.state_dict())
        case_rows.append({"seed": seed, "arm": arm, "outer_updates_replayed_exact": len(result["training"]),
                          "training_seconds": sum(r["seconds"] for r in result["training"]),
                          "training_forward_tokens": sum(r["forward_tokens"] for r in result["training"]),
                          "training_attention_elements": sum(r["attention_score_elements"] for r in result["training"]),
                          "inner_updates": sum(r["inner_updates"] for r in result["training"])})
        print(json.dumps({"event": "E33_CASE_AUDITED", "seed": seed, "arm": arm}), flush=True)
    controls = json.loads((source / "table-controls.json").read_text())
    for row in controls:
        stream = evaluation[row["seed"]][row["stream_index"]]
        assert row["stream_digest"] == digest(stream)
        # Reconstruct from the last matching actually observed input, without
        # using a table implementation or the hidden mapping/context labels.
        history = []
        for session, stored in zip(stream["sessions"], row["sessions"], strict=True):
            expected = []
            for x, y in zip(session["tokens"][:-1], session["tokens"][1:]):
                matches = [b for a, b in history if a == x]
                expected.append(matches[-1] if matches else 0)
                history.append((x, y))
            assert expected == stored["predictions"]
            truth = session["tokens"][1:]
            assert stored["cycle_correct"] == [sum(x == y for x, y in zip(expected[i:i+c.vocab], truth[i:i+c.vocab]))
                                               for i in range(0, len(truth), c.vocab)]
    report = {"event": "E33_PREFLIGHT_AUDITED" if manifest["preflight"] else "E33_AUDITED",
              "source_commit": manifest["source_commit"], "data_audit": data_audit,
              "outer_updates_replayed_exact": replayed_updates, "sessions_replayed_exact": replayed_sessions,
              "cases": case_rows, "rows": rows, "audit_seconds": time.perf_counter() - started,
              "scope": "Fixed small grammar assay; no large-paper replication, novelty, general reasoning or RSI inference."}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "rows")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    main(a.source, a.out)
