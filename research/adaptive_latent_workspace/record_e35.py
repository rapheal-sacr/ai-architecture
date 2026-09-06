"""E35 audit: full-core candidate replay plus independent data/events/metrics."""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from e2e_core import Config, E2ECore, StreamState
from e33_persistent_e2e import digest
from e35_bounded_archive import prepare_data, sha, source_maps
from record_e33 import attention_work, exact_tree
from suffix_archive import ArchiveConfig, SuffixArchive, copy_fast, model_fingerprint


class FullCoreKernel:
    """No shared suffix implementation. Recompute original core per candidate.

    The additional audit work is separate from scored inference work. The core
    has already passed an independent dense/finite-difference mechanism audit.
    """
    def __init__(self, model):
        self.model = model
        self.calls = []

    def features(self, tokens, state):
        self.calls = []
        with torch.no_grad():
            _, next_state, elements = self.model.chunk_logits(tokens, state)
        return (tokens, state.cache, state.position), next_state, elements

    def logits(self, features, fast):
        tokens, cache, position = features
        logits, _, _ = self.model.chunk_logits(tokens, StreamState(fast, cache, position))
        self.calls.append(logits.detach())
        return logits

    def updated(self, loss, fast):
        # Original E2E SGD, using gradients through complete original forwards.
        gradients = torch.autograd.grad(loss, tuple(fast.values()))
        norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
        multiplier = (self.model.cfg.clip / norm.clamp_min(1e-30)).clamp(max=1.)
        changes = [multiplier * g for g in gradients]
        next_fast = {k: (v-self.model.cfg.inner_lr*g).detach().clone().requires_grad_(True)
                     for (k, v), g in zip(fast.items(), changes)}
        return next_fast, float(norm.detach())


def independent_events(before, learner, result, calls, targets):
    """Derive decisions independently from full-core predictions and old state."""
    cfg = learner.config
    n = len(targets)
    candidate = []
    for entry, logits in zip(before["archives"], calls[1:1+len(before["archives"])]):
        loss = F.cross_entropy(logits, targets, reduction="none")
        candidate.append((float(loss.mean()), entry["id"], int(logits.argmax(-1).eq(targets).sum())))
    chosen, selected_loss, selected_correct = None, float(result["losses"].mean()), result["correct"]
    if cfg.select and candidate:
        best = sorted(candidate)[0]
        if best[0] + cfg.switch_margin < selected_loss:
            selected_loss, chosen, selected_correct = best
    assert chosen == result["selected_archive"]
    update = cfg.update_nll_threshold is None or selected_correct != n or selected_loss > cfg.update_nll_threshold
    assert update == result["updated"]
    assert selected_loss == result["selected_current_data_nll"]
    assert result["suffix_forwards"] == 1+len(before["archives"])+int(before["probe"] is not None)+int(chosen is not None and update)
    expected_entries = [{**e, "fast": copy_fast(e["fast"])} for e in before["archives"]]
    for e in expected_entries:
        if e["id"] == chosen:
            e["last_used"] = before["clock"]+1
    counts = dict(before["totals"])
    counts["switches"] += chosen is not None
    counts["prefix_tokens"] += n
    counts["attention_score_elements"] += result["attention_score_elements"]
    counts["suffix_forwards"] += result["suffix_forwards"]
    counts["suffix_tokens"] += n*result["suffix_forwards"]
    counts["gradient_updates"] += update
    counts["gradient_tokens"] += n*update
    probe = copy.deepcopy(before["probe"])
    if probe is not None:
        logits = calls[1+len(before["archives"])]
        pred = logits.argmax(-1)
        losses = F.cross_entropy(logits, targets, reduction="none")
        probe["chunks"] += 1
        probe["n"] += n
        probe["loss_sum"] += float(losses.sum())
        probe["correct"] += int(pred.eq(targets).sum())
        for e, candidate_logits in zip(before["archives"], calls[1:1+len(before["archives"])]):
            p = candidate_logits.argmax(-1)
            loss = F.cross_entropy(candidate_logits, targets, reduction="none")
            old = probe["comparisons"].get(e["id"], {"n": 0, "loss_sum": 0., "agree": 0, "correct": 0})
            probe["comparisons"][e["id"]] = {"n": old["n"]+n, "loss_sum": old["loss_sum"]+float(loss.sum()),
                "agree": old["agree"]+int(p.eq(pred).sum()), "correct": old["correct"]+int(p.eq(targets).sum())}
        if probe["chunks"] < cfg.probation_chunks:
            assert result["probe_event"] is None
            exact_tree(probe, learner.probe)
        else:
            event = {"born": probe["born"], "validated_through": before["clock"]+1, "n": probe["n"],
                     "correct": probe["correct"], "mean_nll": probe["loss_sum"]/probe["n"]}
            assert event["validated_through"]-event["born"] == cfg.probation_chunks
            eligible = [(v["loss_sum"]/v["n"], key) for key, v in probe["comparisons"].items()
                        if v["n"] == probe["n"] and v["agree"] == probe["n"] and v["correct"] >= cfg.admit_accuracy*probe["n"]]
            if event["correct"] < cfg.admit_accuracy*event["n"] or event["mean_nll"] > cfg.admit_nll:
                event["outcome"] = "reject"
                counts["rejections"] += 1
            elif eligible:
                selected = min(eligible)[1]
                event.update(outcome="duplicate", matched_id=selected)
                counts["duplicates"] += 1
                for e in expected_entries:
                    if e["id"] == selected:
                        e["last_used"] = before["clock"]+1
            else:
                evicted = None
                if len(expected_entries) >= cfg.capacity:
                    old = sorted(expected_entries, key=lambda e: (e["last_used"], e["admitted"], e["id"]))[0]
                    evicted = old["id"]
                    expected_entries = [e for e in expected_entries if e["id"] != evicted]
                    counts["evictions"] += 1
                expected_entries.append({"id": before["next_id"], "fast": copy_fast(probe["fast"]),
                                        "admitted": before["clock"]+1, "last_used": before["clock"]+1})
                event.update(outcome="admit", admitted_id=before["next_id"], evicted_id=evicted)
                counts["admissions"] += 1
            exact_tree(event, result["probe_event"])
            exact_tree(learner.probe["fast"], learner.state.fast)
            assert learner.probe["born"] == before["clock"]+1 and learner.probe["n"] == learner.probe["chunks"] == 0
    else:
        assert learner.probe is None and result["probe_event"] is None
    exact_tree(expected_entries, learner.archives)
    assert learner.next_id == counts["admissions"]
    exact_tree(counts, learner.totals)
    assert learner.clock == before["clock"]+1
    assert learner.state.position == before["state"]["position"]+n


def audit_metrics(logits, losses, stream, recorded, protocol):
    targets = torch.tensor(stream["tokens"][1:])
    correct = logits.argmax(-1).eq(targets)
    # Independent logsumexp CE in double precision, then exact sums of the
    # independently replayed float32 per-token CE used by the scorer.
    alternative = logits.double().logsumexp(-1)-logits.double()[torch.arange(len(targets)), targets]
    torch.testing.assert_close(losses.double(), alternative, atol=2e-6, rtol=1e-6)
    def check_range(row, a, b):
        assert row["n"] == b-a and row["correct"] == int(correct[a:b].sum())
        assert row["loss_sum"] == float(losses[a:b].double().sum())
    check_range(recorded["all"], 0, len(targets))
    check_range(recorded["latter_half"], len(targets)//2, len(targets))
    history, eligible = {}, []
    assert len(recorded["sessions"]) == len(stream["segments_evaluator_only"])
    for segment, row in zip(stream["segments_evaluator_only"], recorded["sessions"]):
        a, b, key = segment["start"], segment["end"], digest(segment["map_evaluator_only"])
        assert (row["start"], row["end"], row["map_digest"]) == (a, b, key)
        assert row["recurrence"] == (key in history)
        assert row["previous_exact_map_acquired"] == history.get(key, False)
        window = protocol["acquisition_window"]
        check_range(row["all"], a, b)
        check_range(row["first_window"], a, min(b, a+window))
        check_range(row["last_window"], max(a, b-window), b)
        if history.get(key, False):
            eligible.append(row["first_window"])
        acquired = b-a >= window and int(correct[b-window:b].sum()) >= protocol["acquisition_accuracy"]*window
        assert row["last_window_acquired"] == acquired
        history[key] = acquired
    expected = {"occurrences": len(eligible), **{k: sum(e[k] for e in eligible) for k in ("n", "correct", "loss_sum")}}
    exact_tree(expected, recorded["acquired_return_first_window"])


def main(source, out):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(source)
    manifest = json.loads((source/"manifest.json").read_text())
    complete = json.loads((source/"complete.json").read_text())
    p = manifest["protocol"]
    root = Path(__file__).resolve().parent
    assert sha(source/"manifest.json") == complete["manifest_sha256"]
    assert sha(source/"data.json") == complete["data_sha256"]
    for name, expected in manifest["files"].items():
        assert sha(root/name) == expected, name
        if not manifest["preflight"]:
            rel = (root/name).relative_to(root.parent.parent)
            blob = subprocess.check_output(["git", "show", manifest["source_commit"]+":"+str(rel)])
            assert blob == (root/name).read_bytes()
    for path, expected in manifest["source_files"].items():
        assert sha(path) == expected, path
    cfg = Config(**manifest["config"])
    forbidden = source_maps(json.loads(Path(manifest["source_data"]).read_text()))
    data = json.loads((source/"data.json").read_text())
    assert digest(prepare_data(p, cfg.vocab, forbidden)) == digest(data)
    # Independent map/cycle/continuous-boundary validation, without generator.
    for stream in data["streams"]:
        previous_end = 0
        for segment in stream["segments_evaluator_only"]:
            a, b, mapping = segment["start"], segment["end"], segment["map_evaluator_only"]
            assert a == previous_end and digest(mapping) == segment["map_digest"]
            assert sorted(mapping) == list(range(cfg.vocab))
            node, visited = 0, set()
            while node not in visited:
                visited.add(node)
                node = mapping[node]
            assert node == 0 and len(visited) == cfg.vocab
            assert all(mapping[stream["tokens"][i]] == stream["tokens"][i+1] for i in range(a, b))
            previous_end = b
        assert previous_end == p["targets_per_stream"]
    assert len(complete["tables"]) == len(data["streams"])
    for table, stream in zip(complete["tables"], data["streams"]):
        tokens, memory, predictions = stream["tokens"], [-1]*cfg.vocab, []
        for i in range(0, len(tokens)-1, cfg.chunk):
            for x in tokens[i:i+cfg.chunk]:
                predictions.append(0 if memory[x] < 0 else memory[x])
            for j in range(cfg.chunk):
                memory[tokens[i+j]] = tokens[i+j+1]
        assert predictions == table["predictions"]
        assert table["correct"] == sum(x == y for x, y in zip(predictions, tokens[1:]))
        assert table["logical_int64_key_value_bytes"] == 16*sum(x >= 0 for x in memory)
    expected = {(i, a, s) for i in range(len(data["streams"])) for a in p["source_arms"] for s in p["policies"]}
    assert {(r["stream_index"], r["arm"], r["policy"]) for r in complete["rows"]} == expected
    assert len(complete["rows"]) == len(expected)
    checkpoints, updates, chunks, candidate_forwards = 0, 0, 0, 0
    cases = []
    for row in complete["rows"]:
        stream = data["streams"][row["stream_index"]]
        assert row["seed"] == stream["seed"] and row["regime"] == stream["regime"]
        model = E2ECore(cfg, seed=row["seed"])
        if not manifest["preflight"]:
            model.load_state_dict(torch.load(manifest["model_paths"][f'{row["seed"]}_{row["arm"]}'], weights_only=True)["model"])
        identity = model_fingerprint(model)
        policy = ArchiveConfig(**p["policy_defaults"], **p["policies"][row["policy"]])
        learner = SuffixArchive(model, policy)
        learner.kernel = FullCoreKernel(model)
        resource = row["resource"]
        assert sha(resource["output"]) == resource["output_sha256"]
        assert sha(resource["trace"]) == resource["trace_sha256"]
        output = torch.load(resource["output"], weights_only=True)
        trace = json.loads(Path(resource["trace"]).read_text())
        cp_by_chunk = {c["chunk"]: c for c in resource["checkpoints"]}
        expected_cp = set(range(p["checkpoint_chunks"], p["targets_per_stream"]//cfg.chunk+1, p["checkpoint_chunks"])) | {p["targets_per_stream"]//cfg.chunk}
        assert set(cp_by_chunk) == expected_cp
        tokens = torch.tensor(stream["tokens"])
        peak = sum(learner.tensor_bytes().values())
        assert len(trace) == (len(tokens)-1)//cfg.chunk
        for i in range(0, len(tokens)-1, cfg.chunk):
            before = learner.payload()
            result = learner.step(tokens[i:i+cfg.chunk], tokens[i+1:i+cfg.chunk+1])
            independent_events(before, learner, result, learner.kernel.calls, tokens[i+1:i+cfg.chunk+1])
            assert torch.equal(result["logits"], output["logits"][i:i+cfg.chunk])
            assert torch.equal(result["losses"], output["losses"][i:i+cfg.chunk])
            candidate_forwards += result["suffix_forwards"]
            updates += result["updated"]
            del result["logits"], result["losses"]
            result.update(chunk=learner.clock, position=learner.state.position,
                          archive_ids=[e["id"] for e in learner.archives], tensor_bytes=learner.tensor_bytes())
            exact_tree(result, trace[i//cfg.chunk])
            peak = max(peak, sum(learner.tensor_bytes().values()))
            if learner.clock in cp_by_chunk:
                checkpoint = cp_by_chunk[learner.clock]
                assert sha(checkpoint["path"]) == checkpoint["sha256"]
                stored = torch.load(checkpoint["path"], weights_only=True)
                exact_tree(stored, learner.payload())
                checkpoints += 1
            chunks += 1
        exact_tree(learner.payload(), output["final"])
        exact_tree(learner.totals, resource["totals"])
        exact_tree(learner.tensor_bytes(), resource["final_tensor_bytes"])
        assert peak == resource["peak_persistent_tensor_bytes"]
        assert model_fingerprint(model) == identity
        assert resource["base_parameter_bytes"] == model.specification()["parameter_bytes"]
        assert learner.totals["attention_score_elements"] == attention_work(cfg, len(tokens)-1)
        audit_metrics(output["logits"], output["losses"], stream, row["metrics"], p)
        if row["policy"] == "bank8_no_select_gated":
            reference_row = next(r for r in complete["rows"] if r["stream_index"] == row["stream_index"] and r["arm"] == row["arm"] and r["policy"] == "active_gated")
            reference = torch.load(reference_row["resource"]["output"], weights_only=True)
            exact_tree(output["logits"], reference["logits"])
            exact_tree(output["losses"], reference["losses"])
            exact_tree(output["final"]["state"], reference["final"]["state"])
        cases.append({k: row[k] for k in ("seed", "arm", "regime", "policy", "stream_index")})
        print(json.dumps({"event": "E35_CASE_AUDITED", **cases[-1]}), flush=True)
    result = {"event": "E35_PREFLIGHT_AUDITED" if manifest["preflight"] else "E35_AUDITED",
        "cases": cases, "checkpoints_exact": checkpoints, "chunks_exact": chunks,
        "gradient_updates_exact": updates, "full_core_suffix_candidate_forwards_exact": candidate_forwards,
        "independent_policy_events": True, "independent_metric_and_transition_checks": True,
        "source_and_base_unchanged": True, "no_select_exact_control": True,
        "data_sha256": complete["data_sha256"], "complete_sha256": sha(source/"complete.json"),
        "manifest_sha256": complete["manifest_sha256"], "seconds": time.perf_counter()-started}
    Path(out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.source, args.out)
