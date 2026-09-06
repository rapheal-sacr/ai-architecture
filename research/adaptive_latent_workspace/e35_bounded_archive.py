"""Frozen E35 assay: continuous observations, bounded E2E suffix archive."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import torch

from e2e_core import Config, E2ECore
from e33_persistent_e2e import digest
from record_e33 import attention_work, exact_tree
from suffix_archive import ArchiveConfig, SuffixArchive, model_fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_maps(data):
    return {digest(mapping) for domain in ("train", "evaluation") for streams in data[domain].values()
            for stream in streams for mapping in stream["maps_for_evaluator_only"].values()}


def cycle_map(cycle):
    mapping = [0] * len(cycle)
    for x, y in zip(cycle, cycle[1:] + cycle[:1]):
        mapping[x] = y
    return mapping


def prepare_data(protocol, vocab, forbidden):
    streams, globally_seen = [], set()
    for seed in protocol["source_seeds"]:
        for regime in protocol["regimes"]:
            for replicate in range(protocol["replicates_per_regime_seed"]):
                domain = f'E35|{seed}|{regime["name"]}|{replicate}'
                rng = random.Random(int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"))
                def fresh():
                    cycle = list(range(vocab))
                    rng.shuffle(cycle)
                    return cycle
                cycles = [fresh() for _ in range(regime["initial_contexts"])]
                local = {digest(cycle_map(c)) for c in cycles}
                assert len(local) == len(cycles)
                tokens, segments, previous = [rng.randrange(vocab)], [], None
                while len(tokens) - 1 < protocol["targets_per_stream"]:
                    if previous is not None and len(cycles) < regime["max_contexts"] and rng.random() < regime["new_probability"]:
                        slot = len(cycles)
                        cycles.append(fresh())
                    else:
                        slot = rng.choice([s for s in range(len(cycles)) if s != previous])
                    if previous is not None and rng.random() < regime["mutate_probability"]:
                        a, b = rng.sample(range(vocab), 2)
                        cycles[slot][a], cycles[slot][b] = cycles[slot][b], cycles[slot][a]
                    mapping = cycle_map(cycles[slot])
                    local.add(digest(mapping))
                    start = len(tokens) - 1
                    end = min(protocol["targets_per_stream"], start + rng.randint(regime["min_duration"], regime["max_duration"]))
                    for _ in range(start, end):
                        tokens.append(mapping[tokens[-1]])
                    segments.append({"start": start, "end": end, "slot_evaluator_only": slot,
                                     "map_evaluator_only": mapping, "map_digest": digest(mapping)})
                    previous = slot
                assert not local.intersection(forbidden), "E33 map overlap"
                assert not local.intersection(globally_seen), "Cross-stream map overlap"
                globally_seen.update(local)
                streams.append({"seed": seed, "regime": regime["name"], "replicate": replicate,
                                "domain": domain, "tokens": tokens, "segments_evaluator_only": segments,
                                "distinct_maps_including_initial_slots": len(local)})
    return {"streams": streams, "distinct_maps": len(globally_seen), "source_maps_disjoint": True,
            "cross_stream_maps_disjoint": True, "source_distinct_maps": len(forbidden)}


def summarize(logits, losses, stream, protocol):
    targets = torch.tensor(stream["tokens"][1:])
    correct = logits.argmax(-1).eq(targets)
    def measure(start, end):
        return {"n": end-start, "correct": int(correct[start:end].sum()),
                "loss_sum": float(losses[start:end].double().sum())}
    previous, sessions = {}, []
    window = protocol["acquisition_window"]
    for segment in stream["segments_evaluator_only"]:
        a, b, key = segment["start"], segment["end"], segment["map_digest"]
        last = measure(max(a, b-window), b)
        acquired = last["n"] == window and last["correct"] / window >= protocol["acquisition_accuracy"]
        sessions.append({"start": a, "end": b, "map_digest": key,
            "recurrence": key in previous, "previous_exact_map_acquired": previous.get(key, False),
            "first_window": measure(a, min(b, a+window)), "last_window": last,
            "last_window_acquired": acquired, "all": measure(a, b)})
        previous[key] = acquired
    eligible = [s["first_window"] for s in sessions if s["previous_exact_map_acquired"]]
    return {"all": measure(0, len(targets)), "latter_half": measure(len(targets)//2, len(targets)),
            "sessions": sessions, "acquired_return_first_window": {"occurrences": len(eligible),
              **{k: sum(e[k] for e in eligible) for k in ("n", "correct", "loss_sum")}}}


def table_control(stream, vocab, chunk):
    # Same chunk delay as the neural learner. No hidden map or boundary inputs.
    started = time.perf_counter()
    memory, predictions = {}, []
    tokens = stream["tokens"]
    for i in range(0, len(tokens)-1, chunk):
        predictions.extend(memory.get(x, 0) for x in tokens[i:i+chunk])
        for x, y in zip(tokens[i:i+chunk], tokens[i+1:i+chunk+1]):
            memory[x] = y
    return {"predictions": predictions, "correct": sum(p == y for p, y in zip(predictions, tokens[1:])),
            "n": len(predictions), "logical_int64_key_value_bytes": len(memory)*16,
            "seconds": time.perf_counter()-started, "scope": "Specialized task representation; Python overhead additional"}


def evaluate(model, policy, tokens, directory, checkpoint_chunks):
    directory.mkdir(parents=True, exist_ok=False)
    learner = SuffixArchive(model, policy)
    initial_identity = model_fingerprint(model)
    all_logits, all_losses, trace, checkpoints = [], [], [], []
    peak = sum(learner.tensor_bytes().values())
    compute_seconds, checkpoint_seconds = 0., 0.
    total_started = time.perf_counter()
    for i in range(0, len(tokens)-1, model.cfg.chunk):
        tick = time.perf_counter()
        result = learner.step(tokens[i:i+model.cfg.chunk], tokens[i+1:i+model.cfg.chunk+1])
        compute_seconds += time.perf_counter()-tick
        all_logits.append(result.pop("logits"))
        all_losses.append(result.pop("losses"))
        result.update(chunk=learner.clock, position=learner.state.position,
                      archive_ids=[e["id"] for e in learner.archives], tensor_bytes=learner.tensor_bytes())
        trace.append(result)
        peak = max(peak, sum(learner.tensor_bytes().values()))
        if learner.clock % checkpoint_chunks == 0 or i+model.cfg.chunk == len(tokens)-1:
            tick = time.perf_counter()
            path = directory / f"state_{learner.clock}.pt"
            torch.save(learner.payload(), path)
            stored = torch.load(path, weights_only=True)
            resumed = SuffixArchive(model, policy)
            resumed.restore(stored)
            exact_tree(learner.payload(), resumed.payload())
            learner = resumed
            checkpoints.append({"chunk": learner.clock, "path": str(path), "sha256": sha(path)})
            checkpoint_seconds += time.perf_counter()-tick
    logits, losses = torch.cat(all_logits), torch.cat(all_losses)
    assert model_fingerprint(model) == initial_identity
    assert learner.totals["prefix_tokens"] == len(tokens)-1
    assert learner.totals["attention_score_elements"] == attention_work(model.cfg, len(tokens)-1)
    output = directory / "outputs.pt"
    torch.save({"logits": logits, "losses": losses, "final": learner.payload()}, output)
    trace_path = directory / "trace.json"
    trace_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
    resource = {"compute_seconds": compute_seconds, "checkpoint_seconds": checkpoint_seconds,
        "total_seconds": time.perf_counter()-total_started, "totals": learner.totals,
        "peak_persistent_tensor_bytes": peak, "final_tensor_bytes": learner.tensor_bytes(),
        "base_parameter_bytes": model.specification()["parameter_bytes"],
        "output": str(output), "output_sha256": sha(output), "trace": str(trace_path), "trace_sha256": sha(trace_path),
        "checkpoints": checkpoints}
    return logits, losses, resource


def main(source, audit, out, preflight):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, source, audit, out = Path(__file__).resolve().parent, Path(source), Path(audit), Path(out)
    protocol = json.loads((root / "protocol_e35.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_audit = json.loads(audit.read_text())
    assert source_manifest["source_commit"] == protocol["source_commit"]
    assert source_audit["event"] == "E33_AUDITED" and source_audit["outer_updates_replayed_exact"] == 1152
    cfg = Config(**source_manifest["protocol"]["config"])
    if preflight:
        protocol = copy.deepcopy(protocol)
        protocol.update(source_seeds=[1935], targets_per_stream=256, checkpoint_chunks=32)
        cfg = Config(vocab=12, width=16, hidden=24, heads=2, window=4, chunk=4, inner_lr=.3, init_std=.2)
    files = ("e2e_core.py", "suffix_archive.py", "e35_bounded_archive.py", "record_e35.py", "protocol_e35.json",
             "e33_persistent_e2e.py", "record_e33.py")
    if not preflight:
        for name in files:
            subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", str(root/name)], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "ls-files", "--error-unmatch", str(root/name)], check=True, stdout=subprocess.PIPE)
    assert not out.exists() or not any(out.iterdir()), "Never overwrite experiment output"
    out.mkdir(parents=True, exist_ok=True)
    tracked = {str(audit): sha(audit), str(source/"data.json"): sha(source/"data.json")}
    model_paths = {}
    if not preflight:
        for seed in protocol["source_seeds"]:
            for arm in protocol["source_arms"]:
                path = source / f"{seed}_{arm}" / "trained.pt"
                tracked[str(path)] = sha(path)
                model_paths[f"{seed}_{arm}"] = str(path)
    manifest = {"preflight": preflight, "protocol": protocol, "config": cfg.__dict__,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "files": {name: sha(root/name) for name in files}, "source_files": tracked,
                "source_data": str(source/"data.json"), "source_audit": str(audit), "model_paths": model_paths,
                "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    forbidden = source_maps(json.loads((source/"data.json").read_text()))
    data = prepare_data(protocol, cfg.vocab, forbidden)
    (out/"data.json").write_text(json.dumps(data, separators=(",", ":"))+"\n")
    rows, tables = [], []
    for stream_index, stream in enumerate(data["streams"]):
        tokens = torch.tensor(stream["tokens"])
        tables.append({"stream_index": stream_index, **table_control(stream, cfg.vocab, cfg.chunk)})
        for arm in protocol["source_arms"]:
            model = E2ECore(cfg, seed=stream["seed"])
            if not preflight:
                model.load_state_dict(torch.load(model_paths[f'{stream["seed"]}_{arm}'], weights_only=True)["model"])
            paired = {}
            for policy_name, setting in protocol["policies"].items():
                policy = ArchiveConfig(**protocol["policy_defaults"], **setting)
                directory = out / f'{stream["seed"]}_{arm}_{stream_index}_{policy_name}'
                logits, losses, resource = evaluate(model, policy, tokens, directory, protocol["checkpoint_chunks"])
                paired[policy_name] = resource["output"]
                row = {"seed": stream["seed"], "arm": arm, "stream_index": stream_index,
                       "regime": stream["regime"], "policy": policy_name, "metrics": summarize(logits, losses, stream, protocol),
                       "resource": resource}
                rows.append(row)
                (directory/"result.json").write_text(json.dumps(row, indent=2)+"\n")
                print(json.dumps({"event": "E35_CASE", **{k: row[k] for k in ("seed", "arm", "regime", "policy")},
                                  "seconds": resource["compute_seconds"]}), flush=True)
            left = torch.load(paired["active_gated"], weights_only=True)
            right = torch.load(paired["bank8_no_select_gated"], weights_only=True)
            exact_tree(left["logits"], right["logits"])
            exact_tree(left["losses"], right["losses"])
            exact_tree(left["final"]["state"], right["final"]["state"])
    assert all(sha(path) == expected for path, expected in tracked.items())
    result = {"event": "E35_PREFLIGHT_COMPLETE" if preflight else "E35_COMPLETE", "rows": rows, "tables": tables,
              "data_sha256": sha(out/"data.json"), "manifest_sha256": sha(out/"manifest.json"),
              "no_select_outputs_states_exact": True, "source_files_unchanged": True,
              "seconds": time.perf_counter()-started}
    (out/"complete.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "tables")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.source, args.audit, args.out, args.preflight)
