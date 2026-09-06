"""E33 source-frozen controlled small mechanism assay. CPU by design."""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from e2e_core import Config, E2ECore, StreamState


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def domain_seed(*parts):
    return int.from_bytes(hashlib.sha256(("E33|" + "|".join(map(str, parts))).encode()).digest()[:8], "big")


def make_stream(seed, domain, index, schedule, cycles, vocab):
    rng = torch.Generator().manual_seed(domain_seed(seed, domain, index))
    labels = list(dict.fromkeys(schedule))
    cycles_by_label = {label: torch.randperm(vocab, generator=rng).tolist() for label in labels}
    maps = {}
    for label, cycle in cycles_by_label.items():
        mapping = [None] * vocab
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            mapping[a] = b
        maps[label] = mapping
    sessions = []
    for label in schedule:
        cycle = cycles_by_label[label]
        offset = int(torch.randint(vocab, (1,), generator=rng))
        tokens = [cycle[(offset + t) % vocab] for t in range(cycles * vocab + 1)]
        sessions.append({"label_for_evaluator_only": label, "tokens": tokens})
    return {"seed": seed, "domain": domain, "index": index, "schedule": schedule,
            "cycles": cycles, "maps_for_evaluator_only": maps, "sessions": sessions}


def prepare_data(protocol):
    train, evaluation, train_maps, eval_maps = {}, {}, set(), set()
    vocab = protocol["config"]["vocab"]
    for seed in protocol["seeds"]:
        train[seed] = [make_stream(seed, "train", step, protocol["train_schedule"],
                                   protocol["train_cycles_per_session"], vocab)
                       for step in range(protocol["outer_steps"])]
        evaluation[seed] = []
        for regime in protocol["evaluation_regimes"]:
            for i in range(protocol["evaluation_worlds_per_regime"]):
                stream = make_stream(seed, "eval_" + regime["name"], i, regime["schedule"],
                                     regime["cycles_per_session"], vocab)
                stream["regime"] = regime["name"]
                evaluation[seed].append(stream)
        for stream in train[seed]:
            for mapping in stream["maps_for_evaluator_only"].values():
                hashed = digest(mapping)
                assert hashed not in train_maps, "Unexpected duplicate training transition function"
                train_maps.add(hashed)
        for stream in evaluation[seed]:
            for mapping in stream["maps_for_evaluator_only"].values():
                hashed = digest(mapping)
                assert hashed not in eval_maps, "Unexpected duplicate evaluation transition function"
                eval_maps.add(hashed)
    assert not train_maps.intersection(eval_maps)
    return train, evaluation, {"distinct_train_maps": len(train_maps), "distinct_eval_maps": len(eval_maps),
                               "train_evaluation_map_disjoint": True}


def train_example(model, stream, arm):
    state, losses, work = None, [], []
    for session in stream["sessions"]:
        if state is None or arm != "e2e_carry":
            state = model.initial_state()
        else:
            state = state.clear_context()
        tokens = torch.tensor(session["tokens"], dtype=torch.long)
        _, token_loss, _, state, used = model.sequence(tokens, state=state,
                              mode="frozen" if arm == "static" else "second_order")
        losses.append(token_loss)
        work.extend(used)
    return torch.cat(losses).mean(), work


def summarize_session(logits, losses, tokens, vocab, threshold):
    prediction = logits.argmax(-1)
    target = tokens[1:]
    correct = prediction.eq(target)
    rows = []
    for begin in range(0, len(target), vocab):
        end = begin + vocab
        rows.append({"cycle": begin // vocab, "correct": int(correct[begin:end].sum()),
                     "n": vocab, "mean_nll": float(losses[begin:end].mean())})
    return {"cycles": rows, "correct": int(correct.sum()), "n": len(target),
            "mean_nll": float(losses.mean()), "last_cycle_acquired": rows[-1]["correct"] / vocab >= threshold,
            "predictions": prediction.tolist()}


def table_control(stream, vocab):
    """One last-observed successor per token. No latent context ID is read.

    For unseen inputs, predict uniform probability and deterministic token0;
    for seen inputs, predict last successor. Accuracy only: zero probabilities
    would give infinite NLL on conflicting changes. No invented smoothing.
    """
    memory = {}
    rows = []
    start = time.perf_counter()
    for session in stream["sessions"]:
        tokens = session["tokens"]
        correct, predictions = [], []
        for x, y in zip(tokens[:-1], tokens[1:]):
            prediction = memory.get(x, 0)
            predictions.append(prediction)
            correct.append(int(prediction == y))
            memory[x] = y
        rows.append({"predictions": predictions,
                     "cycle_correct": [sum(correct[i:i+vocab]) for i in range(0, len(correct), vocab)]})
    return {"sessions": rows, "seconds": time.perf_counter() - start,
            "max_entries": vocab, "logical_int64_key_value_bytes": 16 * vocab,
            "storage_caveat": "Logical entries only; Python dictionary overhead excluded.",
            "queries_and_updates": sum(len(s["tokens"]) - 1 for s in stream["sessions"])}


def evaluate(model, stream, policy, threshold, checkpoint_dir):
    state, sessions, total_work = None, [], []
    start = time.perf_counter()
    for i, session in enumerate(stream["sessions"]):
        if state is None or policy != "carry_ttt":
            state = model.initial_state()
        else:
            state = state.clear_context()
        tokens = torch.tensor(session["tokens"], dtype=torch.long)
        _, losses, logits, state, work = model.sequence(tokens, state=state,
                    mode="frozen" if policy == "frozen" else "first_order", detach_between=True)
        total_work.extend(work)
        row = summarize_session(logits, losses, tokens, model.cfg.vocab, threshold)
        # For replay and independent score reconstruction, preserve full logits,
        # losses and the complete operational state after every document.
        cp = checkpoint_dir / f"session-{i}.pt"
        torch.save({"logits": logits, "losses": losses, "state": state.payload()}, cp)
        row["checkpoint"] = str(cp)
        sessions.append(row)
    return {"sessions": sessions, "seconds": time.perf_counter() - start,
            "forward_tokens": sum(w["forward_tokens"] for w in total_work),
            "attention_score_elements": sum(w["attention_score_elements"] for w in total_work),
            "inner_updates": sum(w["inner_updates"] for w in total_work),
            "final_state_tensor_bytes": state.bytes()}


def main(out, preflight):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol_e33.json").read_text())
    if preflight:
        protocol["seeds"] = [1933]
        protocol["outer_steps"] = 2
        protocol["train_cycles_per_session"] = 2
        protocol["evaluation_worlds_per_regime"] = 1
        protocol["evaluation_regimes"] = [{"name": "preflight", "schedule": "ABA", "cycles_per_session": 2}]
        protocol["config"].update(width=16, hidden=24, heads=2)
    files = ("e2e_core.py", "e33_persistent_e2e.py", "protocol_e33.json")
    if not preflight:
        for name in files:
            subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", str(root / name)], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "ls-files", "--error-unmatch", str(root / name)], check=True, stdout=subprocess.PIPE)
    out = Path(out)
    assert not out.exists() or not any(out.iterdir()), "Never overwrite or restart a live/scored run"
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"source_commit": commit, "preflight": preflight, "protocol": protocol,
                "protocol_digest": digest(protocol), "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1},
                "concurrency": "E32 GPU experiment may run concurrently; wall time is observed, not isolated hardware latency.",
                "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train, evaluation, data_audit = prepare_data(protocol)
    data = {"train": train, "evaluation": evaluation, "audit": data_audit}
    (out / "data.json").write_text(json.dumps(data) + "\n")
    cfg = Config(**protocol["config"])
    cases = {}
    for seed in protocol["seeds"]:
        for arm in protocol["training_arms"]:
            model = E2ECore(cfg, seed=domain_seed(seed, "initialization"))
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["outer_lr"],
                                         weight_decay=protocol["outer_weight_decay"])
            cases[(seed, arm)] = {"model": model, "optimizer": optimizer, "training": []}
    # Rotate arms in a fixed predeclared order on every outer update. This does
    # not make wall time perfectly paired, but avoids running whole arms in
    # different portions of the background GPU experiment.
    for step in range(protocol["outer_steps"]):
        for (seed, arm), case in cases.items():
            model, optimizer = case["model"], case["optimizer"]
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, work = train_example(model, train[seed][step], arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["outer_clip"])
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step()
            case["training"].append({"step": step + 1, "stream_digest": digest(train[seed][step]),
                "loss": loss.item(), "outer_grad_norm": float(norm), "seconds": time.perf_counter() - tick,
                "forward_tokens": sum(w["forward_tokens"] for w in work),
                "attention_score_elements": sum(w["attention_score_elements"] for w in work),
                "inner_updates": sum(w["inner_updates"] for w in work), "outer_updates": 1})
        if (step + 1) % 16 == 0 or step == protocol["outer_steps"] - 1:
            print(json.dumps({"event": "E33_TRAIN", "step": step + 1,
                              "losses": {f"{s}_{a}": c["training"][-1]["loss"] for (s, a), c in cases.items()},
                              "elapsed": time.perf_counter() - start}), flush=True)
    results = []
    for (seed, arm), case in cases.items():
        model = case["model"]
        case_dir = out / f"{seed}_{arm}"
        case_dir.mkdir()
        torch.save({"model": model.state_dict(), "optimizer": case["optimizer"].state_dict(),
                    "config": asdict(cfg), "rng": torch.get_rng_state()}, case_dir / "trained.pt")
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
        result = {"seed": seed, "arm": arm, "model": model.specification(), "training": case["training"], "evaluation": []}
        for policy in protocol["evaluation_policies"][arm]:
            for index, stream in enumerate(evaluation[seed]):
                cpdir = case_dir / policy / str(index)
                cpdir.mkdir(parents=True)
                row = evaluate(model, stream, policy, protocol["acquisition_cycle_accuracy"], cpdir)
                row.update(policy=policy, stream_index=index, stream_digest=digest(stream), regime=stream["regime"])
                result["evaluation"].append(row)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
        result["evaluation_model_weights_unchanged"] = True
        (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append({"seed": seed, "arm": arm, "path": str(case_dir / "result.json")})
        print(json.dumps({"event": "E33_CASE", "seed": seed, "arm": arm, "elapsed": time.perf_counter() - start}), flush=True)
    controls = [{"seed": seed, "stream_index": i, "stream_digest": digest(stream), **table_control(stream, cfg.vocab)}
                for seed, streams in evaluation.items() for i, stream in enumerate(streams)]
    (out / "table-controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    complete = {"event": "E33_PREFLIGHT_COMPLETE" if preflight else "E33_COMPLETE",
                "source_commit": commit, "cases": results, "data_audit": data_audit,
                "seconds": time.perf_counter() - start, "status": "Await independent audit; no success inferred"}
    (out / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.out, args.preflight)
