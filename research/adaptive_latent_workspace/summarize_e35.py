"""Only summarize a completed, hash-matched E35 audit; no checkpoint selection."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from e35_bounded_archive import sha


def summed(rows, metric):
    values = [r["metrics"][metric] for r in rows]
    n = sum(v["n"] for v in values)
    correct = sum(v["correct"] for v in values)
    loss = sum(v["loss_sum"] for v in values)
    return {"n": n, "correct": correct, "accuracy": correct/n if n else None,
            "mean_nll": loss/n if n else None,
            "occurrences": sum(v.get("occurrences", 0) for v in values)}


def main(source, audit, out):
    source, audit, out = Path(source), Path(audit), Path(out)
    completed = json.loads((source/"complete.json").read_text())
    audited = json.loads(audit.read_text())
    assert audited["event"] in ("E35_AUDITED", "E35_PREFLIGHT_AUDITED")
    assert audited["complete_sha256"] == sha(source/"complete.json")
    assert audited["manifest_sha256"] == sha(source/"manifest.json")
    assert audited["data_sha256"] == sha(source/"data.json")
    manifest = json.loads((source/"manifest.json").read_text())
    p = manifest["protocol"]
    groups = defaultdict(list)
    for row in completed["rows"]:
        groups[(row["arm"], row["regime"], row["policy"])].append(row)
    aggregate = []
    for (arm, regime, policy), rows in sorted(groups.items()):
        assert {r["seed"] for r in rows} == set(p["source_seeds"])
        totals = {key: sum(r["resource"]["totals"][key] for r in rows) for key in rows[0]["resource"]["totals"]}
        aggregate.append({"arm": arm, "regime": regime, "policy": policy,
            "all": summed(rows, "all"), "latter_half": summed(rows, "latter_half"),
            "acquired_return_first_window": summed(rows, "acquired_return_first_window"),
            "compute_seconds": sum(r["resource"]["compute_seconds"] for r in rows),
            "checkpoint_seconds": sum(r["resource"]["checkpoint_seconds"] for r in rows),
            "totals": totals, "suffix_forward_multiplier": totals["suffix_tokens"]/totals["prefix_tokens"],
            "gradient_chunk_fraction": totals["gradient_tokens"]/totals["prefix_tokens"],
            "peak_persistent_tensor_bytes": max(r["resource"]["peak_persistent_tensor_bytes"] for r in rows),
            "base_parameter_bytes": rows[0]["resource"]["base_parameter_bytes"]})
    paired = []
    criterion = []
    for arm in p["source_arms"]:
        seed_wins = []
        for regime in [r["name"] for r in p["regimes"]]:
            for seed in p["source_seeds"]:
                rows = [r for r in completed["rows"] if r["arm"] == arm and r["regime"] == regime and r["seed"] == seed]
                assert len(rows) == len(p["policies"])
                get = lambda name: next(r for r in rows if r["policy"] == name)
                baseline, bank = get("active_gated"), get("bank8_gated")
                bm, am = baseline["metrics"]["all"], bank["metrics"]["all"]
                common = [(x["first_window"], y["first_window"])
                          for x, y in zip(baseline["metrics"]["sessions"], bank["metrics"]["sessions"])
                          if x["previous_exact_map_acquired"] and y["previous_exact_map_acquired"]]
                paired.append({"arm": arm, "regime": regime, "seed": seed,
                    "active_gated_accuracy": bm["correct"]/bm["n"], "bank8_accuracy": am["correct"]/am["n"],
                    "active_gated_nll": bm["loss_sum"]/bm["n"], "bank8_nll": am["loss_sum"]/am["n"],
                    "bank8_time_ratio": bank["resource"]["compute_seconds"]/baseline["resource"]["compute_seconds"],
                    "common_acquired_return_occurrences": len(common),
                    "common_acquired_return_active": {k: sum(x[k] for x,y in common) for k in ("n", "correct", "loss_sum")},
                    "common_acquired_return_bank8": {k: sum(y[k] for x,y in common) for k in ("n", "correct", "loss_sum")}})
                if regime == "recurring":
                    seed_wins.append(am["loss_sum"] <= bm["loss_sum"])
        base = next(r for r in aggregate if r["arm"] == arm and r["regime"] == "recurring" and r["policy"] == "active_gated")
        bank = next(r for r in aggregate if r["arm"] == arm and r["regime"] == "recurring" and r["policy"] == "bank8_gated")
        criterion.append({"arm": arm, "aggregate_nll_lower": bank["all"]["mean_nll"] < base["all"]["mean_nll"],
                          "aggregate_error_lower": bank["all"]["correct"] > base["all"]["correct"],
                          "every_seed_nll_nonworsening": all(seed_wins)})
    passed = all(all(v for k,v in c.items() if k != "arm") for c in criterion)
    data = json.loads((source/"data.json").read_text())
    tables = []
    for regime in [r["name"] for r in p["regimes"]]:
        chosen = [t for t in completed["tables"] if data["streams"][t["stream_index"]]["regime"] == regime]
        n, correct = sum(t["n"] for t in chosen), sum(t["correct"] for t in chosen)
        tables.append({"regime": regime, "n": n, "correct": correct, "accuracy": correct/n,
                       "compute_seconds": sum(t["seconds"] for t in chosen),
                       "max_logical_tensor_equivalent_bytes": max(t["logical_int64_key_value_bytes"] for t in chosen)})
    result = {"event": "E35_SUMMARY", "preflight": manifest["preflight"], "source_commit": manifest["source_commit"],
        "aggregate": aggregate, "paired_seeds": paired, "limited_recurrence_criterion": criterion,
        "limited_recurrence_criterion_passed": passed, "table_control": tables,
        "audit_sha256": sha(audit), "complete_sha256": sha(source/"complete.json"),
        "full_goal_achieved": False}
    lines = ["# E35 audited tables", "", "All operational predictions are scored before updates/selection; time and memory are additional costs.", "",
        "| Training | Regime | Policy | Accuracy | NLL | Latter-half accuracy | Acquired return accuracy (occurrences) | Suffix work/input | Updated chunks | CPU seconds | Peak state KiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        ret = r["acquired_return_first_window"]
        ret_text = "unacquired" if ret["accuracy"] is None else f'{ret["accuracy"]*100:.1f}% ({ret["occurrences"]})'
        lines.append(f'|{r["arm"]}|{r["regime"]}|{r["policy"]}|{r["all"]["accuracy"]*100:.2f}%|{r["all"]["mean_nll"]:.4f}|{r["latter_half"]["accuracy"]*100:.2f}%|{ret_text}|{r["suffix_forward_multiplier"]:.2f}|{r["gradient_chunk_fraction"]*100:.1f}%|{r["compute_seconds"]:.3f}|{r["peak_persistent_tensor_bytes"]/1024:.1f}|')
    lines.extend(["", "Return eligibility differs by learner; use the common acquired subset in the paired JSON for controlled return comparisons. State bytes exclude shared base, Python, graphs and outputs. CPU seconds sum all seeds; not isolated accelerator throughput.", "", "| Seed | Training | Regime | Active gated accuracy | Bank8 accuracy | Active gated NLL | Bank8 NLL | Time ratio |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in paired:
        lines.append(f'|{r["seed"]}|{r["arm"]}|{r["regime"]}|{r["active_gated_accuracy"]*100:.2f}%|{r["bank8_accuracy"]*100:.2f}%|{r["active_gated_nll"]:.4f}|{r["bank8_nll"]:.4f}|{r["bank8_time_ratio"]:.2f}|')
    lines.extend(["", "| Regime | Specialized delayed table accuracy | Table CPU seconds | Logical key/value bytes |", "|---|---:|---:|---:|"])
    for r in tables:
        lines.append(f'|{r["regime"]}|{r["accuracy"]*100:.2f}%|{r["compute_seconds"]:.6f}|{r["max_logical_tensor_equivalent_bytes"]}|')
    lines.extend(["", f"Limited recurrence criterion passed: {passed}. This is not full architecture or goal success."])
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"TABLES.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"event": "E35_SUMMARIZED", "preflight": manifest["preflight"], "groups": len(aggregate), "criterion": criterion}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.source, args.audit, args.out)
