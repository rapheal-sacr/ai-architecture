"""Complete-audit-only E33 tables. No interim checkpoint/arm selection."""
import argparse
import collections
import json
from pathlib import Path


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main(source, audit, out):
    source, out = Path(source), Path(out)
    audit = json.loads(Path(audit).read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    protocol = manifest["protocol"]
    expected_event = "E33_PREFLIGHT_AUDITED" if manifest["preflight"] else "E33_AUDITED"
    assert audit["event"] == expected_event
    assert audit["source_commit"] == manifest["source_commit"]
    seeds, arms = protocol["seeds"], protocol["training_arms"]
    nworld = protocol["evaluation_worlds_per_regime"]
    expected_updates = len(seeds) * len(arms) * protocol["outer_steps"]
    expected_sessions = (len(seeds) * sum(map(len, protocol["evaluation_policies"].values())) *
                         nworld * sum(len(r["schedule"]) for r in protocol["evaluation_regimes"]))
    assert audit["outer_updates_replayed_exact"] == expected_updates
    assert audit["sessions_replayed_exact"] == expected_sessions
    groups = collections.defaultdict(list)
    lookup = {}
    for row in audit["rows"]:
        key = row["seed"], row["arm"], row["policy"], row["regime"], row["world"]
        assert key not in lookup
        lookup[key] = row
        groups[row["arm"], row["policy"], row["regime"]].append(row)
    expected_cells = sum(map(len, protocol["evaluation_policies"].values())) * len(protocol["evaluation_regimes"])
    assert len(groups) == expected_cells
    summary = []
    vocab = protocol["config"]["vocab"]
    for (arm, policy, regime), rows in sorted(groups.items()):
        assert len(rows) == len(seeds) * nworld
        summary.append({"arm": arm, "policy": policy, "regime": regime, "streams": len(rows),
            "initial_acquired": sum(r["first_context_acquired"] for r in rows),
            "all_intervening_acquired": sum(r["all_pre_return_contexts_acquired"] for r in rows),
            "return_reacquired": sum(r["return_last_cycle_acquired"] for r in rows),
            "initial_last_cycle_accuracy": mean(rows, "initial_last_cycle_correct") / vocab,
            "return_first_cycle_accuracy": mean(rows, "return_first_cycle_correct") / vocab,
            "return_first_cycle_nll": mean(rows, "return_first_cycle_nll"),
            "return_last_cycle_accuracy": mean(rows, "return_last_cycle_correct") / vocab,
            "overall_accuracy": sum(r["total_correct"] for r in rows) / sum(r["total_tokens"] for r in rows),
            "overall_nll": sum(r["mean_nll"] * r["total_tokens"] for r in rows) / sum(r["total_tokens"] for r in rows),
            "evaluation_seconds": sum(r["evaluation_seconds"] for r in rows)})
    pairs = []
    for seed in seeds:
        for regime in protocol["evaluation_regimes"]:
            pairs_for_worlds = []
            for world in range(nworld):
                a = lookup[seed, "e2e_carry", "carry_ttt", regime["name"], world]
                b = lookup[seed, "e2e_reset", "carry_ttt", regime["name"], world]
                pairs_for_worlds.append({"nll_delta": a["mean_nll"] - b["mean_nll"],
                    "return_first_correct_delta": (a["return_first_cycle_correct"] - b["return_first_cycle_correct"]) / vocab,
                    "return_first_nll_delta": a["return_first_cycle_nll"] - b["return_first_cycle_nll"],
                    "initial_acquired_delta": int(a["first_context_acquired"]) - int(b["first_context_acquired"]),
                    "return_acquired_delta": int(a["return_last_cycle_acquired"]) - int(b["return_last_cycle_acquired"])})
            pairs.append({"seed": seed, "regime": regime["name"], "worlds": nworld,
                          **{k: mean(pairs_for_worlds, k) for k in pairs_for_worlds[0]}})
    controls = json.loads((source / "table-controls.json").read_text())
    data = json.loads((source / "data.json").read_text())
    table_groups = collections.defaultdict(list)
    for row in controls:
        stream = data["evaluation"][str(row["seed"])][row["stream_index"]]
        table_groups[stream["regime"]].append(row)
    table_rows = []
    for regime, rows in sorted(table_groups.items()):
        assert len(rows) == len(seeds) * nworld
        table_rows.append({"regime": regime, "streams": len(rows),
            "initial_last_cycle_accuracy": sum(r["sessions"][0]["cycle_correct"][-1] for r in rows) / (len(rows) * vocab),
            "return_first_cycle_accuracy": sum(r["sessions"][-1]["cycle_correct"][0] for r in rows) / (len(rows) * vocab),
            "return_last_cycle_accuracy": sum(r["sessions"][-1]["cycle_correct"][-1] for r in rows) / (len(rows) * vocab),
            "overall_accuracy": sum(sum(sum(s["cycle_correct"]) for s in r["sessions"]) for r in rows) /
                                sum(r["queries_and_updates"] for r in rows),
            "seconds": sum(r["seconds"] for r in rows)})
    costs = collections.defaultdict(list)
    for case in audit["cases"]:
        costs[case["arm"]].append(case)
    cost_rows = []
    for arm, rows in sorted(costs.items()):
        assert len(rows) == len(seeds)
        cost_rows.append({"arm": arm, **{k: sum(r[k] for r in rows) for k in (
            "training_seconds", "training_forward_tokens", "training_attention_elements", "inner_updates", "outer_updates_replayed_exact")}})
    for key in ("training_forward_tokens", "training_attention_elements"):
        assert len({r[key] for r in cost_rows}) == 1
    report = {"event": "E33_PREFLIGHT_SUMMARIZED" if manifest["preflight"] else "E33_SUMMARIZED",
              "source_commit": audit["source_commit"], "summary": summary,
              "carry_training_minus_reset_training_both_carry_eval": pairs,
              "table_controls": table_rows, "costs": cost_rows,
              "outer_updates_replayed_exact": expected_updates, "sessions_replayed_exact": expected_sessions,
              "scope": "Paired small grammar assay; acquisition gates and all reversals retained. Not general reasoning, novelty or RSI."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# E33 audited tables", "", "All registered cases completed and exact replay passed. Interpret with E33-PROTOCOL.md.", "",
        "Cells share trained checkpoints and paired data. Counts are streams, not independent model replications.", "",
        "| Regime | Training | Evaluation | Initial acquired | All before return acquired | Return acquired | First-return accuracy | Overall NLL |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"|{r['regime']}|{r['arm']}|{r['policy']}|{r['initial_acquired']}/{r['streams']}|{r['all_intervening_acquired']}/{r['streams']}|{r['return_reacquired']}/{r['streams']}|{r['return_first_cycle_accuracy']:.1%}|{r['overall_nll']:.4f}|")
    lines += ["", "Carry-trained minus reset-trained; both use carried SGD at evaluation:", "",
        "| Seed | Regime | Overall NLL delta | First-return accuracy delta | First-return NLL delta |",
        "|---|---|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"|{r['seed']}|{r['regime']}|{r['nll_delta']:+.4f}|{100*r['return_first_correct_delta']:+.2f} pp|{r['return_first_nll_delta']:+.4f}|")
    lines += ["", "Specialized last-observed-successor table; no hidden context ID:", "",
        "| Regime | Initial last-cycle accuracy | First-return accuracy | Return last-cycle accuracy | Overall accuracy |",
        "|---|---:|---:|---:|---:|"]
    for r in table_rows:
        lines.append(f"|{r['regime']}|{r['initial_last_cycle_accuracy']:.1%}|{r['return_first_cycle_accuracy']:.1%}|{r['return_last_cycle_accuracy']:.1%}|{r['overall_accuracy']:.1%}|")
    lines += ["", "All-seed training costs; equal observations and forward attention entries, not equal FLOPs:", "",
        "| Arm | Seconds | Forward tokens | Attention-score entries | Inner updates | Outer updates |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in cost_rows:
        lines.append(f"|{r['arm']}|{r['training_seconds']:.3f}|{r['training_forward_tokens']}|{r['training_attention_elements']}|{r['inner_updates']}|{r['outer_updates_replayed_exact']}|")
    lines += ["", "Wall times were measured on the shared machine. They are not isolated throughput benchmarks.",
              "Table storage excludes Python overhead; neural tensor storage excludes process/graph/output overhead.",
              "No source-model numerical replication, broad reasoning or autonomous procedure improvement is inferred.", ""]
    (out / "TABLES.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k not in ("summary", "carry_training_minus_reset_training_both_carry_eval", "table_controls", "costs")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.source, args.audit, args.out)
