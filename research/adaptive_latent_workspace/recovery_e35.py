"""Post hoc censored recovery assay on already-audited E35 predictions.

Criterion inherited from E35: at least 75% of 16 outcomes, observed at a real
four-target chunk boundary, all 16 targets within the evaluator's current map
segment. No boundary or criterion is fed back into learning. Report every
segment, including right censoring and subsequent loss of criterion.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch

from e35_bounded_archive import sha


def assay(correct, start, end, window=16, chunk=4, required=12):
    prefix = [0]
    for value in correct:
        prefix.append(prefix[-1]+int(value))
    endpoints = range(((start+window+chunk-1)//chunk)*chunk, end+1, chunk)
    passed = [(point, prefix[point]-prefix[point-window] >= required) for point in endpoints]
    hits = [point for point, ok in passed if ok]
    first = hits[0] if hits else None
    return dict(start=start, end=end, duration=end-start,
                first_criterion_endpoint=first,
                recovered=first is not None, censored=first is None,
                restricted_observations=(first if first is not None else end)-start,
                restricted_lifetime_fraction=((first if first is not None else end)-start)/(end-start),
                below_criterion_after_recovery=any(not ok for point, ok in passed if first is not None and point > first),
                evaluable_windows=len(passed))


def reference(correct, start, end, window=16, chunk=4, required=12):
    flags = [(point, sum(correct[point-window:point]) >= required)
             for point in range(start+window, end+1) if point % chunk == 0]
    first = next((point for point, flag in flags if flag), None)
    return first, any(not flag for point, flag in flags if first is not None and point > first), len(flags)


def preflight():
    # Boundary alignment, no qualifying window, short segment, and recovery
    # followed by a failure. Both calculations are checked for every segment.
    tested = 0
    for correct in ([1]*80, [0]*80, [1]*24+[0]*56, [0]*20+[1]*60,
                    [int((i*7)%13 < 9) for i in range(80)]):
        for start in range(8):
            for end in range(start+1, 81):
                result = assay(correct, start, end)
                first, revoked, windows = reference(correct, start, end)
                assert (result["first_criterion_endpoint"], result["below_criterion_after_recovery"], result["evaluable_windows"]) == (first, revoked, windows)
                assert 0 < result["restricted_lifetime_fraction"] <= 1
                tested += 1
    assert assay([1]*80, 1, 80)["first_criterion_endpoint"] == 20
    assert assay([1]*80, 1, 16)["censored"]
    assert assay([1]*24+[0]*56, 0, 80)["below_criterion_after_recovery"]
    return dict(segments_checked=tested, independent_direct_window_agreement=True)


def main(source, audit, out):
    began = time.perf_counter()
    torch.set_num_threads(1)
    checks = preflight()
    source, audit, out = Path(source), Path(audit), Path(out)
    verified = json.loads(audit.read_text())
    assert verified["event"] == "E35_AUDITED"
    for name in ("complete", "manifest", "data"):
        assert sha(source/f"{name}.json") == verified[f"{name}_sha256"]
    assert not out.exists() or not any(out.iterdir()), "Do not overwrite an assay"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(scope="Post hoc restricted observations to first 75%-of-16 criterion, checked only at four-target chunk boundaries",
                    thresholds="Inherited E35 acquisition window/accuracy; chunk-boundary recovery timing is a new post hoc assay",
                    censoring="Every unrecovered segment right-censored at its actual end; not dropped or labeled recovered",
                    subset="All segments, and separately previously acquired exact-map returns under each learner's original E35 definition",
                    limitations="Restricted delay is optimistic about unobserved future recovery; first hit does not imply stable competence. No recovery wall time was measured.",
                    preflight=checks, audit_sha256=sha(audit), complete_sha256=sha(source/"complete.json"),
                    script_sha256=sha(__file__))
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    done = json.loads((source/"complete.json").read_text())
    data = json.loads((source/"data.json").read_text())
    manifest = json.loads((source/"manifest.json").read_text())
    assert manifest["protocol"]["acquisition_window"] == 16
    assert manifest["protocol"]["acquisition_accuracy"] == .75 and manifest["config"]["chunk"] == 4
    groups = defaultdict(list)
    cases = []
    for case in done["rows"]:
        resource = case["resource"]
        assert sha(resource["output"]) == resource["output_sha256"]
        outputs = torch.load(resource["output"], weights_only=True)
        stream = data["streams"][case["stream_index"]]
        correct = outputs["logits"].argmax(-1).eq(torch.tensor(stream["tokens"][1:])).tolist()
        rows = []
        for segment, previous in zip(stream["segments_evaluator_only"], case["metrics"]["sessions"]):
            a, b = segment["start"], segment["end"]
            assert (a,b) == (previous["start"],previous["end"])
            row = assay(correct, a, b)
            assert reference(correct,a,b) == (row["first_criterion_endpoint"],row["below_criterion_after_recovery"],row["evaluable_windows"])
            row.update(previous_exact_map_acquired=previous["previous_exact_map_acquired"],
                       final_original_window_acquired=previous["last_window_acquired"])
            rows.append(row)
            groups[(case["arm"],case["regime"],case["policy"],"all")].append(row)
            if row["previous_exact_map_acquired"]:
                groups[(case["arm"],case["regime"],case["policy"],"previously_acquired")].append(row)
        cases.append({**{k:case[k] for k in ("arm","regime","policy","seed","stream_index")},"segments":rows})
    summary = []
    for (arm,regime,policy,subset),rows in sorted(groups.items()):
        hits = [r for r in rows if r["recovered"]]
        summary.append(dict(arm=arm,regime=regime,policy=policy,subset=subset,segments=len(rows),
            recovered=len(hits),censored=len(rows)-len(hits),
            no_complete_evaluable_window=sum(r["evaluable_windows"] == 0 for r in rows),
            censored_with_evaluable_window=sum(r["censored"] and r["evaluable_windows"] > 0 for r in rows),
            mean_restricted_lifetime_fraction=statistics.mean(r["restricted_lifetime_fraction"] for r in rows),
            duration_weighted_restricted_fraction=sum(r["restricted_observations"] for r in rows)/sum(r["duration"] for r in rows),
            median_observations_among_recovered=statistics.median(r["restricted_observations"] for r in hits) if hits else None,
            recovered_then_below_criterion=sum(r["below_criterion_after_recovery"] for r in rows),
            final_original_window_acquired=sum(r["final_original_window_acquired"] for r in rows)))
    assert protocol["script_sha256"] == sha(__file__)
    (out/"cases.json").write_text(json.dumps(cases,separators=(",",":"))+"\n")
    result = dict(event="E35_RECOVERY_SUMMARIZED",protocol=protocol,summary=summary,
                  cases=len(cases),segments=sum(len(c["segments"]) for c in cases),
                  seconds=time.perf_counter()-began,cases_sha256=sha(out/"cases.json"))
    (out/"complete.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("protocol","summary")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--audit",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    main(args.source,args.audit,args.out)
