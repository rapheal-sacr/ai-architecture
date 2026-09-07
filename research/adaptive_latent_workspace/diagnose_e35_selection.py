"""Post hoc E35 switch interventions; not a new trained controller.

Replay all bank8 cases. At every switch with a following chunk, compare the
actual selected-and-updated state with staying on the prior active state and
applying its own fixed update gate. Score both on the same next real chunk.
The latter branch is discarded and never changes the original trajectory.
All original operational logits/events/final payloads must replay exactly.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from e2e_core import Config, E2ECore
from e35_bounded_archive import sha
from record_e33 import exact_tree
from suffix_archive import ArchiveConfig, SuffixArchive


def main(source, audit, out, *, development=False):
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source, audit, out = Path(source), Path(audit), Path(out)
    done = json.loads((source/"complete.json").read_text())
    verified = json.loads(audit.read_text())
    manifest = json.loads((source/"manifest.json").read_text())
    data = json.loads((source/"data.json").read_text())
    assert verified["event"] == ("E35_PREFLIGHT_AUDITED" if development else "E35_AUDITED")
    assert manifest["preflight"] == development
    for name in ("complete", "manifest", "data"):
        assert sha(source/f"{name}.json") == verified[f"{name}_sha256"]
    root = Path(__file__).resolve().parent
    assert all(sha(root/name) == value for name, value in manifest["files"].items())
    assert not out.exists() or not any(out.iterdir()), "Never overwrite a diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(scope="Post hoc one-next-chunk causal switch diagnosis, all bank8 cases and eligible switches",
                    intervention="stay on current fast state; apply original .5-NLL/any-error gate to current observations",
                    outcome="next-chunk CE and errors, before any next-chunk update",
                    stratification="whether all current and next target tokens use one unchanged exact map (evaluator only)",
                    limitation="Local intervention on bank-selected histories, not a whole-trajectory replacement-policy estimate",
                    development=development, source=str(source), audit_sha256=sha(audit),
                    complete_sha256=sha(source/"complete.json"),
                    files={name: sha(root/name) for name in (Path(__file__).name, "e2e_core.py", "suffix_archive.py")})
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    cfg = Config(**manifest["config"])
    p = manifest["protocol"]
    totals = defaultdict(int)
    cases = []
    for row in done["rows"]:
        if row["policy"] != "bank8_gated":
            continue
        tick = time.perf_counter()
        stream = data["streams"][row["stream_index"]]
        tokens = torch.tensor(stream["tokens"])
        model = E2ECore(cfg, seed=row["seed"])
        if not development:
            model_path = manifest["model_paths"][f'{row["seed"]}_{row["arm"]}']
            assert sha(model_path) == manifest["source_files"][model_path]
            model.load_state_dict(torch.load(model_path, weights_only=True)["model"])
        setting = ArchiveConfig(**p["policy_defaults"], **p["policies"][row["policy"]])
        assert setting.update_nll_threshold == .5
        learner = SuffixArchive(model, setting)
        resource = row["resource"]
        assert sha(resource["output"]) == resource["output_sha256"]
        assert sha(resource["trace"]) == resource["trace_sha256"]
        original = torch.load(resource["output"], weights_only=True)
        trace = json.loads(Path(resource["trace"]).read_text())
        maps = [None] * (len(tokens)-1)
        for segment in stream["segments_evaluator_only"]:
            maps[segment["start"]:segment["end"]] = [segment["map_digest"]]*(segment["end"]-segment["start"])
        records = []
        for index, offset in enumerate(range(0, len(tokens)-1, cfg.chunk)):
            saved = trace[index]
            before = learner.state.detached() if saved["selected_archive"] is not None else None
            inputs, targets = tokens[offset:offset+cfg.chunk], tokens[offset+1:offset+cfg.chunk+1]
            result = learner.step(inputs, targets)
            assert torch.equal(result["logits"], original["logits"][offset:offset+cfg.chunk])
            assert torch.equal(result["losses"], original["losses"][offset:offset+cfg.chunk])
            for key, value in result.items():
                if key not in ("logits", "losses"):
                    exact_tree(value, saved[key])
            totals["operational_chunks_exact"] += 1
            if before is None or offset+2*cfg.chunk > len(tokens)-1:
                continue
            # Original full core supplies the counterfactual gradient and caches.
            # This intentionally does not reuse SharedSuffix.updated.
            active_nll = float(result["losses"].mean())
            update = result["correct"] < cfg.chunk or active_nll > setting.update_nll_threshold
            _, losses, logits, stayed, work = model.step(inputs, targets, before,
                                                        mode="first_order" if update else "frozen")
            assert torch.equal(logits.detach(), result["logits"])
            assert torch.equal(losses.detach(), result["losses"])
            totals["diagnostic_full_core_forwards"] += 1
            totals["diagnostic_gradient_updates"] += work["inner_updates"]
            totals["diagnostic_attention_elements"] += work["attention_score_elements"]
            next_inputs = tokens[offset+cfg.chunk:offset+2*cfg.chunk]
            next_targets = tokens[offset+cfg.chunk+1:offset+2*cfg.chunk+1]
            with torch.no_grad():
                actual_next, _, actual_elements = model.chunk_logits(next_inputs, learner.state)
                stayed_next, _, stayed_elements = model.chunk_logits(next_inputs, stayed)
                assert torch.equal(actual_next, original["logits"][offset+cfg.chunk:offset+2*cfg.chunk])
                selected_nll = F.cross_entropy(actual_next, next_targets, reduction="none").mean().item()
                stayed_nll = F.cross_entropy(stayed_next, next_targets, reduction="none").mean().item()
            totals["diagnostic_full_core_forwards"] += 2
            totals["diagnostic_attention_elements"] += actual_elements+stayed_elements
            records.append(dict(chunk=index+1, selected_id=result["selected_archive"],
                current_nll=active_nll, archive_current_nll=result["selected_current_data_nll"],
                selected_next_nll=selected_nll, stayed_next_nll=stayed_nll,
                selected_next_correct=int(actual_next.argmax(-1).eq(next_targets).sum()),
                stayed_next_correct=int(stayed_next.argmax(-1).eq(next_targets).sum()),
                n=cfg.chunk, unchanged_map=len(set(maps[offset:offset+2*cfg.chunk])) == 1,
                actual_updated=result["updated"], stayed_updated=update))
        exact_tree(learner.payload(), original["final"])
        totals["final_payloads_exact"] += 1
        case = {k: row[k] for k in ("seed", "arm", "regime", "stream_index", "policy")}
        case.update(records=records, seconds=time.perf_counter()-tick)
        cases.append(case)
        (out/f'{row["seed"]}_{row["arm"]}_{row["regime"]}.json').write_text(json.dumps(case, separators=(",", ":"))+"\n")
        print(json.dumps(dict(event="E35_SELECTION_CASE", **{k:v for k,v in case.items() if k != "records"},
                              switches_diagnosed=len(records))), flush=True)
    groups = []
    for arm in p["source_arms"]:
        for regime in [r["name"] for r in p["regimes"]]:
            for subset in ("all", "unchanged_map", "crosses_change"):
                records = [r for c in cases if c["arm"] == arm and c["regime"] == regime
                           for r in c["records"] if subset == "all" or r["unchanged_map"] == (subset == "unchanged_map")]
                n = len(records)
                groups.append(dict(arm=arm, regime=regime, subset=subset, switches=n,
                    hindsight_mean_nll_gain=sum(r["current_nll"]-r["archive_current_nll"] for r in records)/n if n else None,
                    next_mean_nll_delta_selection_minus_stay=sum(r["selected_next_nll"]-r["stayed_next_nll"] for r in records)/n if n else None,
                    next_selection_worse_nll=sum(r["selected_next_nll"] > r["stayed_next_nll"] for r in records),
                    next_selected_correct=sum(r["selected_next_correct"] for r in records),
                    next_stayed_correct=sum(r["stayed_next_correct"] for r in records),
                    next_tokens=sum(r["n"] for r in records)))
    assert all(sha(root/name) == value for name, value in protocol["files"].items())
    result = dict(event="E35_SELECTION_DIAGNOSED", protocol=protocol, groups=groups,
                  totals=dict(totals), cases=len(cases), seconds=time.perf_counter()-began,
                  diagnostic_forward_tokens=totals["diagnostic_full_core_forwards"]*cfg.chunk,
                  diagnostic_gradient_tokens=totals["diagnostic_gradient_updates"]*cfg.chunk)
    (out/"complete.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("protocol", "groups")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    main(args.source, args.audit, args.out, development=args.development)
