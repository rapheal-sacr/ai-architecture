"""Post hoc finite-horizon switch removal from actual damaged E35 states.

At chunks1024 and1536 in every bank8 case, retain the full saved state and
disable only future archive selection. Continue for256 chunks. Archive/probe
scoring and admission remain charged. Compare against the original trajectory
at4/64/256/1024 targets. Full-core active-only replay independently verifies
all new operational predictions and fast/KV states from the same saved state.
"""
import argparse
from collections import defaultdict
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from e2e_core import Config, E2ECore, StreamState
from e35_bounded_archive import sha
from record_e33 import exact_tree
from record_e35 import FullCoreKernel
from suffix_archive import ArchiveConfig, SuffixArchive


def main(source, audit, out):
    began=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source,audit,out=Path(source),Path(audit),Path(out)
    verified=json.loads(audit.read_text())
    assert verified["event"] == "E35_AUDITED"
    for name in ("complete","manifest","data"):
        assert sha(source/f"{name}.json") == verified[f"{name}_sha256"]
    done=json.loads((source/"complete.json").read_text())
    manifest=json.loads((source/"manifest.json").read_text())
    data=json.loads((source/"data.json").read_text())
    root=Path(__file__).resolve().parent
    assert all(sha(root/name) == digest for name,digest in manifest["files"].items())
    assert not out.exists() or not any(out.iterdir()), "Do not overwrite a diagnosis"
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope="Post hoc finite-horizon intervention, not a trained policy or goal-completion test",
                  checkpoints=[1024,1536], horizons=[4,64,256,1024], policy="bank8_gated",
                  intervention="Disable only subsequent selection from each exact saved state; retain archives, probe, updates, all costs",
                  reference="Full-core active-only continuation from identical fast/KV state; exact at every chunk",
                  limitation="36 checkpoint-conditioned branches; does not explain all long-run interactions or guarantee future gains",
                  source_complete_sha256=sha(source/"complete.json"),audit_sha256=sha(audit),
                  files={name:sha(root/name) for name in (Path(__file__).name,"e2e_core.py","suffix_archive.py","record_e35.py")})
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    cfg=Config(**manifest["config"])
    assert cfg.chunk == 4
    p=manifest["protocol"]
    rows=[]
    exact_chunks=0
    for case in done["rows"]:
        if case["policy"] != protocol["policy"]:
            continue
        model_path=manifest["model_paths"][f'{case["seed"]}_{case["arm"]}']
        assert sha(model_path) == manifest["source_files"][model_path]
        model=E2ECore(cfg,seed=case["seed"])
        model.load_state_dict(torch.load(model_path,weights_only=True)["model"])
        stream=data["streams"][case["stream_index"]]
        tokens=torch.tensor(stream["tokens"])
        original_path=case["resource"]["output"]
        assert sha(original_path) == case["resource"]["output_sha256"]
        original=torch.load(original_path,weights_only=True)
        for checkpoint in protocol["checkpoints"]:
            record=next(r for r in case["resource"]["checkpoints"] if r["chunk"] == checkpoint)
            assert sha(record["path"]) == record["sha256"]
            saved=torch.load(record["path"],weights_only=True)
            original_config=ArchiveConfig(**saved["config"])
            assert original_config.select and original_config.capacity == 8
            changed=copy.deepcopy(saved)
            changed["config"]["select"]=False
            learner=SuffixArchive(model,replace(original_config,select=False))
            learner.restore(changed)
            reference=SuffixArchive(model,replace(original_config,capacity=0,select=False))
            reference.state=StreamState.restore(saved["state"])
            reference.kernel=FullCoreKernel(model)
            offset=checkpoint*cfg.chunk
            assert offset+max(protocol["horizons"]) <= len(tokens)-1
            logits,losses=[],[]
            before_totals=dict(learner.totals)
            tick=time.perf_counter()
            for begin in range(offset,offset+max(protocol["horizons"]),cfg.chunk):
                inputs= tokens[begin:begin+cfg.chunk]
                targets=tokens[begin+1:begin+cfg.chunk+1]
                result=learner.step(inputs,targets)
                independent=reference.step(inputs,targets)
                assert result["selected_archive"] is None
                assert torch.equal(result["logits"],independent["logits"])
                assert torch.equal(result["losses"],independent["losses"])
                exact_tree(learner.state.payload(),reference.state.payload())
                if begin == offset:
                    assert torch.equal(result["logits"],original["logits"][begin:begin+cfg.chunk])
                logits.append(result["logits"])
                losses.append(result["losses"])
                exact_chunks+=1
            seconds=time.perf_counter()-tick
            logits,losses=torch.cat(logits),torch.cat(losses)
            target=tokens[offset+1:offset+1+max(protocol["horizons"])]
            metrics=[]
            for horizon in protocol["horizons"]:
                actual_logits=original["logits"][offset:offset+horizon]
                actual_losses=original["losses"][offset:offset+horizon]
                metrics.append(dict(targets=horizon,
                    keep_selecting_correct=int(actual_logits.argmax(-1).eq(target[:horizon]).sum()),
                    stop_selecting_correct=int(logits[:horizon].argmax(-1).eq(target[:horizon]).sum()),
                    keep_selecting_loss_sum=float(actual_losses.double().sum()),
                    stop_selecting_loss_sum=float(losses[:horizon].double().sum())))
            row={k:case[k] for k in ("seed","arm","regime","stream_index")}
            row.update(checkpoint_chunk=checkpoint,metrics=metrics,seconds_including_reference=seconds,
                       branch_work={k:learner.totals[k]-v for k,v in before_totals.items()},
                       independent_reference_work=reference.totals)
            rows.append(row)
            path=out/f'{case["seed"]}_{case["arm"]}_{case["regime"]}_{checkpoint}.pt'
            torch.save(dict(logits=logits,losses=losses,final=learner.payload()),path)
            row.update(output=str(path),output_sha256=sha(path))
            print(json.dumps(dict(event="E35_STOP_SELECTION_CASE",**{k:v for k,v in row.items() if k not in ("metrics","branch_work","independent_reference_work")})),flush=True)
    groups=defaultdict(list)
    for row in rows:
        for metric in row["metrics"]:
            groups[(row["arm"],row["regime"],row["checkpoint_chunk"],metric["targets"])].append(metric)
    aggregate=[]
    for (arm,regime,checkpoint,horizon),group in sorted(groups.items()):
        n=sum(r["targets"] for r in group)
        aggregate.append(dict(arm=arm,regime=regime,checkpoint_chunk=checkpoint,horizon=horizon,
            keep_selecting_accuracy=sum(r["keep_selecting_correct"] for r in group)/n,
            stop_selecting_accuracy=sum(r["stop_selecting_correct"] for r in group)/n,
            keep_selecting_nll=sum(r["keep_selecting_loss_sum"] for r in group)/n,
            stop_selecting_nll=sum(r["stop_selecting_loss_sum"] for r in group)/n,
            stop_nll_lower_seeds=sum(r["stop_selecting_loss_sum"] < r["keep_selecting_loss_sum"] for r in group)))
    assert all(sha(root/name) == digest for name,digest in protocol["files"].items())
    result=dict(event="E35_STOP_SELECTION_DIAGNOSED",protocol=protocol,rows=rows,aggregate=aggregate,
                exact_full_core_reference_chunks=exact_chunks,seconds=time.perf_counter()-began)
    (out/"complete.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(event=result["event"],cases=len(rows),exact_chunks=exact_chunks,seconds=result["seconds"])),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--audit",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    main(args.source,args.audit,args.out)
