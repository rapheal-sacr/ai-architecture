from pathlib import Path
import json,shutil

HERE=Path(__file__).resolve().parent
source=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e1/complete.json')
data=json.loads(source.read_text())
rows=[]
for arm in data['protocol']['arms']:
    rs=[r for r in data['records'] if r['arm']==arm]
    mean=lambda vals:sum(vals)/len(vals)
    rows.append(dict(arm=arm,mean_mse=mean([s['prequential_mse'] for r in rs for s in r['segments']]),
        return_first32=mean([s['first_32_batch_mse'] for r in rs for s in r['segments'] if s['returning']]),
        final_heldout=mean([r['segments'][-1]['heldout_current_mse'] for r in rs]),
        seconds=mean([r['wall_seconds'] for r in rs]),
        parameters=mean([r['costs']['parameter_count'] for r in rs]),
        bytes=mean([r['costs']['stored_tensor_bytes'] for r in rs]),
        forward_examples=mean([r['costs']['forward_examples'] for r in rs]),
        training_examples=mean([r['costs']['training_examples'] for r in rs]),
        modules=mean([r['costs']['module_count'] for r in rs])))
(HERE/'results').mkdir(exist_ok=True)
shutil.copyfile(source,HERE/'results/e1_complete.json')
(HERE/'results/e1_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E1 result: context reuse is not solved','',
'Four frozen seeds, five arms, 65,536 observations per arm/seed. Every model actually learns nonlinear teacher functions. Context IDs reach only the labeled oracle. All scored predictions precede observation of their targets.','',
'| Arm | Stream MSE | First 32 returning batches MSE | Seconds | Parameters | Stored tensor bytes |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['mean_mse']:.5f} | {r['return_first32']:.5f} | {r['seconds']:.3f} | {r['parameters']:.0f} | {r['bytes']:.0f} |")
report+=['','These are equal-experience, one-update-per-batch comparisons, not equal-FLOP or equal-parameter comparisons. Replay uses additional stored samples; module banks evaluate every persistent model after observing each batch. Tensor bytes include model weights, optimizer states and replay storage, but not Python object overhead. CPU timings are local measurements, not frontier hardware estimates.','',
'The module bank does not approach the oracle upper bound and has worse return-task loss than context-conditioned replay. It creates only two or three modules for four contexts. The first architecture hypothesis is therefore **not established**. A lower overall error than a particular baseline is insufficient to claim efficient persistent reuse.','',
'A source-level failure path is that the active module is updated during the novelty-confirmation window. It can partially fit a new regime before the detector decides to allocate capacity, overwriting old behavior. E2 will test temporary adaptation isolated from persistent modules and direct reuse checks against established models. This is a new intervention and protocol; E1 stays unchanged.','',
'No long-horizon closed-loop task, memory compression, plasticity-renewal result or learned routing-cost reduction has been measured by E1. The overall research goal remains active.']
(HERE/'E1-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
