from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e7/complete.json').read_text())
shutil.copyfile(RUNS/'e7/complete.json',HERE/'results/e7_complete.json')
rows=[]
for arm in data['protocol']['arms']:
    rs=[r for r in data['records'] if r['arm']==arm]
    rows.append(dict(arm=arm,mse=st.mean(r['mse'] for r in rs),seconds=st.mean(r['wall_seconds'] for r in rs),
        late_mse=st.mean(b['mse'] for r in rs for b in r['blocks'][-10:]),
        replacements=st.mean(r['costs'].get('feature_replacements',0) for r in rs),
        tensor_bytes=st.mean(r['costs']['stored_tensor_bytes'] for r in rs),
        last_training_saturation=st.mean(r['costs']['last_training_saturation'] for r in rs) if arm!='context_replay' else None))
(HERE/'results/e7_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E7: feature renewal helps prediction with extra compute','',
'The official generate-and-test implementation is applied to the same two-layer tanh learner with its custom AdamGnT optimizer. Nonrenewal controls use that same optimizer. This separates the renewal intervention from optimizer choice; it does not reproduce the published small-ReLU CBP experiment.','',
'| Arm | MSE | Final ten blocks MSE | Seconds | Tensor bytes | Replacements |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['mse']:.6f} | {r['late_mse']:.6f} | {r['seconds']:.2f} | {r['tensor_bytes']:.0f} | {r['replacements']:.1f} |")
lookup={r['arm']:r for r in rows}
report+=['']
for name,base in [('cbp_gnt','adam_gnt'),('replay_cbp_gnt','replay_gnt')]:
    r=lookup[name];b=lookup[base]
    report.append(f"{name} reduces whole-stream error {100*(1-r['mse']/b['mse']):.1f}% versus {base}, with {100*(r['seconds']/b['seconds']-1):.1f}% more runtime. The error direction agrees on both seeds, but this is a two-seed screen rather than a powered replication.")
report += ['',
'Replay plus renewal has the lowest whole-stream mean prediction error here. The incremental gain from renewal with replay is much smaller than without replay, and its extra runtime remains. In the final ten blocks, replay without renewal has lower mean error (0.04661 versus 0.04886), so the whole-stream gain does not show continuing late-time superiority. Fixed-budget comparisons are needed before calling this efficient. The final training-batch saturation diagnostic decreases with renewal; it is not a measure of global feature quality or proof of a causal saturation mechanism.','',
'This result concerns shared prediction on the external fixed-function stream. It neither solves hidden conflicting contexts nor establishes memory compression or general reasoning. No conclusions about CBP across its published settings follow from this limited port.','',
'The next registered screen moves to closed-loop partial-observation MiniGrid tasks using a pinned recurrent PPO baseline. It tests whether feature renewal remains useful with action-dependent data and sparse reward. That pilot is not the complete proposed architecture, and its 40/80-step tasks are not adequate evidence for very long-horizon competence.']
(HERE/'E7-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
