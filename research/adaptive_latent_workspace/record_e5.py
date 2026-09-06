from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e5/complete.json').read_text())
shutil.copyfile(RUNS/'e5/complete.json',HERE/'results/e5_complete.json')
rows=[]
for arm in data['protocol']['arms']:
    rs=[r for r in data['records'] if r['arm']==arm]
    rows.append(dict(arm=arm,mse=st.mean(s['prequential_mse'] for r in rs for s in r['segments']),
        returning_mse=st.mean(s['first_32_batch_mse'] for r in rs for s in r['segments'] if s['returning']),
        last_quarter_mse=st.mean(s['prequential_mse'] for r in rs for s in r['segments'][-8:]),
        seconds=st.mean(r['wall_seconds'] for r in rs),
        parameters=st.mean(r['costs']['parameter_count'] for r in rs),
        tensor_bytes=st.mean(r['costs']['stored_tensor_bytes'] for r in rs),
        forwards=st.mean(r['costs']['forward_examples'] for r in rs),
        training_examples=st.mean(r['costs']['training_examples'] for r in rs)))
    h=118 if arm in ('wide_context_replay','compact_wide_replay') else 64
    d=44 if 'replay' in arm else 8
    weights=d*h+h*h+h*4
    # Dense matmuls only. Input tensors do not require gradients, so the first
    # layer's input-gradient matmul is absent. Activations/Adam/router excluded.
    rows[-1]['estimated_dense_mac']=rows[-1]['forwards']*weights+rows[-1]['training_examples']*(2*weights-d*h)
(HERE/'results/e5_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
identical=[]
for seed in data['protocol']['seeds']:
    pair={r['arm']:r for r in data['records'] if r['seed']==seed}
    a=pair['isolated_adaptation'];b=pair['active_first_isolation']
    identical.append(all(x[k]==y[k] for x,y in zip(a['segments'],b['segments'])
        for k in ('prequential_mse','clean_prequential_mse','first_32_batch_mse','last_32_batch_mse','heldout_current_mse')))
report=['# E5: protection survives capacity controls; routing can be cheaper','',
'Four fresh seeds each receive 262,144 observations. The protocol was frozen before running any of these outcomes.','',
'| Arm | Stream MSE | Returning early MSE | Last quarter MSE | Seconds | Parameters | Tensor bytes | Forward examples |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['mse']:.6f} | {r['returning_mse']:.6f} | {r['last_quarter_mse']:.6f} | {r['seconds']:.2f} | {r['parameters']:.0f} | {r['tensor_bytes']:.0f} | {r['forwards']:.0f} |")
lookup={r['arm']:r for r in rows};old=lookup['isolated_adaptation'];cheap=lookup['active_first_isolation']
report += ['',
'Increasing context replay to approximately the same parameter count as four isolated modules does not remove the candidate\'s prediction advantage on this family. The small-buffer variant also loses. These are parameter/near-storage controls, not exact FLOP matching. Replay performs nearly twice as many training examples, and the wide network has higher cost per forward example.','',
'Copying temporary updates into the source persistent module eliminates most of the advantage and leaves one persistent module on every seed. The intervention changes only protection during unresolved adaptation; subsequent module counts and routing decisions can therefore differ. This supports the importance of protection within this implementation and tested family.','',
f'Active-first routing has identical stored prediction metrics and held-out current-context metrics on all four seeds: {all(identical)}. This check compares aggregate scored metrics, not saved per-example prediction tensors. It reduces measured time by {100*(1-cheap["seconds"]/old["seconds"]):.1f}% and counted forward examples by {100*(1-cheap["forwards"]/old["forwards"]):.1f}%.','',
'This supports a cheap active-model acceptance test when regimes are well separated. It does not establish a learned router or calibrated confidence. A broadly acceptable but suboptimal active model may hide a better alternative; overlapping contexts remain an adversarial follow-up.','',
'E4\'s independent transfer failure remains in force. Better routing does not solve unnecessary context partitioning, memory consolidation, or general reasoning. No closed-loop task was executed in E5.','',
'For width-aware arithmetic context, the following estimates count dense forward and backward matrix multiply-accumulates. They exclude activation functions, optimizer arithmetic, copies, replay sampling and routing decisions; they are not measured total FLOPs. For each MLP, S = d*h + h*h + h*o and MAC = forward_examples*S + training_examples*(2*S - d*h). The subtraction accounts for inputs that do not require gradients.','',
'| Arm | Estimated dense MAC, billions |','|---|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['estimated_dense_mac']/1e9:.3f} |")
(HERE/'E5-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2));print('identical aggregate metrics',identical)
