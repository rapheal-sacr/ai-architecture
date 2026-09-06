from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e12/complete.json').read_text())
shutil.copyfile(RUNS/'e12/complete.json',HERE/'results/e12_complete.json')
rows=[];report=['# E12: fresh acquisition, unseen action chains and return-task retention','',
'Both entropy choices begin at random initialization with two fresh training seeds. The learner retains weights and optimizer state across two-room, four-room, key-door, and returning two-room stages. No stage identity is supplied to the policy. The third environment requires using a key, a behavior absent from MultiRoom.','',
'| Seed | Arm | Training stage | Two-room / 50 | Four-room / 50 | Key-door / 50 | Training seconds | Evaluation seconds |',
'|---|---|---|---:|---:|---:|---:|---:|']
for r in data['records']:
    for s in r['stages']:
        row=dict(seed=r['seed'],arm=r['arm'],stage=s['stage'],environment=s['environment'],max_steps=s['max_steps'],
            success_rates=[e['success_rate'] for e in s['evaluations']],training_seconds=s['training_seconds'],evaluation_seconds=s['evaluation_seconds'],
            tensor_bytes=s['tensor_bytes'],counts=s['counts'],parameters=r['parameters'],peak_gpu_bytes=s['peak_gpu_bytes'])
        rows.append(row);n=[round(x*50) for x in row['success_rates']]
        report.append(f"| {r['seed']} | {r['arm']} | {s['stage']}: {s['environment']} | {n[0]} | {n[1]} | {n[2]} | {s['training_seconds']:.1f} | {s['evaluation_seconds']:.1f} |")
report+=['',
'Neither fixed entropy setting reliably acquires the harder sequence. Standard entropy ends four-room training with 0/50 four-room success in both seeds and ends key-door training with 1/50 and 3/50 key-door success. It reacquires the original task at 50/50 in both seeds. Zero entropy collapses to 0/50 on all three tasks after four-room training in both seeds, and remains at zero after key-door training. On return, one seed stays at zero while the other reacquires two-room success at 50/50, with 6/50 four-room and 1/50 key-door success. Thus E10\'s entropy-off correction does not generalize as a sufficient repair.', '',
'The zero-entropy failures are not limited to immediately following a task switch: seed 13101 has strong four-room training returns before a late collapse, and seed 13102 has strong initial two-room returns before a late collapse within that stage. Frozen endpoint evaluation is reported without selecting an earlier successful checkpoint. The precise cause of these late instabilities is not isolated by this two-arm experiment. Neither more entropy nor its deletion alone is a reliable learning safeguard.', '',
'Each arm/seed receives 1,048,576 training frames. The tasks have 40/80/640-step episode limits. This expands the action-chain and time horizon of the prior pilot but does not establish general reasoning or very long-horizon agency.','',
'Evaluation uses frozen weights and separately seeded episodes. Its scores do not influence updates, stage order, entropy or checkpoint selection. Identical evaluation seeds recur across stages to measure retention; they are not additional independent training replications. Students do not reset between stages.','',
'Complete counts include training, rollout and evaluation forwards, optimizer updates and model/optimizer tensor bytes. Per-stage checkpoints permit continuation without throwing away complete stages. Peak GPU allocation does not include all driver or process RAM. No energy measurement or fixed-time benchmark was run.','',
'The architecture and objective choices are fixed by this experiment. This is a baseline/mechanism test, not a demonstration that the system learned its own objective. It contains no world model, planner, neural compression mechanism or persistent learned procedure. Its successes or failures constrain those subsequent proposals rather than completing the overall goal.']
(HERE/'results/e12_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'E12-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
