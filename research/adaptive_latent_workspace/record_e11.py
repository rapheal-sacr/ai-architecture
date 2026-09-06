from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e11/complete.json').read_text())
shutil.copyfile(RUNS/'e11/complete.json',HERE/'results/e11_complete.json')
report=['# E11: self-application versus equal-budget conventional meta-updates','',
'The fixed control receives the same eight proposal opportunities, task seeds, learning rate, moment coefficients and admission rule as E9. It sets the learned multiplier to one. This controls additional meta-training while preserving the proposed self-updater and its frozen predecessor unchanged. New final-task seeds were frozen before scored execution.','',
'A replay preflight reconstructed replicate 12101 from its bootstrap checkpoint and exactly matched E9 final controller tensors and all eight admission decisions. This verifies the proposal-loop adapter; it is not a new scored replicate.','',
'| Replicate | Self-updates accepted | Fixed updates accepted | Prior bootstrap seconds | Prior self-proposal seconds | Fixed proposal seconds |','|---|---:|---:|---:|---:|---:|']
for r in data['records']:
    old=json.loads((RUNS/'e9'/f"{r['replicate']}.json").read_text())
    report.append(f"| {r['replicate']} | {old['accepted_self_updates']}/8 | {r['proposals']['accepted']}/8 | {r['inherited_cost']['bootstrap_seconds']:.2f} | {r['inherited_cost']['self_trial_seconds']:.2f} | {r['proposals']['seconds']:.2f} |")
report+=['','Query error follows; lower is better. The final tasks are paired within each setting. Settings share teacher seeds and are correlated, so the number of settings won is descriptive, not a significance test.','',
'| Replicate | Distribution | Steps | Frozen MSE | Self-applied MSE | Fixed-meta MSE | Self lower than fixed: tasks / 16 |',
'|---|---|---:|---:|---:|---:|---:|']
rows=[]
for r in data['records']:
    for dist in data['protocol']['final_distributions']:
        for horizon in data['protocol']['final_horizons']:
            pair={x['arm']:x for x in r['final'] if x['distribution']==dist and x['horizon']==horizon}
            a=pair['self_applied'];b=pair['fixed_meta_adam'];c=pair['frozen_bootstrap']
            assert [x['seed'] for x in a['rows']]==[x['seed'] for x in b['rows']]
            valid=all(x['query_mse'] is not None for x in (a,b,c))
            wins=sum(x['query_mse']<y['query_mse'] for x,y in zip(a['rows'],b['rows']) if x['query_mse'] is not None and y['query_mse'] is not None)
            row=dict(replicate=r['replicate'],distribution=dist,horizon=horizon,frozen_mse=c['query_mse'],self_mse=a['query_mse'],fixed_meta_mse=b['query_mse'],paired_self_task_wins=wins,diverged_tasks=sum(x['diverged_tasks'] for x in pair.values()))
            rows.append(row)
            fmt=lambda x:'diverged' if x is None else f'{x:.6f}'
            report.append(f"| {r['replicate']} | {dist} | {horizon} | {fmt(c['query_mse'])} | {fmt(a['query_mse'])} | {fmt(b['query_mse'])} | {wins}/{len(a['rows'])} |")
for rep in data['protocol']['replicates']:
    rs=[x for x in rows if x['replicate']==rep and x['self_mse'] is not None and x['fixed_meta_mse'] is not None]
    report+=['',f"Replicate {rep}: self-application has lower mean error than conventional meta-updates in {sum(x['self_mse']<x['fixed_meta_mse'] for x in rs)}/{len(rs)} settings, and lower mean error than freezing in {sum(x['self_mse']<x['frozen_mse'] for x in rs)}/{len(rs)} settings."]
report+=['',
'These are the two previously studied bootstrap initializations, not a new full-pipeline replication. The control only tests the declared fixed Adam-shaped rule; it does not search all meta-optimizers or learning rates. All rejected proposals and divergent evaluations remain in the complete records.','',
'Bootstrap and prior self-trial work are inherited costs, not free checkpoints. The E9 records preserve their detailed training and evaluation counts. E11 records all extra conventional proposal and final-evaluation work. CPU evaluation overlapped E10 GPU work: wall times are descriptive, and no fixed-time efficiency claim follows.','',
'Persistent controller parameters still coexist with students that reset between tasks. This control cannot establish persistent factual memory, general reasoning, open-ended recursive improvement or long-horizon agent efficiency.']
(HERE/'E11-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e11_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
