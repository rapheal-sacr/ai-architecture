from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e9/complete.json').read_text())
shutil.copyfile(RUNS/'e9/complete.json',HERE/'results/e9_complete.json')
successful=[r for r in data['records'] if 'failed' not in r];rows=[]
report=['# E9: learned procedure and bounded self-application','',
'This experiment distinguishes fixed-procedure student learning, bootstrap meta-learning of an update policy, and that learned policy proposing changes to its own parameters. It tests only a small fixed controller family.','',
'A deterministic double-precision preflight matched the unrolled meta-gradient to a central finite difference and verified initial equivalence to the declared fixed optimizer. A separate dummy end-to-end run exercised bootstrap, self-proposal, admission, final evaluation and counters. Neither preflight is a scored result.','',
'| Replicate | Bootstrap seconds | Accepted self-updates | Self-trial seconds | Controller parameters |',
'|---|---:|---:|---:|---:|']
for r in data['records']:
    if 'failed' in r:report.append(f"| {r['replicate']} | failed: {r['failed']} | — | — | — |")
    else:report.append(f"| {r['replicate']} | {r['bootstrap_seconds']:.2f} | {r['accepted_self_updates']}/{len(r['self_rounds'])} | {sum(t['seconds'] for t in r['self_rounds']):.2f} | {r['controller_parameters']} |")
report+=['','All rejected trials remain in the complete JSON and WD JSONL logs. Divergent tasks are recorded explicitly, never averaged away as missing observations. Admission uses fresh validation tasks plus a retained validation sample; final tasks are separate.','',
'| Distribution | Inner steps | Arm | Replicates | Query MSE | Prequential MSE | Seconds per 16 tasks | Divergent tasks |',
'|---|---:|---|---:|---:|---:|---:|---:|']
for distribution in data['protocol']['final_distributions']:
    for horizon in data['protocol']['final_horizons']:
        for arm in ('adam_default','adam_selected','frozen_bootstrap','self_applied'):
            rs=[x for r in successful for x in r['final'] if x['distribution']==distribution and x['horizon']==horizon and x['arm']==arm]
            if not rs:continue
            diverged=sum(r['diverged_tasks'] for r in rs)
            q=None if diverged else st.mean(r['query_mse'] for r in rs)
            pre=None if diverged else st.mean(r['prequential_mse'] for r in rs)
            row=dict(distribution=distribution,horizon=horizon,arm=arm,replicates=len(rs),query_mse=q,
                prequential_mse=pre,seconds=st.mean(r['seconds'] for r in rs),diverged_tasks=diverged);rows.append(row)
            fmt=lambda x:'diverged' if x is None else f'{x:.6f}'
            report.append(f"| {distribution} | {horizon} | {arm} | {len(rs)} | {fmt(q)} | {fmt(pre)} | {row['seconds']:.3f} | {diverged} |")
report+=['','## Self-application versus the frozen predecessor','']
for r in successful:
    improved=worsened=equal=divergent=0
    for distribution in data['protocol']['final_distributions']:
        for horizon in data['protocol']['final_horizons']:
            pair={x['arm']:x for x in r['final'] if x['distribution']==distribution and x['horizon']==horizon}
            a=pair['self_applied']['query_mse'];b=pair['frozen_bootstrap']['query_mse']
            if a is None or b is None:divergent+=1
            elif a<b:improved+=1
            elif a>b:worsened+=1
            else:equal+=1
    report.append(f"Replicate {r['replicate']}: {improved} settings improved, {worsened} worsened, {equal} tied, {divergent} had divergence. These are nine paired distribution/horizon summaries, not nine independent replications.")
report += ['',
'An admitted change is not automatically useful self-improvement. The final comparisons above determine whether it transfers; the acceptance count only records what the finite admission sample allowed. A small gain on some tasks cannot justify a claim of reliable recursive improvement across the tested family.','',
'The selected fixed optimizer chooses its learning rate on development tasks. Its selection work is stored separately. Learned-controller inference costs, bootstrap meta-training, unsuccessful self-update trials and final evaluation all appear in the complete accounting. No fixed-time final comparison was run; lower query error at equal inner updates is not proof of efficiency.','',
'An additional causal control is missing: continue meta-training the same bootstrap controller with a conventional meta-optimizer under the same proposal/admission budget. E9 compares self-application with freezing, so it cannot attribute any gain specifically to self-application rather than extra meta-training. The self-update code really uses its own learned rule; the benefit of that choice remains unproven.','',
'The controller persists across tasks, but student weights reset between meta tasks. This experiment therefore does not establish persistent task-specific memory or long-horizon agent competence. The architecture and admission rule remain fixed; bounded self-application is not open-ended architecture invention. No full-goal completion is licensed.']
(HERE/'results/e9_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'E9-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
