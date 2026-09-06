from pathlib import Path
import json,math,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e10-v2/complete.json').read_text());old=json.loads((RUNS/'e8/complete.json').read_text())
shutil.copyfile(RUNS/'e10-v2/complete.json',HERE/'results/e10_complete.json')
rows=[];checks=[]
for r in data['records']:
    if r['arm']=='entropy_shared':
        prior=next(x for x in old['records'] if x['seed']==r['seed'] and x['arm']=='recurrent_ppo')['stages'][1]
        check=dict(seed=r['seed'],training_equal=r['training']==prior['training'],evaluation_common_fields_equal=all(all(x[k]==y[k] for k in x) for x,y in zip(r['evaluations'],prior['evaluations'])),maximum_absolute_training_difference={k:max(abs(x[k]-y[k]) for x,y in zip(r['training'],prior['training'])) for k in r['training'][0]})
        checks.append(check)
    rows.append(dict(seed=r['seed'],arm=r['arm'],two_room_success=r['evaluations'][0]['success_rate'],four_room_success=r['evaluations'][1]['success_rate'],seconds=r['seconds'],training_frames=r['training_frames'],counts=r['counts'],parameters=r['parameters'],peak_gpu_bytes=r['peak_gpu_bytes'],last_entropy=r['training'][-1]['entropy'] if r['training'] else None))
report=['# E10: entropy and critic-gradient counterfactual','',
'Two factors change from identical E8 two-room checkpoints: the entropy bonus and whether critic gradients reach the shared CNN/LSTM. Actor gradients remain enabled in every learning arm. A frozen-policy reference receives no updates. Each learning arm uses 262,144 four-room frames and the same starting optimizer/RNG state.','',
'The first CUDA launch failed before its first optimizer update because restored custom optimizer step tensors stayed on CPU. The corrected adapter moves state to its parameter device and passed CUDA preflight. The failed launch and source hashes remain recorded in E10-RUNTIME-NOTE.md; its unmeasured startup time is not counted as zero.','',
'| Seed | Arm | Two-room successes / 50 | Four-room successes / 50 | Last entropy | Training seconds |','|---|---|---:|---:|---:|---:|']
for r in rows:
    entropy='—' if r['last_entropy'] is None else f"{r['last_entropy']:.6f}"
    report.append(f"| {r['seed']} | {r['arm']} | {round(50*r['two_room_success'])} | {round(50*r['four_room_success'])} | {entropy} | {r['seconds']:.1f} |")
report+=['','The maximum seven-action entropy is log(7) = '+f'{math.log(7):.6f}.','', '## Checkpoint control reproduction','']
for c in checks:report.append(f"Seed {c['seed']}: complete training rows equal to E8 = {c['training_equal']}; all recorded evaluation fields equal = {c['evaluation_common_fields_equal']}; maximum training metric differences = {c['maximum_absolute_training_difference']}.")
report+=['',
'The factors can change exploration and subsequent data as well as direct gradient interference. This is a closed-loop causal intervention, not an attribution of every downstream effect to one local gradient. Successful retention alone does not establish acquisition of the harder task.','',
'These checkpoints and environment families were already studied in E8. The counterfactual is a development diagnostic, not fresh final validation. Only two training seeds are tested. Fifty evaluation episodes per checkpoint are not fifty independently trained models.','',
'All learning arms have the same parameter count and collected frame budget. Counts and peak allocated GPU bytes are retained in the JSON. Peak allocation excludes driver/context memory and process RAM. E11 CPU work overlaps part of execution, so timings do not establish fixed-time superiority. The frozen arm has zero training work by definition.','',
'This experiment does not implement memory compression, learned dynamics, planning or recursive objective learning. Any useful objective change here is researcher-selected and must not be described as autonomous procedural improvement.']
(HERE/'E10-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e10_summary.json').write_text(json.dumps(dict(rows=rows,reproduction=checks),indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
colors=['#64748b','#007f82','#b45309','#7c3aed']
for i,seed in enumerate(sorted({r['seed'] for r in data['records']})):
    for color,arm in zip(colors,data['protocol']['arms'][:4]):
        r=next(r for r in data['records'] if r['seed']==seed and r['arm']==arm)
        for ax,key in [(axes[i,0],'entropy'),(axes[i,1],'mean_return')]:
            ax.plot([x['frames'] for x in r['training']],[x[key] for x in r['training']],label=arm.replace('_',' '),color=color,linewidth=1.2)
    for ax in axes[i]:ax.set_xlabel('Four-room training frames');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[i,0].set_title(f'Seed {seed}: action entropy');axes[i,1].set_title(f'Seed {seed}: rollout return')
    axes[i,0].axhline(math.log(7),color='gray',linestyle=':',linewidth=1)
axes[0,0].legend(fontsize=7);fig.suptitle('Counterfactuals from identical learned policies; two development seeds')
fig.savefig(HERE/'results/e10_objective.png',dpi=180)
print(json.dumps(dict(rows=rows,reproduction=checks),indent=2))
