from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
for split in ('dev','fresh'):
    shutil.copyfile(RUNS/f'e2-{split}/complete.json',HERE/f'results/e2_{split}_complete.json')
data=json.loads((RUNS/'e2-fresh/complete.json').read_text())
rows=[];samples={}
for arm in data['protocol']['arms']:
    rs=[r for r in data['records'] if r['arm']==arm]
    returns=[np.mean([s['first_32_batch_mse'] for s in r['segments'] if s['returning']]) for r in rs]
    samples[arm]=returns
    rows.append(dict(arm=arm,stream_mse=float(np.mean([s['prequential_mse'] for r in rs for s in r['segments']])),
        returning_mse=float(np.mean(returns)),seconds=float(np.mean([r['wall_seconds'] for r in rs])),
        parameters=float(np.mean([r['costs']['parameter_count'] for r in rs])),
        tensor_bytes=float(np.mean([r['costs']['stored_tensor_bytes'] for r in rs])),
        forward_examples=float(np.mean([r['costs']['forward_examples'] for r in rs])),
        training_examples=float(np.mean([r['costs']['training_examples'] for r in rs]))))
(HERE/'results/e2_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
ref=next(r for r in rows if r['arm']=='context_replay');ours=next(r for r in rows if r['arm']=='isolated_adaptation')
gain=1-ours['returning_mse']/ref['returning_mse']
report=['# E2: temporary adaptation improves inferred-context reuse','',
'E1 failed to preserve returning contexts. E2 isolates adaptation from established modules while checking whether an existing model explains the newly observed outcomes. After a new temporary model consistently improves observed prediction, it can occupy a persistent slot.','',
'The implementation and protocol were held fixed for four fresh teacher seeds after four development seeds. These are nonlinear networks trained through actual gradient updates; the correct function is not selected from a supplied finite library. Each arm/seed receives 65,536 observations.','',
'| Arm | Stream MSE | Returning-context early MSE | Seconds | Parameters | Tensor bytes |','|---|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['stream_mse']:.5f} | {r['returning_mse']:.5f} | {r['seconds']:.3f} | {r['parameters']:.0f} | {r['tensor_bytes']:.0f} |")
report += ['',f'The returning-context reduction relative to context-conditioned replay is {100*gain:.1f}%, with the same direction on all four fresh seeds. This supports H1/H2 for the tested abrupt recurring-regime problem. It does not establish a general continual-learning solution.','',
'Equal observations and optimizer-update count are not equal compute or parameters. The candidate uses more model parameters than the replay baseline but less tensor state because it does not retain that baseline\'s replay buffer. It evaluates multiple models for routing. Local timings are indicative and not a controlled isolated-device benchmark; the exact operation/example and storage counts remain available.','',
'The oracle still performs considerably better on returning contexts. No task ID or future target is given to the candidate. It must incur identification delay after a change; oracle-selected predictions are never substituted into its score.','',
'Temporary isolation and altered routing/admission rules are bundled in this intervention. The comparison supports the package; further ablations are required to isolate individual causal contributions.','',
'The next failure search is E3: more noise, short contexts, more contexts than module capacity, gradual drift and one-million-observation streams. Beyond E3, closed-loop planning, episodic memory compression, learned routing cost reduction, shared representation learning and an independent external benchmark remain required.','',
'A train-only preflight found and repaired a nonexistent PyTorch API call before any E2 registered outcomes were produced. No protocol thresholds or completed E1 results were changed.']
(HERE/'E2-RESULT.md').write_text('\n'.join(report)+'\n')
names={'context_replay':'Context\nreplay','module_bank':'Original\nmodule bank','isolated_adaptation':'Isolated\nadaptation','oracle_modules':'Oracle\ntask IDs'}
colors=['#718096','#a0aec0','#007f82','#b276b2']
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for i,r in enumerate(rows):
    axes[0].bar(i,r['returning_mse'],color=colors[i],width=.6)
    axes[0].scatter([i]*4,samples[r['arm']],s=18,color='#222222',zorder=3)
    axes[1].scatter(r['seconds'],r['stream_mse'],s=80,color=colors[i])
    axes[1].annotate(names[r['arm']],(r['seconds'],r['stream_mse']),xytext=(5,3),textcoords='offset points',fontsize=8)
axes[0].set_xticks(range(4),[names[r['arm']] for r in rows]);axes[0].set_ylabel('Early returning-context MSE (lower is better)')
axes[0].set_title('Fresh teachers: four paired seeds')
axes[1].set_xlabel('Seconds per 65,536-observation stream');axes[1].set_ylabel('Whole-stream MSE (lower is better)')
axes[1].set_title('Measured local cost and prediction error');axes[1].set_xlim(1.7,3.5)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(HERE/'results/e2_fresh.png',dpi=180)
print(json.dumps(rows,indent=2));print('returning-context relative reduction',gain)
