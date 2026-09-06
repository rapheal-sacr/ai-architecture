from pathlib import Path
import json,shutil,statistics as st,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e8/complete.json').read_text())
shutil.copyfile(RUNS/'e8/complete.json',HERE/'results/e8_complete.json')
rows=[]
for r in data['records']:
    for s in r['stages']:
        rows.append(dict(seed=r['seed'],arm=r['arm'],stage=s['stage'],
            two_room_success=s['evaluations'][0]['success_rate'],four_room_success=s['evaluations'][1]['success_rate'],
            seconds=s['seconds'],parameters=r['parameters'],tensor_bytes=s['tensor_bytes'],peak_gpu_bytes=s['peak_gpu_allocated_bytes'],
            replacements=s['feature_replacements'],last_entropy=s['training'][-1]['entropy'],forward_examples=s['forward_examples']))
(HERE/'results/e8_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E8: closed-loop transfer and retention fail during harder-task learning','',
'The registered pilot completed for two seeds and two arms, each with 786,432 training frames across two-room, four-room and returning two-room environments. Policies see partial 7×7 symbolic images. Weights, optimizer state and renewal utility persist across stages; only episodic recurrent state resets.','',
'| Seed | Arm | After training on | Two-room successes / 50 | Four-room successes / 50 | Seconds | Cumulative replacements |',
'|---|---|---|---:|---:|---:|---:|']
labels=['two rooms','four rooms','returning two rooms']
for r in rows:report.append(f"| {r['seed']} | {r['arm']} | {labels[r['stage']]} | {round(50*r['two_room_success'])} | {round(50*r['four_room_success'])} | {r['seconds']:.1f} | {r['replacements']} |")
report += ['',
'All arms acquire the two-room task in the first stage. Four-room training then loses almost all measured two-room competence while failing the four-room test. Both arms reacquire two-room success when training returns there. Reacquisition is not the same as retention. Feature renewal in the actor/critic heads does not prevent this failure.','',
'There can be nonzero four-room transfer after returning to the easier task; the complete per-seed results above preserve that outcome. It does not reverse the failed acquisition/retention result during the registered harder-task stage, and two seeds cannot establish a broad transfer effect.','',
'The final harder-task training entropy approaches log(7), the maximum for seven actions, while reward and value estimates approach zero. That is consistent with objective-driven movement toward uniform actions under weak reward feedback. It is not yet a causal proof. E10 tests entropy and critic-representation interference from the same starting checkpoints; the E8 outcome is not silently corrected.','',
'The policy has no private-grid or transition-function access. Evaluation uses frozen weights and fresh environment seeds; evaluation RNG is restored afterward. No evaluation checkpoint selection or hyperparameter tuning occurs within this run.','',
'Costs include rollout/update time and forward-example counts. Tensor-byte counts cover model, optimizer and renewal state; peak GPU allocation also includes runtime training tensors. They do not measure all process RAM or energy. Small reporting and CPU preflight tasks may have overlapped; timings are local indicative measurements, not isolated-device claims.','',
'This is a fixed-procedure closed-loop pilot using pinned PPO and CNN/LSTM code. It does not demonstrate recursive self-improvement, a learned world model, memory compression or general reasoning. The environment limits are only 40/80 steps, so this cannot establish very long-horizon efficiency. The broader architecture remains unverified.']
(HERE/'E8-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
for arm,color in [('recurrent_ppo','#64748b'),('recurrent_ppo_renewal','#007f82')]:
    for ax,key in zip(axes,['two_room_success','four_room_success']):
        means=[st.mean(r[key] for r in rows if r['arm']==arm and r['stage']==s) for s in range(3)]
        ax.plot(range(3),means,marker='o',color=color,label=arm.replace('_',' '))
        for seed in data['protocol']['seeds']:
            ys=[next(r[key] for r in rows if r['arm']==arm and r['seed']==seed and r['stage']==s) for s in range(3)]
            ax.scatter(range(3),ys,color=color,s=18,alpha=.5)
for ax,title in zip(axes,['Previously learned two-room task','Harder four-room task']):
    ax.set_title(title);ax.set_xticks(range(3),['After 2 rooms','After 4 rooms','After return to 2'])
    ax.set_ylim(-.03,1.05);ax.set_ylabel('Frozen-policy success rate');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8);fig.suptitle('Closed-loop pilot: two training seeds, 50 held-out episodes per task/checkpoint')
fig.savefig(HERE/'results/e8_closed_loop.png',dpi=180)
print(json.dumps(rows,indent=2))
