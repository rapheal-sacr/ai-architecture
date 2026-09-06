from pathlib import Path
import json,shutil,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e14/complete.json').read_text())
shutil.copyfile(RUNS/'e14/complete.json',HERE/'results/e14_complete.json')
rows=[];costs=[]
report=['# E14: neural dynamics and planning help selectively; protected memory does not engage','',
'The learner acquires a neural transition/reward model from actual observations and uses it to search action sequences. Model weights, optimizer state and bounded replay persist across hidden gravity changes. The planner receives observations and model predictions only; it cannot access simulator equations, private state or the gravity parameter.','',
'Twenty-step replay planning improves actual frozen-policy return over the one-step version in seven of eight paired seed/stage settings. It remains poor at high gravity, and one initialization is weak in the initial normal-gravity evaluation. Guarded sharing never creates a temporary model, grows a module or merges: its thresholds do not activate those mechanisms in this normalized target space. Its outcomes therefore cannot support a claim that protection or consolidation caused a gain.','',
'| Seed | Arm | Stage / gravity | Mean training episode return | Frozen evaluation return | Prequential target MSE |',
'|---|---|---|---:|---:|---:|']
for r in data['records']:
    costs.append(dict(seed=r['seed'],arm=r['arm'],seconds=r['seconds'],training_environment_steps=r['training_environment_steps'],evaluation_environment_steps=r['evaluation_environment_steps'],probe_environment_steps=r['probe_environment_steps'],costs=r['costs']))
    for s in r['stages']:
        row=dict(seed=r['seed'],arm=r['arm'],stage=s['stage'],gravity=s['gravity'],training_mean_return=st.mean(s['training_returns']),evaluation_mean_return=s['evaluation']['mean_return'],prequential_mse=s['training'][-1]['prequential_mse'],training_seconds=s['training_seconds'],evaluation_seconds=s['evaluation']['seconds'],costs=s['costs'])
        rows.append(row);mse='—' if row['prequential_mse'] is None else f"{row['prequential_mse']:.6f}"
        report.append(f"| {r['seed']} | {r['arm']} | {s['stage']} / {s['gravity']:g} | {row['training_mean_return']:.2f} | {row['evaluation_mean_return']:.2f} | {mse} |")
report+=['','Higher return (closer to zero) is better. Each frozen evaluation is only four 200-step episodes, with two independently trained seeds; stage summaries share trained state and are not independent replications. Prequential MSE mixes normalized angular/velocity changes and reward, so it is not a direct control-performance metric.','',
'## Complete costs','',
'| Seed | Arm | Total seconds | Forward examples | Parameters | Stored tensor bytes | Modules | Merges |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in costs:
    c=r['costs'] or {};report.append(f"| {r['seed']} | {r['arm']} | {r['seconds']:.2f} | {c.get('forward_examples',0)} | {c.get('parameter_count',0)} | {c.get('stored_tensor_bytes',0)} | {c.get('module_count',0)} | {c.get('merges',0)} |")
short=[r for r in costs if r['arm']=='replay_mpc1'];long=[r for r in costs if r['arm']=='replay_mpc20']
report+=['',f"Twenty-step versus one-step replay uses {st.mean(r['costs']['forward_examples'] for r in long)/st.mean(r['costs']['forward_examples'] for r in short):.2f}× forward examples and {st.mean(r['seconds'] for r in long)/st.mean(r['seconds'] for r in short):.2f}× measured time. Both receive 12,800 training environment steps per seed and 3,200 frozen evaluation steps; learned arms also receive 640 diagnostic probe steps that never train the model.", '',
'The guarded arm uses fewer replay slots (128 local anchors versus the ordinary 2,048-example reservoir). Its smaller stored tensor count reflects that budget difference, not demonstrated semantic compression. The acting learner cannot read the archived per-stage research checkpoints. Costs count imagined, training, routing and evaluation model calls; counts are examples, not complete FLOPs. GPU research runs overlap this CPU pilot, so no isolated-hardware or fixed-time superiority claim follows.','',
'## Multi-step probes','',
'Four new initial states per stage receive the same fixed random action sequence in the real environment and the frozen learned model. One-step targets use each real current observation; the imagined trajectory accumulates its own predictions. These probe actions differ from the planner-selected control distribution, so probe error cannot by itself explain a control failure. Receding-horizon replanning also corrects state using real observations each step.','',
'| Seed | Arm | Stage | Mean one-step target MSE | State MSE at 20 steps | State MSE at 40 steps | Predicted minus actual probe return |',
'|---|---|---:|---:|---:|---:|---:|']
probes=[]
for r in data['records']:
    for s in r['stages']:
        p=s['probe']
        if p is None:continue
        row=dict(seed=r['seed'],arm=r['arm'],stage=s['stage'],one_step_target_mse=st.mean(z['one_step_target_mse'] for z in p['rows']),
            state_mse20=p['rows'][19]['imagined_state_mse'],state_mse40=p['rows'][39]['imagined_state_mse'],optimism=st.mean(p['predicted_returns'])-st.mean(p['actual_returns']))
        probes.append(row);report.append(f"| {r['seed']} | {r['arm']} | {s['stage']} | {row['one_step_target_mse']:.6f} | {row['state_mse20']:.6f} | {row['state_mse40']:.6f} | {row['optimism']:.2f} |")
report+=['',
'E15 separately tests the same search with an explicitly privileged accurate dynamics/reward model. This distinguishes failures of learned prediction from failures that persist in the planner/search budget. That diagnostic reference is not available to this agent.','',
'The prototype now contains actual neural world-model learning and planning with persistent replay. It does not implement the full recurrent/latent architecture, general reasoning, a learned update procedure, recursive self-improvement or reliable memory compression. Episodes last 200 steps and search spans twenty; neither is a demonstration of very long-horizon competence. Strong model-free and uncertainty-aware model-based comparisons remain missing.']
(HERE/'E14-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e14_summary.json').write_text(json.dumps(dict(rows=rows,costs=costs,probes=probes),indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for arm,color in [('random','#a1a1aa'),('replay_mpc1','#64748b'),('replay_mpc20','#007f82'),('context_mpc20','#b45309'),('guarded_mpc20','#7c3aed')]:
    for i,seed in enumerate(data['protocol']['seeds']):
        ys=[r['evaluation_mean_return'] for r in rows if r['seed']==seed and r['arm']==arm]
        axes[i].plot(range(4),ys,marker='o',label=arm.replace('_',' '),color=color,linewidth=1.4)
for ax,seed in zip(axes,data['protocol']['seeds']):
    ax.set_title(f'Training seed {seed}');ax.set_xticks(range(4),['g=10','g=30','g=5','return g=10']);ax.set_ylabel('Mean frozen evaluation return');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8);fig.suptitle('E14: learned-model planning, four evaluation episodes per stage')
fig.savefig(HERE/'results/e14_planning.png',dpi=180)
print(json.dumps(costs,indent=2))
