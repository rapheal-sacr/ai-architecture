from pathlib import Path
import json,shutil,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e4/complete.json').read_text())
shutil.copyfile(RUNS/'e4/complete.json',HERE/'results/e4_complete.json')
rows=[]
for arm in data['protocol']['arms']:
    rs=[r for r in data['records'] if r['arm']==arm]
    rows.append(dict(arm=arm,mse=st.mean(r['mse'] for r in rs),seconds=st.mean(r['wall_seconds'] for r in rs),
        first_ten_blocks_mse=st.mean(b['mse'] for r in rs for b in r['blocks'][:10]),
        last_ten_blocks_mse=st.mean(b['mse'] for r in rs for b in r['blocks'][-10:]),
        modules=st.mean(r['costs']['module_count'] for r in rs),
        capacity_hits=st.mean(r['capacity_hits'] for r in rs),
        forward_examples=st.mean(r['costs']['forward_examples'] for r in rs),
        training_examples=st.mean(r['costs']['training_examples'] for r in rs),
        tensor_bytes=st.mean(r['costs']['stored_tensor_bytes'] for r in rs)))
(HERE/'results/e4_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E4: external transfer fails against context replay','',
'The unchanged isolated-adaptation learner loses to context-conditioned replay on both seeds of the pinned loss-of-plasticity repository\'s slowly changing regression generator. Both arms receive exactly the same million-example stream per seed. Only learner input/output dimensions change to match the external task.','',
'| Arm | Whole-stream MSE | First ten blocks | Last ten blocks | Seconds | Persistent modules | Capacity hits | Tensor bytes |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['arm']} | {r['mse']:.5f} | {r['first_ten_blocks_mse']:.5f} | {r['last_ten_blocks_mse']:.5f} | {r['seconds']:.2f} | {r['modules']:.0f} | {r['capacity_hits']:.0f} | {r['tensor_bytes']:.0f} |")
report += ['',
'The source teacher is a fixed LTU network. Fifteen observed input bits change slowly while five bits vary rapidly. It is not the hidden switching-function problem used for E1–E3. Shared prediction can be useful across changes because the conditional target function itself stays fixed.','',
'**Rejected scope:** the E2 rule is not established as an efficient general continual-learning architecture. High prediction error is not enough to infer that a new persistent context is needed. The observed slot exhaustion is consistent with inappropriate partitioning or limited representational capacity; this experiment does not distinguish those causes.','',
'**Interpretation limits:** this is an external-generator transfer screen, not a reproduction of the published loss-of-plasticity findings. We use batch size 32, two seeds, one million scored observations, 64-wide tanh learners and Adam, whereas the published protocols differ. No CBP arm was run, so no comparison with feature renewal is licensed. The generator creates an extra block; only the first million observations are scored. The source\'s locally generated pickle is never passed to a learner; only X and already-observed Y are. Stream hashes verify identical data across arms.','',
'Blocks contain at least 10,000 samples (usually 10,016 because batches straddle the boundary). Exact observation counts are stored. Timings are local CPU measurements, with no simultaneous long experiment; small reporting/setup tasks may overlap.','',
'The next design must test whether new observations require a different conditional rule or can be incorporated into shared learning, while charging the test and consolidation costs. Simply enlarging the module cap does not resolve the binding storage/routing cost exposed here.']
(HERE/'E4-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
colors={'online':'#718096','context_replay':'#ad5e00','isolated_adaptation':'#007f82'}
for ax,seed in zip(axes,data['protocol']['seeds']):
    for r in data['records']:
        if r['seed']!=seed:continue
        blocks=r['blocks'];window=5
        xx=[st.mean(b['end_observation'] for b in blocks[i:i+window])/1e6 for i in range(0,len(blocks),window)]
        yy=[st.mean(b['mse'] for b in blocks[i:i+window]) for i in range(0,len(blocks),window)]
        ax.plot(xx,yy,label=r['arm'].replace('_',' '),color=colors[r['arm']])
    ax.set_title(f'External teacher seed {seed}');ax.set_xlabel('Million observations');ax.set_ylabel('Prequential MSE (five-block mean)')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(HERE/'results/e4_external.png',dpi=180)
print(json.dumps(rows,indent=2))
