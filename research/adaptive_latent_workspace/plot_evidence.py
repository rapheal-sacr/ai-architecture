from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
cases=[('E5: abrupt recurring functions','e5_summary.json','active_first_isolation',None),
       ('E4: independent changing-input stream','e4_summary.json','isolated_adaptation',None),
       ('E6: sample-tested sharing','e6_summary.json','guarded_sharing','external'),
       ('E7: replay and feature renewal','e7_summary.json','replay_cbp_gnt',None)]
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for ax,(title,file,arm,case) in zip(axes.flat,cases):
    rows=json.loads((HERE/'results'/file).read_text())
    if case:rows=[r for r in rows if r['case']==case]
    d={r['arm']:r for r in rows};ours=d[arm];ref=d['context_replay']
    x=ours['seconds']/ref['seconds'];y=ours['mse']/ref['mse']
    ax.axhline(1,color='#8c8c8c',lw=1,ls='--');ax.axvline(1,color='#8c8c8c',lw=1,ls='--')
    ax.scatter([1],[1],marker='x',s=80,color='#555555',label='Context replay')
    ax.scatter([x],[y],s=110,color='#007f82' if x<1 and y<1 else '#b65b30',zorder=3)
    ax.annotate(f'{arm.replace("_"," ")}\nTime {x:.2f}×; error {y:.2f}×',(x,y),xytext=(0,10),
        textcoords='offset points',ha='center',fontsize=9)
    ax.set_title(title,fontsize=11);ax.set_xlim(.6,1.95);ax.set_ylim(.2,1.7)
    ax.set_xlabel('Measured runtime / context-replay runtime');ax.set_ylabel('Whole-stream MSE / context-replay MSE')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.suptitle('Lower error and lower cost have not transferred together\nEach panel uses its own frozen experiment and paired baseline',fontsize=13)
fig.savefig(HERE/'results/evidence_map.png',dpi=170)
fig.savefig(HERE/'results/evidence_map.svg')
