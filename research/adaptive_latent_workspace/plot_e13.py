from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results/e13_complete.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),layout='constrained')
for arm,color,label in [('context_replay','#64748b','Context replay'),('guarded_sharing','#007f82','Protected sharing')]:
    rs=[r for r in data['records'] if r['arm']==arm];curves=[];work=[]
    for r in rs:
        x=np.array([b['end_observation'] for b in r['blocks']]);loss=np.cumsum([b['mse']*b['observations'] for b in r['blocks']])/x
        f=np.array([b['forward_examples'] for b in r['blocks']]);curves.append(loss);work.append(f)
        axes[0].plot(x/1e6,loss,color=color,alpha=.25,linewidth=1)
        axes[1].plot(x/1e6,f/1e6,color=color,alpha=.25,linewidth=1)
    axes[0].plot(x/1e6,np.mean(curves,axis=0),color=color,label=label,linewidth=2)
    axes[1].plot(x/1e6,np.mean(work,axis=0)/1e6,color=color,label=label,linewidth=2)
for ax in axes:
    ax.axvline(1,color='#999999',linestyle=':',linewidth=1);ax.set_xlabel('Observed examples (millions)');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].set_title('Error after the preserved first million');axes[0].set_xlim(.9,5);axes[0].set_ylim(.05,.13);axes[0].set_ylabel('Cumulative prequential mean squared error');axes[0].legend(fontsize=9)
axes[1].set_title('Additional model work remains substantial');axes[1].set_ylabel('Cumulative model forward examples (millions)')
fig.suptitle('E13: exact first-million prefix, then continued input changes\nBold: mean of two streams; faint: individual streams')
fig.savefig(HERE/'results/e13_amortization.png',dpi=180)
