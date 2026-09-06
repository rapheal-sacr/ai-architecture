from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
a=json.loads((HERE/'results/e20_summary.json').read_text());b=json.loads((HERE/'results/e21_summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
styles=[('inferred_ema',16,'Inferred\nsmall'),('inferred_ema',64,'Inferred\nwide'),('oracle_onehot_diagnostic',16,'Oracle\nsmall'),('oracle_onehot_diagnostic',64,'Oracle\nwide')]
r=[next(r for r in a if r['family']=='conflicting_functions' and r['context_mode']==m and r['width']==w) for m,w,_ in styles]
axes[0].bar(np.arange(4),[x['mean_mse'] for x in r],color=['#527a9f','#91b6d4','#bd762f','#dca26e'])
axes[0].set_xticks(np.arange(4),[s[2] for s in styles]);axes[0].set_ylabel('Mean prequential MSE');axes[0].set_title('E20: correct context helps more than width\nConflicting functions; oracle is diagnostic only',fontsize=10)
for family,color in [('conflicting_functions','#276b9f'),('input_shift','#ba713c')]:
    rows={(r['context_decay'],r['assignment']):r for r in b if r['family']==family}
    order=[(.95,'prior_assignment'),(.95,'posterior_assignment'),(.5,'prior_assignment'),(.5,'posterior_assignment')]
    base=rows[order[0]]['mean_mse'];values=[rows[k]['mean_mse']/base for k in order]
    axes[1].plot(np.arange(4),values,'o-',color=color,label=family.replace('_',' '))
axes[1].set_xticks(np.arange(4),['Slow\nprior write','Slow\nposterior write','Fast\nprior write','Fast\nposterior write'])
axes[1].axhline(1,color='gray',ls='--',lw=.8);axes[1].set_ylabel('MSE / slow-prior baseline');axes[1].legend(fontsize=8)
axes[1].set_title('E21: a cheap causal repair has negative transfer\nSame student capacity and replay budget',fontsize=10)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Inference and memory assignment are binding implementation choices\nSeparate fresh seed sets in E20 and E21; means of four worlds per family',fontsize=11)
fig.savefig(HERE/'results/e20_e21_context_constraints.png',dpi=180);print(HERE/'results/e20_e21_context_constraints.png')
