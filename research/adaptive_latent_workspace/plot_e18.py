from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results/e18_complete.json').read_text());rs=data['records']
fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for ax,family in zip(axes,data['protocol']['families']):
    for arm,color,label in [('adam_01','#777777','Fixed Adam 0.01'),('online_meta_adam','#ce7839','Online conventional meta-update'),('online_self_applied','#276b9f','Online self-application')]:
        ratios=[]
        for rep in data['protocol']['controller_replicates']:
            for seed in data['protocol']['stream_seeds']:
                pair={r['arm']:r['result'] for r in rs if r['replicate']==rep and r['stream_seed']==seed and r['family']==family}
                baseline=np.cumsum(pair['frozen_bootstrap']['clean_batch_mse']);value=np.cumsum(pair[arm]['clean_batch_mse'])
                ratios.append(value/baseline)
        ratios=np.array(ratios);x=np.arange(1,ratios.shape[1]+1)*32
        # Show the continuation after the first ordinary/trial block, not a noisy initial ratio.
        selected=np.arange(47,len(x),48)
        ax.plot(x[selected],ratios.mean(0)[selected],color=color,label=label,lw=1.8)
        if arm!='adam_01':ax.fill_between(x[selected],ratios.min(0)[selected],ratios.max(0)[selected],color=color,alpha=.12)
    ax.axhline(1,color='black',ls='--',lw=.9);ax.set_title(family.replace('_',' ').capitalize())
    ax.set_xlabel('Observed stream examples');ax.set_ylabel('Cumulative MSE / frozen procedure');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='best')
fig.suptitle('E18: procedure changes during persistent learning\nLines: mean paired ratio; bands: range of four dependent replicate/world cases',fontsize=11)
fig.savefig(HERE/'results/e18_online_procedure.png',dpi=180);print(HERE/'results/e18_online_procedure.png')
