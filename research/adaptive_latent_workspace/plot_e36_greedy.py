"""Means and every case; descriptive paired intervention, no inferred CI."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, target):
    data=json.loads(source.read_text())
    assert data['status']==data['audit']['status']=='complete'
    names=['static_ttt','first_order_ttt','e2e_ttt','counts']
    labels=['Static + updates','First-order meta','E2E meta','Rolling counts']
    regimes=['stationary','recurring','drifting','noisy']
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    colors=['#89949f','#147c88']
    for ax,regime in zip(axes.flat,regimes):
        for i,name in enumerate(names):
            rows=[r for r in data['cases'] if r['condition']==name and r['regime']==regime]
            for j,key in enumerate(['original_goals','greedy_goals']):
                values=[r[key] for r in rows]
                center=i+(j-.5)*.34
                ax.bar(center,np.mean(values),width=.31,color=colors[j],alpha=.8,
                       label=['Original soft selection','Greedy, same exploration'][j] if i==0 else None)
                offsets=np.linspace(-.10,.10,len(values))
                ax.scatter(center+offsets,values,s=14,color='#18252d',zorder=3,alpha=.8)
        ax.set_title(regime.title(),loc='left',fontweight='bold')
        ax.set_xticks(range(4),labels,fontsize=9)
        ax.set_ylim(0,310)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=.18)
        ax.set_ylabel('Goals completed / 512 real actions')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('Action selection changes the usefulness of a learned model',x=.06,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.045,'Bars: means. Dots: all cases; three neural initializations share two worlds per regime.\n'
             'Post hoc intervention on eight reused worlds. Different actions change experienced evidence.\n'
             'No retraining; identical query/update counts. These results do not establish general reasoning or efficient self-improvement.',fontsize=10)
    fig.subplots_adjust(top=.86,bottom=.17,left=.07,right=.98,hspace=.3)
    fig.savefig(target,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plot(a.input,a.out)
