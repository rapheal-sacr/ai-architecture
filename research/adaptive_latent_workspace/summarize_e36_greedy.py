"""All-case summary of the audited post hoc E36 policy intervention."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def summarize(folder, parent, dest):
    raw=json.loads((folder/'complete.json').read_text())
    audit=json.loads((folder/'audit.json').read_text())
    assert raw['status']==audit['status']=='complete'
    assert audit['scored_complete_sha256']==hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest()
    assert raw['intervention']['parent_complete_sha256']==hashlib.sha256((parent/'complete.json').read_bytes()).hexdigest()
    assert not raw['intervention']['preflight'] and len(raw['evaluation'])==80
    prior=json.loads((parent/'complete.json').read_text())
    lookup={}
    for c in prior['evaluation']:
        name=c['condition'] if isinstance(c['condition'],str) else c['condition']['name']
        lookup[c.get('seed'),name,c['world']]=c
    specs={s['name']:s for s in json.loads((parent/'data.json').read_text())['evaluation']}
    cases=[]
    groups=defaultdict(list)
    for c in raw['evaluation']:
        name=c['condition'] if isinstance(c['condition'],str) else c['condition']['name']
        p=lookup[c.get('seed'),name,c['world']]
        observed=json.loads(Path(c['result_file']).read_text())
        assert sum(r['reward'] for r in observed['rows'])==c['goals']
        assert sum(r['correct'] for r in observed['rows'])==c['correct']
        case=dict(seed=c.get('seed'),condition=name,world=c['world'],regime=specs[c['world']]['regime'],
                  original_goals=p['goals'],greedy_goals=c['goals'],goal_difference=c['goals']-p['goals'],
                  original_accuracy=p['correct']/p['actions'],greedy_accuracy=c['correct']/c['actions'],
                  original_nll=p['mean_nll'],greedy_nll=c['mean_nll'],
                  original_acting_seconds=p['acting_seconds'],greedy_acting_seconds=c['acting_seconds'],
                  work=c['work'],storage=c['storage'],
                  observed_pairs=len({(r['observation'],r['executed_action']) for r in observed['rows']}),
                  result_file=c['result_file'],result_sha256=hashlib.sha256(Path(c['result_file']).read_bytes()).hexdigest())
        assert c['work']==p['work']
        if name!='counts':
            assert c['storage']==p['storage']
        cases.append(case)
        groups[name,case['regime']].append(case)
    aggregates=[]
    for (name,regime),rows in groups.items():
        aggregates.append(dict(condition=name,regime=regime,cases=len(rows),worlds=len({r['world'] for r in rows}),
            better=sum(r['goal_difference']>0 for r in rows),tied=sum(r['goal_difference']==0 for r in rows),
            worse=sum(r['goal_difference']<0 for r in rows),
            **{f'mean_{k}':statistics.mean(r[k] for r in rows) for k in ['original_goals','greedy_goals','goal_difference','original_accuracy','greedy_accuracy','original_acting_seconds','greedy_acting_seconds']}))
    result=dict(status='complete',intervention=raw['intervention'],audit=audit,cases=cases,aggregates=aggregates,
                scoring_seconds=raw['seconds'],input_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
                scope='Reused eight worlds, post hoc intervention. Different policies cause different experienced data. Not independent seed-world trials or fresh transfer.')
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# E36 greedy intervention: all audited pairs','',result['scope'],'',
           '| Model | Regime | Cases / worlds | Soft goals | Greedy goals | Better / tied / worse |',
           '|---|---|---:|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['condition']} | {r['regime']} | {r['cases']} / {r['worlds']} | {r['mean_original_goals']:.2f} | {r['mean_greedy_goals']:.2f} | {r['better']} / {r['tied']} / {r['worse']} |")
    lines+=['','All80cases execute512actions. Neural model-query, gradient-update and Bellman','work totals equal their original cases exactly; persistent neural tensors also','match. Counts history occupancy may differ with experienced trajectories.','Float64 greedy Bellman arithmetic changes per-operation cost; original acting','timing sometimes overlapped a training audit. Timings are not an isolated','algorithmic speed comparison. Training was reused, not repeated or free.','',
            f"New audit: {audit['evaluation_actions']} actions, {audit['evaluation_updates']} updates, "
            f"{audit['exact_checkpoints']-36} evaluation checkpoints. The36training checkpoints "
            'are inherited from the hashed prior audit, not newly replayed.','',
            'Full paired cases, observed costs, prediction metrics and artifact hashes are in summary.json.']
    (dest/'tables.md').write_text('\n'.join(lines)+'\n')
    print('E36_GREEDY_SUMMARIZED',len(cases),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    summarize(a.input,a.parent,a.out)
