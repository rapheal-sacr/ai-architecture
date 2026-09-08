"""Post hoc privileged-model intervention; no learned capability credit.

Execute the original soft planner and greedy variants on identical frozen
world definitions with exact CURRENT transition probabilities. Never reveal
future goals or future noise realizations. Keep all policies/worlds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from action_navigation import NavigationObservation
from action_planning import PlanningConfig,goal_values,sample_action
from audit_e36 import direct_world_check,independent_plan
from e36_action_learning import plan
from e36_data import make_world,private_schedule


def run(folder,out):
    torch.set_num_threads(1)
    assert not out.exists()
    out.mkdir(parents=True)
    started=time.perf_counter()
    data=json.loads((folder/'data.json').read_text())
    complete=json.loads((folder/'complete.json').read_text())
    audit=json.loads((folder.parent/'e36-audit.json').read_text())
    assert audit['status']=='complete'
    assert audit['scored_complete_sha256']==hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest()
    protocol=complete['manifest']['protocol']
    cfg=PlanningConfig(**protocol['planning'])
    n,a=protocol['states'],protocol['actions']
    result=dict(status='running',scope='post hoc privileged current-map/noise diagnostic, not learned performance',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_data_sha256=hashlib.sha256((folder/'data.json').read_bytes()).hexdigest(),
                cases=[],independently_replayed_actions=0,max_independent_policy_error=0.)
    for spec in data['evaluation']:
        schedule=private_schedule(spec)
        for mode in ['original_soft','greedy_same_exploration','greedy_no_exploration']:
            world=make_world(spec)
            generator=random.Random(spec['policy_seed'])
            rows=[]
            clock=time.perf_counter()
            for t in range(protocol['evaluation_steps']):
                obs=world.observe()
                mapping=torch.tensor(spec['maps'][schedule[t]])
                probability=(1-spec['noise_rate'])*torch.nn.functional.one_hot(mapping,n).float()+spec['noise_rate']/n
                if mode=='original_soft':
                    policy=plan(probability,obs,cfg)
                    # Log zero is intentional: the independent reference handles
                    # these impossible transitions as zero softmax probabilities.
                    independent=independent_plan(probability.double().log().numpy(),obs.state,obs.goal,protocol)
                    error=float(abs(independent-policy.numpy()).max())
                    assert error<3e-6
                    result['max_independent_policy_error']=max(result['max_independent_policy_error'],error)
                else:
                    q=goal_values(probability,obs.goal,cfg)[obs.state]
                    best=(q==q.max()).float()
                    best/=best.sum()
                    epsilon=cfg.exploration if mode=='greedy_same_exploration' else 0.
                    policy=(1-epsilon)*best+epsilon/a
                action=sample_action(policy,generator.random())
                _,_,receipt=world.step(action)
                rows.append(receipt|dict(action_probabilities=policy.tolist()))
            seconds=time.perf_counter()-clock
            final=direct_world_check(spec,rows,n)
            result['independently_replayed_actions']+=len(rows)
            case=dict(world=spec['name'],regime=spec['regime'],policy=mode,goals=final['completed'],actions=len(rows),
                      seconds=seconds,rows=rows,bellman_transition_terms=len(rows)*cfg.horizon*n*n*a,
                      learned_state_bytes=0,privileged_information='current map and true noise rate; no future goals or realized noise')
            result['cases'].append(case)
            print('ORACLE_PLANNER',case['world'],mode,case['goals'],flush=True)
    result.update(status='complete',seconds=time.perf_counter()-started)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('E36_ORACLE_PLANNER_DIAGNOSED',result['independently_replayed_actions'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.input,a.out)
