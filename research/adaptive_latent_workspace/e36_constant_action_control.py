"""Supplemental structural shortcut control; does not alter frozen E36.

Each action is a Hamiltonian cycle. Repeat fixed action0 or1, regardless of
observations/goals, on the already-frozen worlds. Added before scored action
results were available; both fixed policies are retained, never best-selected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from action_navigation import NavigationWorld


def run(folder,out):
    out.mkdir(parents=True,exist_ok=False)
    data=json.loads((folder/'data.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    protocol=manifest['protocol']
    n,actions=protocol['states'],protocol['actions']
    cases=[]
    checked=0
    for spec in data['evaluation']:
        for fixed in range(actions):
            start=time.perf_counter()
            world=NavigationWorld(spec['maps'],spec['durations'],spec['slots'],
                                  goal_seed=spec['goal_seed'],noise_rate=spec['noise_rate'],noise_seed=spec['noise_seed'])
            receipts=[world.step(fixed)[2] for _ in range(protocol['evaluation_steps'])]
            acting=time.perf_counter()-start
            # Independent physical replay bypasses NavigationWorld completely.
            goals=random.Random(spec['goal_seed'])
            noise=random.Random(spec['noise_seed'])
            state=0
            goal=goals.choice([s for s in range(n) if s!=state])
            schedule=[slot for slot,length in zip(spec['slots'],spec['durations']) for _ in range(length)]
            for t,row in enumerate(receipts):
                assert row['observation']==state and row['goal']==goal
                state=spec['maps'][schedule[t]][state][fixed]
                if spec['noise_rate'] and noise.random()<spec['noise_rate']:
                    state=noise.randrange(n)
                reward=int(state==goal)
                if reward:
                    goal=goals.choice([s for s in range(n) if s!=goal])
                assert row['outcome']==state and row['reward']==reward and row['next_goal']==goal
                assert row['executed_action']==fixed
                checked+=1
            case=dict(world=spec['name'],regime=spec['regime'],fixed_action=fixed,
                      actions=len(receipts),goals=sum(r['reward'] for r in receipts),
                      acting_and_receipt_seconds=acting,learned_state_bytes=0,
                      transition_queries=0,updates=0,receipts=receipts)
            cases.append(case)
    # For stationary/noiseless worlds, fresh goal distance is uniform1..N-1.
    # Independent renewal recurrence gives finite-horizon expected goal count.
    horizon=protocol['evaluation_steps']
    expectation=[0.]*(horizon+1)
    for t in range(1,horizon+1):
        expectation[t]=sum(1+expectation[t-d] for d in range(1,n) if d<=t)/(n-1)
    # Verify each released mapping has exactly the assumed cycle property.
    cycle_checks=0
    for spec in data['evaluation']:
        for mapping in spec['maps']:
            for a in range(actions):
                for origin in range(n):
                    state=origin
                    seen=set()
                    for _ in range(n):
                        assert state not in seen
                        seen.add(state)
                        state=mapping[state][a]
                    assert state==origin and len(seen)==n
                    cycle_checks+=1
    result=dict(status='complete',cases=cases,independently_checked_actions=checked,
                cycle_checks=cycle_checks,stationary_expected_goals=expectation[-1],
                stationary_asymptotic_goals_per_action=2/n,
                input_data_sha256=hashlib.sha256((folder/'data.json').read_bytes()).hexdigest(),
                frozen_e36_commit=manifest['commit'],
                own_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    for c in cases:
        print(c['world'],c['fixed_action'],c['goals'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.input,a.out)
