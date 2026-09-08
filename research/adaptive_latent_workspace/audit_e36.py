"""E36 exact replay plus separate environment, update and metric checks.

Training replay reuses train_example; evaluation updates bypass ActionE2E.
All scored data remain fixed. Scored source files must be byte-identical.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from e2e_core import Config,E2ECore,StreamState
from action_navigation import domain_seed,digest,ObservedTransitionPlanner,NavigationObservation
from action_planning import PlanningConfig,sample_action
from e36_action_learning import train_example,add_work,CountControl,plan


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            equal(x,y)
    else:
        assert a==b,(a,b)


def direct_world_check(spec,rows,states):
    slots=[x for x,n in zip(spec['slots'],spec['durations']) for _ in range(n)]
    goals=random.Random(spec['goal_seed'])
    noise=random.Random(spec['noise_seed'])
    state=0
    goal=goals.choice(list(range(1,states)))
    count=0
    for t,row in enumerate(rows):
        assert row['observation']==state and row['goal']==goal and row['step']==t+1
        state=spec['maps'][slots[t]][state][row['executed_action']]
        if spec['noise_rate'] and noise.random()<spec['noise_rate']:
            state=noise.randrange(states)
        reward=int(state==goal)
        count+=reward
        if reward:
            goal=goals.choice([i for i in range(states) if i!=goal])
        assert (row['outcome'],row['reward'],row['next_goal'])==(state,reward,goal)
    return dict(state=state,goal=goal,completed=count,rng=goals.getstate(),noise_rng=noise.getstate())


def independent_plan(scores,current,goal,protocol):
    # NumPy float64 explicit Bellman recurrence, separate from torch matmul.
    logits=np.asarray(scores,dtype=np.float64)
    probs=np.exp(logits-logits.max(axis=-1,keepdims=True))
    probs/=probs.sum(axis=-1,keepdims=True)
    cfg=protocol['planning']
    value=np.zeros(protocol['states'])
    for _ in range(cfg['horizon']):
        q=np.zeros((protocol['states'],protocol['actions']))
        for s in range(protocol['states']):
            for a in range(protocol['actions']):
                q[s,a]=sum(probs[s,a,y]*(1. if y==goal else cfg['discount']*value[y]) for y in range(protocol['states']))
        value=q.max(axis=1)
    q=q[current]/cfg['temperature']
    p=np.exp(q-q.max())
    p/=p.sum()
    return (1-cfg['exploration'])*p+cfg['exploration']/protocol['actions']


def run(folder,out,allow_source_change=False,stage='all',training_audit=None):
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    complete=None if stage=='training' else json.loads((folder/'complete.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    protocol=manifest['protocol']
    root=Path(__file__).parent
    changed=[]
    for filename,sha in manifest['sources'].items():
        if hashlib.sha256((root/filename).read_bytes()).hexdigest()!=sha:
            changed.append(filename)
    assert not changed or (allow_source_change and manifest['development'])
    assert hashlib.sha256((folder/'data.json').read_bytes()).hexdigest()==manifest['data_sha256']
    data=json.loads((folder/'data.json').read_text())
    result=dict(training_updates=0,training_records=0,evaluation_actions=0,evaluation_updates=0,
                exact_checkpoints=0,serial_map_queries=0,max_serial_map_error=0.,max_independent_policy_error=0.,
                development_source_changes=changed)
    models={}
    previous_seconds=0.
    if stage=='evaluation':
        assert training_audit is not None
        previous=json.loads(training_audit.read_text())
        assert previous['status']=='complete_training'
        assert previous['data_sha256']==manifest['data_sha256']
        assert previous['sources']==manifest['sources']
        assert previous['frozen_commit']==manifest['commit']
        for path,sha in previous['final_checkpoint_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        for key in ['training_updates','training_records','exact_checkpoints']:
            result[key]=previous[key]
        previous_seconds=previous['seconds']
        result['training_audit_file_sha256']=hashlib.sha256(training_audit.read_bytes()).hexdigest()
        for seed in protocol['seeds']:
            for arm in protocol['training_arms']:
                model=E2ECore(Config(**protocol['config']),seed=domain_seed(seed,'e36_init'))
                saved=torch.load(folder/f'{seed}-{arm}'/f'training-{protocol["outer_steps"]}.pt',weights_only=False)
                model.load_state_dict(saved['model'])
                models[seed,arm]=model
    for seed in ([] if stage=='evaluation' else protocol['seeds']):
        for arm in protocol['training_arms']:
            case=folder/f'{seed}-{arm}'
            model=E2ECore(Config(**protocol['config']),seed=domain_seed(seed,'e36_init'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['outer_lr'],weight_decay=protocol['outer_weight_decay'])
            initial=torch.load(case/'training-0.pt',weights_only=False)
            equal(model.state_dict(),initial['model'])
            equal(optimizer.state_dict(),initial['optimizer'])
            logs=json.loads((case/'training.json').read_text())
            assert len(logs)==protocol['outer_steps'], 'Final training checkpoints are not all available'
            for t,example in enumerate(data['training'][str(seed)]):
                direct_world_check(example['world'],example['records'],protocol['states'])
                optimizer.zero_grad(set_to_none=True)
                loss,logits,work=train_example(model,example,arm,protocol)
                assert logs[t]['example_sha256']==digest(example)
                assert float(loss.detach())==logs[t]['loss']
                equal(logits,torch.load(case/f'train-logits-{t+1}.pt',weights_only=True))
                equal(work,logs[t]['work'])
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['outer_clip'])
                assert float(norm)==logs[t]['outer_grad_norm']
                optimizer.step()
                result['training_updates']+=1
                result['training_records']+=len(example['records'])
                cp=case/f'training-{t+1}.pt'
                if cp.exists():
                    saved=torch.load(cp,weights_only=False)
                    equal(model.state_dict(),saved['model'])
                    equal(optimizer.state_dict(),saved['optimizer'])
                    result['exact_checkpoints']+=1
            models[seed,arm]=model
            print('E36_AUDIT_TRAIN',seed,arm,flush=True)
    if stage=='training':
        result.update(status='complete_training',seconds=time.perf_counter()-started,
                      frozen_commit=manifest['commit'],sources=manifest['sources'],data_sha256=manifest['data_sha256'],
                      final_checkpoint_sha256={str(folder/f'{seed}-{arm}'/f'training-{protocol["outer_steps"]}.pt'):
                          hashlib.sha256((folder/f'{seed}-{arm}'/f'training-{protocol["outer_steps"]}.pt').read_bytes()).hexdigest()
                          for seed in protocol['seeds'] for arm in protocol['training_arms']})
        out.write_text(json.dumps(result,indent=2)+'\n')
        print('E36_TRAINING_AUDITED',json.dumps(result),flush=True)
        return
    lookup={spec['name']:spec for spec in data['evaluation']}
    for case in complete['evaluation']:
        raw=json.loads(Path(case['result_file']).read_text())
        spec=lookup[raw['world']]
        direct=direct_world_check(spec,raw['rows'],protocol['states'])
        assert direct['completed']==raw['goals']
        assert sum(r['correct'] for r in raw['rows'])==raw['correct']
        condition=raw['condition']
        if isinstance(condition,str):
            n,a=protocol['states'],protocol['actions']
            generator=random.Random(spec['policy_seed'])
            controller=ObservedTransitionPlanner(n,a) if condition=='bfs' else CountControl(n,a,protocol['count_control_window']) if condition=='counts' else None
            checkpoints={int(Path(p).stem.split('-')[1]):p for p in raw['checkpoints']}
            for t,row in enumerate(raw['rows']):
                observation=NavigationObservation(row['observation'],row['goal'])
                if condition=='random':
                    action=generator.randrange(a)
                    predicted=0
                    assert row['nll']==math.log(n)
                elif condition=='bfs':
                    action=controller.act(observation)
                    predicted=controller.records.get((observation.state,action),0)
                    controller.observe_transition(observation.state,action,row['outcome'])
                else:
                    probabilities=controller.probabilities()
                    policy=plan(probabilities,observation,PlanningConfig(**protocol['planning']))
                    equal(policy.tolist(),row['action_probabilities'])
                    action=sample_action(policy,generator.random())
                    predicted=int(probabilities[observation.state,action].argmax())
                    assert row['nll']==-math.log(float(probabilities[observation.state,action,row['outcome']]))
                    controller.observe(observation.state,action,row['outcome'])
                    reference=torch.zeros_like(controller.counts)
                    for s in range(n):
                        for j in range(a):
                            for y in controller.rows[s][j]:
                                reference[s,j,y]+=1
                    equal(controller.counts,reference)
                assert action==row['executed_action'] and predicted==row['prediction']
                if t+1 in checkpoints:
                    saved=torch.load(checkpoints[t+1],weights_only=False)
                    if controller is not None:
                        equal(controller.payload(),saved['learner'])
                    equal(generator.getstate(),saved['policy_rng'])
                    for key,value in direct_world_check(spec,raw['rows'][:t+1],n).items():
                        equal(value,saved['world'][key])
                    result['exact_checkpoints']+=1
            result['evaluation_actions']+=len(raw['rows'])
            continue
        model=models[case['seed'],condition['training']]
        state=model.initial_state().detached()
        mapscores=torch.load(Path(case['result_file']).parent/'map-logits.pt',weights_only=True)
        policy_rng=random.Random(spec['policy_seed'])
        checkpoints={int(Path(p).stem.split('-')[1]):p for p in raw['checkpoints']}
        total={}
        for t,row in enumerate(raw['rows']):
            n,a=protocol['states'],protocol['actions']
            token=torch.tensor([row['observation'],n+row['executed_action']])
            logits,next_state,elements=model.chunk_logits(token,state)
            actual=logits[-1,:n]
            assert int(actual.argmax())==row['prediction']
            assert row['correct']==int(int(actual.argmax())==row['outcome'])
            loss=F.cross_entropy(actual[None],torch.tensor([row['outcome']]))
            assert float(loss.detach())==row['nll']
            torch.testing.assert_close(actual,mapscores[t,row['observation'],row['executed_action']],atol=3e-5,rtol=3e-5)
            p=independent_plan(mapscores[t].numpy(),row['observation'],row['goal'],protocol)
            err=float(abs(p-np.array(row['action_probabilities'])).max())
            assert err<3e-6
            result['max_independent_policy_error']=max(result['max_independent_policy_error'],err)
            cumulative=np.cumsum(np.array(row['action_probabilities'],dtype=np.float64))
            choice=min(int(np.sum(cumulative<=policy_rng.random())),a-1)
            assert choice==row['executed_action']
            if t%protocol['checkpoint_interval_eval']==0:
                with torch.no_grad():
                    independent=torch.stack([model.chunk_logits(torch.tensor([s,n+j]),state)[0][-1,:n] for s in range(n) for j in range(a)]).reshape(n,a,n)
                error=float((independent-mapscores[t]).abs().max())
                assert error<3e-5
                result['serial_map_queries']+=n*a
                result['max_serial_map_error']=max(result['max_serial_map_error'],error)
            slot=row['private_diagnostic']['slot']
            assert int((mapscores[t].argmax(-1)==torch.tensor(spec['maps'][slot])).sum())==row['private_diagnostic']['map_correct']
            fast=state.fast
            if condition['update']!='frozen':
                gradients=torch.autograd.grad(loss,tuple(fast.values()))
                norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
                scale=(model.cfg.clip/norm.clamp_min(1e-30)).clamp(max=1.)
                fast={k:v-model.cfg.inner_lr*(scale*g) for (k,v),g in zip(fast.items(),gradients)}
                result['evaluation_updates']+=1
            state=StreamState(fast,next_state.cache,next_state.position).detached()
            expected_work=dict(hypothetical_queries=n*a,forward_tokens=2*(n*a+1),
                               attention_score_elements=(n*a+1)*elements,inner_updates=int(condition['update']!='frozen'),
                               gradient_tokens=2*int(condition['update']!='frozen'),vectorized_batches=1,
                               bellman_iterations=protocol['planning']['horizon'],
                               bellman_transition_terms=protocol['planning']['horizon']*n*n*a,forwards=1)
            equal(expected_work,row['work'])
            result['evaluation_actions']+=1
            add_work(total,row['work'])
            if t+1 in checkpoints:
                saved=torch.load(checkpoints[t+1],weights_only=False)
                equal(state.payload(),saved['learner']['stream'])
                assert saved['learner']['observation']==row['outcome']
                assert saved['learner']['real_steps']==t+1
                equal(policy_rng.getstate(),saved['policy_rng'])
                prefix=direct_world_check(spec,raw['rows'][:t+1],n)
                for key,value in prefix.items():
                    equal(value,saved['world'][key])
                result['exact_checkpoints']+=1
        equal(total,raw['work'])
        assert abs(sum(r['nll'] for r in raw['rows'])/len(raw['rows'])-raw['mean_nll'])<1e-14
        print('E36_AUDIT_EVAL',case['seed'],condition['name'],raw['world'],flush=True)
    elapsed=time.perf_counter()-started
    result.update(status='complete',seconds=elapsed+previous_seconds,
                  current_stage_seconds=elapsed,prior_training_audit_seconds=previous_seconds,
                  scored_complete_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('E36_AUDITED',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-development-source-change',action='store_true')
    p.add_argument('--stage',choices=['all','training','evaluation'],default='all')
    p.add_argument('--training-audit',type=Path)
    a=p.parse_args()
    run(a.input,a.out,a.allow_development_source_change,a.stage,a.training_audit)
