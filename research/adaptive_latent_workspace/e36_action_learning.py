"""E36 predictive meta-training and closed-loop evaluation; no goal-policy training."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import time

import torch

from action_e2e import ActionE2E
from action_navigation import ObservedTransitionPlanner, domain_seed, digest
from action_planning import PlanningConfig, goal_values, sample_action, transition_logits
from e2e_core import Config,E2ECore
from e36_data import prepare,make_world,private_schedule


SOURCES=['e36_action_learning.py','e36_data.py','protocol_e36.json','e2e_core.py',
         'action_e2e.py','action_navigation.py','action_planning.py']


def add_work(total,extra):
    for key,value in extra.items():
        if key!='inner_grad_norm':
            total[key]=total.get(key,0)+value


def train_example(model,example,arm,protocol):
    adapter=ActionE2E(model,states=protocol['states'],actions=protocol['actions'])
    records=example['records']  # No map, goal or future label enters forecast.
    state=adapter.initial_state(records[0]['observation'])
    losses,logits,total=[],[],{}
    mode={'static':'frozen','first_order':'first_order','e2e':'second_order'}[arm]
    for row in records:
        assert state.observation==row['observation']
        forecast=adapter.forecast(state,row['executed_action'])
        loss,state,work,receipt=adapter.observe(forecast,row['outcome'],mode=mode)
        assert all(receipt[k]==row[k] for k in ['step','observation','executed_action','outcome'])
        losses.append(loss)
        logits.append(forecast.logits.detach())
        add_work(total,work)
    return torch.stack(losses).mean(),torch.stack(logits),total


def plan(probabilities,observation,cfg):
    q=goal_values(probabilities,observation.goal,cfg)[observation.state]
    p=(1-cfg.exploration)*(q/cfg.temperature).softmax(-1)+cfg.exploration/len(q)
    return p


class CountControl:
    def __init__(self,states,actions,window):
        self.states,self.actions,self.window=states,actions,window
        self.rows=[[deque(maxlen=window) for _ in range(actions)] for _ in range(states)]
        self.counts=torch.zeros(states,actions,states,dtype=torch.float32)

    def probabilities(self):
        counts=self.counts+.25
        return counts/counts.sum(-1,keepdim=True)

    def observe(self,s,a,y):
        row=self.rows[s][a]
        if len(row)==self.window:
            self.counts[s,a,row[0]]-=1
        row.append(y)
        self.counts[s,a,y]+=1

    def payload(self):
        return dict(rows=[[list(q) for q in row] for row in self.rows],counts=self.counts.clone(),window=self.window)


def evaluate(model,spec,condition,protocol,out,*,control=None):
    out.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    world=make_world(spec)
    policy_rng=random.Random(spec['policy_seed'])
    cfg=PlanningConfig(**protocol['planning'])
    n,a=protocol['states'],protocol['actions']
    if control is None:
        adapter=ActionE2E(model,states=n,actions=a)
        state=adapter.initial_state(world.observe().state).detached()
        initial_parameters={k:v.detach().clone() for k,v in model.state_dict().items()}
    elif control=='bfs':
        controller=ObservedTransitionPlanner(n,a)
    elif control=='counts':
        controller=CountControl(n,a,protocol['count_control_window'])
    schedule=private_schedule(spec)
    rows,full_logits=[],[]
    total={}
    checkpoints=[]
    acting_seconds=0.
    for t in range(protocol['evaluation_steps']):
        obs=world.observe()
        tick=time.perf_counter()
        work={}
        map_prediction=None
        policy=None
        if control is None:
            # Inference map queries have no autograd history. The separately
            # executed forecast below retains the graph for its real update.
            with torch.no_grad():
                scores,work=transition_logits(adapter,state)
                probabilities=scores.softmax(-1)
                policy=plan(probabilities,obs,cfg)
            action=sample_action(policy,policy_rng.random())
            forecast=adapter.forecast(state,action)
            predicted=int(forecast.logits.argmax())
            assert torch.allclose(forecast.logits,scores[obs.state,action],atol=3e-5,rtol=3e-5)
            full_logits.append(scores.cpu())
            map_prediction=scores.argmax(-1)
        elif control=='bfs':
            action=controller.act(obs)
            predicted=controller.records.get((obs.state,action),0)
            map_prediction=torch.tensor([[controller.records.get((s,j),0) for j in range(a)] for s in range(n)])
        elif control=='counts':
            probabilities=controller.probabilities()
            policy=plan(probabilities,obs,cfg)
            action=sample_action(policy,policy_rng.random())
            predicted=int(probabilities[obs.state,action].argmax())
            map_prediction=probabilities.argmax(-1)
        elif control=='random':
            action=policy_rng.randrange(a)
            predicted=0  # Uniform predictive distribution's deterministic tie.
        else:
            raise ValueError(control)
        if control is None or control=='counts':
            work.update(bellman_iterations=cfg.horizon,bellman_transition_terms=cfg.horizon*n*n*a)
        # No private map/schedule labels above this line enter a decision.
        new,reward,receipt=world.step(action)
        if control is None:
            loss,state,real_work,learning_receipt=adapter.observe(forecast,new.state,mode=condition['update'])
            assert learning_receipt['executed_action']==receipt['executed_action']
            assert learning_receipt['outcome']==receipt['outcome']
            loss_value=float(loss.detach())
            add_work(work,real_work)
            state=state.detached()
        elif control=='bfs':
            controller.observe_transition(obs.state,action,new.state)
            loss_value=None  # No invented calibrated distribution for deterministic BFS.
        elif control=='counts':
            loss_value=-math.log(float(probabilities[obs.state,action,new.state]))
            controller.observe(obs.state,action,new.state)
        else:
            loss_value=math.log(n)
        acting_seconds+=time.perf_counter()-tick
        add_work(total,work)
        true_map=torch.tensor(spec['maps'][schedule[t]])
        row=dict(**receipt,prediction=predicted,correct=int(predicted==new.state),nll=loss_value,
                 private_diagnostic=dict(slot=schedule[t],map_correct=int((map_prediction==true_map).sum()) if map_prediction is not None else None),
                 work=work)
        if policy is not None:
            row['action_probabilities']=policy.tolist()
        rows.append(row)
        if (t+1)%protocol['checkpoint_interval_eval']==0 or t+1==protocol['evaluation_steps']:
            payload=dict(step=t+1,world=world.payload(),policy_rng=policy_rng.getstate())
            if control is None:
                payload['learner']=adapter.payload(state)
            elif control!='random':
                payload['learner']=controller.payload()
            filename=out/f'state-{t+1}.pt'
            torch.save(payload,filename)
            checkpoints.append(str(filename))
    if control is None:
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial_parameters.items())
        torch.save(torch.stack(full_logits),out/'map-logits.pt')
        storage=dict(base_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                     persistent_tensor_bytes=state.stream.bytes(),base_parameters=sum(p.numel() for p in model.parameters()))
    elif control=='bfs':
        storage=dict(logical_int64_record_bytes=len(controller.records)*3*8,entries=len(controller.records),
                     edge_visits=controller.edge_visits,writes=controller.writes)
    elif control=='counts':
        storage=dict(count_tensor_bytes=controller.counts.numel()*4,
                     logical_uint8_history_bytes=sum(len(q) for row in controller.rows for q in row),
                     observations=protocol['evaluation_steps'])
    else:
        storage={}
    result=dict(world=spec['name'],condition=condition if control is None else control,
                goals=sum(r['reward'] for r in rows),correct=sum(r['correct'] for r in rows),
                mean_nll=sum(r['nll'] for r in rows)/len(rows) if rows[0]['nll'] is not None else None,
                actions=len(rows),rows=rows,work=total,storage=storage,checkpoints=checkpoints,
                acting_seconds=acting_seconds,total_seconds=time.perf_counter()-begin)
    if control is None:
        result['base_parameters_unchanged']=True
    (out/'result.json').write_text(json.dumps(result)+'\n')
    return {k:v for k,v in result.items() if k!='rows'}|dict(result_file=str(out/'result.json'))


def run(out,development=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parent
    protocol=json.loads((root/'protocol_e36.json').read_text())
    if development:
        protocol.update(seeds=[38001],outer_steps=2,train_durations=[8,8,8],evaluation_steps=32,
                        evaluation_worlds_per_regime=1,checkpoint_interval_train=1,checkpoint_interval_eval=16,
                        evaluation_duration_range=[8,12])
        protocol['config'].update(width=16,hidden=24,heads=2,layers=3)
    else:
        for filename in SOURCES:
            subprocess.run(['git','diff','--exit-code','HEAD','--',str(root/filename)],check=True,stdout=subprocess.PIPE)
            subprocess.run(['git','ls-files','--error-unmatch',str(root/filename)],check=True,stdout=subprocess.PIPE)
    assert not out.exists() or not any(out.iterdir()), 'Never overwrite a run'
    out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),development=development,
                  protocol=protocol,sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in SOURCES},
                  runtime=dict(torch=torch.__version__,threads=1,device='cpu'),status='running')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=prepare(protocol)
    (out/'data.json').write_text(json.dumps(data)+'\n')
    manifest['data_sha256']=hashlib.sha256((out/'data.json').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cases={}
    for seed in protocol['seeds']:
        for arm in protocol['training_arms']:
            model=E2ECore(Config(**protocol['config']),seed=domain_seed(seed,'e36_init'))
            opt=torch.optim.AdamW(model.parameters(),lr=protocol['outer_lr'],weight_decay=protocol['outer_weight_decay'])
            folder=out/f'{seed}-{arm}'
            folder.mkdir()
            cases[seed,arm]=dict(model=model,optimizer=opt,folder=folder,training=[],work={})
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),folder/'training-0.pt')
    for step in range(protocol['outer_steps']):
        for (seed,arm),case in cases.items():
            tick=time.perf_counter()
            opt,model=case['optimizer'],case['model']
            opt.zero_grad(set_to_none=True)
            example=data['training'][str(seed)][step]
            loss,logits,work=train_example(model,example,arm,protocol)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['outer_clip'])
            assert torch.isfinite(loss) and torch.isfinite(norm)
            opt.step()
            add_work(case['work'],work)
            case['training'].append(dict(step=step+1,loss=float(loss.detach()),outer_grad_norm=float(norm),
                                         work=work,seconds=time.perf_counter()-tick,example_sha256=digest(example)))
            torch.save(logits,case['folder']/f'train-logits-{step+1}.pt')
            if (step+1)%protocol['checkpoint_interval_train']==0 or step+1==protocol['outer_steps']:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1),case['folder']/f'training-{step+1}.pt')
                (case['folder']/'training.json').write_text(json.dumps(case['training'])+'\n')
        if (step+1)%16==0 or development:
            print('E36_TRAIN',step+1,{f'{s}-{a}':round(c['training'][-1]['loss'],4) for (s,a),c in cases.items()},flush=True)
    results=[]
    for seed in protocol['seeds']:
        for condition in protocol['evaluation_arms']:
            model=cases[seed,condition['training']]['model']
            for spec in data['evaluation']:
                folder=out/f'eval-{seed}-{condition["name"]}-{spec["name"]}'
                result=evaluate(model,spec,condition,protocol,folder)
                result['seed']=seed
                results.append(result)
                (out/'evaluation-progress.json').write_text(json.dumps(results)+'\n')
                print('E36_EVAL',seed,condition['name'],spec['name'],result['goals'],round(result['mean_nll'],4),flush=True)
    for control in ['random','bfs','counts']:
        for spec in data['evaluation']:
            result=evaluate(None,spec,{},protocol,out/f'control-{control}-{spec["name"]}',control=control)
            results.append(result)
            print('E36_CONTROL',control,spec['name'],result['goals'],flush=True)
    training=[dict(seed=seed,arm=arm,work=case['work'],seconds=sum(r['seconds'] for r in case['training']),
                   final_loss=case['training'][-1]['loss'],training_file=str(case['folder']/'training.json'),
                   checkpoint=str(case['folder']/f'training-{protocol["outer_steps"]}.pt')) for (seed,arm),case in cases.items()]
    manifest['status']='scored_complete_await_audit'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    complete=dict(status='complete',manifest=manifest,training=training,evaluation=results,
                  process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  seconds=time.perf_counter()-begin)
    (out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('E36_COMPLETE',len(results),round(complete['seconds'],3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--development',action='store_true')
    args=p.parse_args()
    run(args.out,args.development)
