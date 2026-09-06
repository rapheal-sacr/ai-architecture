from pathlib import Path
import argparse,hashlib,json,time
import torch
import e9_recursive_updater as e9

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def proposals(cfg,rep,initial,use_self,log_path=None):
    phi=e9.leaves(initial);m=[torch.zeros_like(p) for p in phi];v=[torch.zeros_like(p) for p in phi]
    accepted=0;rows=[];prefix=rep*100000;start=time.perf_counter()
    anchors=[prefix+10000+i for i in range(cfg['retention_validation_tasks'])]
    for t in range(cfg['self_rounds']):
        started=time.perf_counter()
        train=[prefix+30000+t*10+i for i in range(cfg['tasks_per_meta_update'])]
        loss=e9.meta_loss(cfg,phi,train);grads=torch.autograd.grad(loss,phi);e9.COUNTS['meta_gradient_calls']+=1
        candidate,cm,cv=e9.update(phi,[g.detach() for g in grads],m,v,accepted+1,loss.detach(),
            [p.detach() for p in phi] if use_self else None,cfg,cfg['self_learning_rate'])
        candidate=e9.leaves(candidate)
        fresh=[prefix+40000+t*10+i for i in range(cfg['fresh_validation_tasks_per_round'])]
        before=e9.evaluate(cfg,phi,fresh);after=e9.evaluate(cfg,candidate,fresh)
        old=e9.evaluate(cfg,phi,anchors);new=e9.evaluate(cfg,candidate,anchors)
        accept=(all(x['query_mse'] is not None for x in (before,after,old,new)) and
            after['query_mse']<before['query_mse']*(1-cfg['accept_improvement_fraction']) and
            new['query_mse']<=old['query_mse']*(1+cfg['retention_tolerance_fraction']))
        if accept:phi=candidate;m=[z.detach() for z in cm];v=[z.detach() for z in cv];accepted+=1
        row=dict(round=t,accepted=accept,training_meta_loss=float(loss.detach()),fresh_before=before,fresh_after=after,
            anchor_before=old,anchor_after=new,seconds=time.perf_counter()-started)
        rows.append(row)
        if log_path:
            with log_path.open('a') as f:f.write(json.dumps(dict(phase='fixed_meta_adam',trial=row,cumulative_counts=e9.COUNTS.copy()))+'\n')
        print(json.dumps(dict(replicate=rep,phase='self_replay' if use_self else 'fixed_meta_adam',round=t,accepted=accept)),flush=True)
    return phi,dict(rounds=rows,accepted=accepted,seconds=time.perf_counter()-start,counts=e9.COUNTS.copy())

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    cfg=json.loads((HERE/'protocol_e9.json').read_text());spec=json.loads((HERE/'protocol_e11.json').read_text())
    old_identity={n:digest(HERE/n) for n in ('protocol_e9.json','e9_recursive_updater.py')}
    identity={n:digest(HERE/n) for n in ('protocol_e9.json','e9_recursive_updater.py','protocol_e11.json','e11_meta_control.py')}
    a.out.mkdir(parents=True,exist_ok=True)
    def load(rep,kind):
        path=a.runs/'e9'/f'{rep}-{kind}.pt'
        data=torch.load(path,map_location='cpu',weights_only=False)
        assert data['identity']==old_identity
        return data['phi'],dict(path=str(path),sha256=digest(path))
    if not a.execute:
        rep=spec['replicates'][0];initial,_=load(rep,'bootstrap');expected,_=load(rep,'self-applied')
        got,record=proposals(cfg,rep,initial,True)
        original=json.loads((a.runs/'e9'/f'{rep}.json').read_text())
        assert all(torch.equal(x,y) for x,y in zip(got,expected))
        assert [r['accepted'] for r in record['rounds']]==[r['accepted'] for r in original['self_rounds']]
        result=dict(identity=identity,replicate=rep,exact_final_tensor_equality=True,identical_admissions=True,
            verification_cost=record)
        (a.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print('E11 self-application replay preflight passed',flush=True);return
    records=[]
    for rep in spec['replicates']:
        dest=a.out/f'{rep}.json';trial_path=a.out/f'{rep}-trials.jsonl'
        if dest.exists():
            record=json.loads(dest.read_text());assert record['identity']==identity;records.append(record);continue
        assert not trial_path.exists(),'Partial log exists; inspect before restarting'
        for k in e9.COUNTS:e9.COUNTS[k]=0
        started=time.perf_counter();initial,boot_source=load(rep,'bootstrap');self_phi,self_source=load(rep,'self-applied')
        prior=json.loads((a.runs/'e9'/f'{rep}.json').read_text());assert prior['identity']==old_identity
        control,proposal_record=proposals(cfg,rep,initial,False,trial_path)
        torch.save(dict(phi=[p.detach() for p in control],identity=identity),a.out/f'{rep}-fixed-meta-adam.pt')
        final=[]
        for dist in spec['final_distributions']:
            for horizon in spec['final_horizons']:
                seeds=[rep*100000+spec['final_task_seed_offset']+i for i in range(spec['final_tasks'])]
                for name,phi in [('frozen_bootstrap',initial),('self_applied',self_phi),('fixed_meta_adam',control)]:
                    result=e9.evaluate(cfg,phi,seeds,steps=horizon,distribution=dist)
                    result.update(arm=name,distribution=dist,horizon=horizon);final.append(result)
                    with trial_path.open('a') as f:f.write(json.dumps(dict(phase='final_evaluation',result=result,cumulative_counts=e9.COUNTS.copy()))+'\n')
                    print(json.dumps(dict(replicate=rep,arm=name,distribution=dist,horizon=horizon,query_mse=result['query_mse'])),flush=True)
        record=dict(replicate=rep,identity=identity,bootstrap_source=boot_source,self_applied_source=self_source,
            inherited_cost=dict(bootstrap_seconds=prior['bootstrap_seconds'],self_trial_seconds=sum(x['seconds'] for x in prior['self_rounds']),
                prior_result_sha256=digest(a.runs/'e9'/f'{rep}.json')),
            proposals=proposal_record,final=final,cost_counts=e9.COUNTS.copy(),seconds=time.perf_counter()-started)
        dest.write_text(json.dumps(record,indent=2)+'\n');records.append(record)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E11_COMPLETE',flush=True)

if __name__=='__main__':main()
