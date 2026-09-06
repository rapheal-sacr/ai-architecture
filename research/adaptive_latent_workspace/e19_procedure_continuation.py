from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from e17_persistent_student import PersistentStudent,make_stream
from e18_online_procedure import segment_results
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def equal_state(a,b,exclude=()):
    assert set(a)==set(b)
    for k in a:
        if k in exclude:continue
        x,y=a[k],b[k]
        if isinstance(x,torch.Tensor):assert torch.equal(x,y),k
        elif isinstance(x,dict):equal_state(x,y)
        elif isinstance(x,list):
            assert len(x)==len(y)
            for u,v in zip(x,y):
                if isinstance(u,torch.Tensor):assert torch.equal(u,v),k
                else:assert u==v,k
        else:assert x==y,k

def continuation(cfg,seed,family,prefix,length):
    short=cfg.copy();short['updates']=prefix
    longer=cfg.copy();longer['updates']=prefix+length
    old,old_segments=make_stream(short,seed,family);full,full_segments=make_stream(longer,seed,family)
    for i,(before,after) in enumerate(zip(old,full)):
        assert all(torch.equal(a,b) for a,b in zip(before,after)),f'Changed prefix batch {i}'
    clipped=[dict(s,stop=min(s['stop'],prefix)) for s in full_segments if s['start']<prefix]
    assert clipped==old_segments
    segments=[dict(s,start=max(0,s['start']-prefix),stop=s['stop']-prefix,left_truncated=s['start']<prefix)
        for s in full_segments if s['stop']>prefix]
    check=dict(prefix_batches=prefix,exact_x_y_clean=True,exact_clipped_segments=True,
        original_last_segment=old_segments[-1],continuing_first_segment=segments[0])
    return full[prefix:],segments,check

def preflight(cfg,sc,source,bootstrap):
    a=PersistentStudent.restore(copy.deepcopy(source));b=PersistentStudent.restore(copy.deepcopy(source));b.phi=[p.clone() for p in bootstrap]
    equal_state(a.state(),b.state(),exclude=('phi',))
    before_a=[p.clone() for p in a.phi];before_b=[p.clone() for p in b.phi]
    rows,segs,check=continuation(sc,95,'conflicting_functions',37,8)
    assert torch.equal(a.predict(rows[0][0]),b.predict(rows[0][0]))
    for x,y,_ in rows:
        a.observe(x,y);b.observe(x,y)
    assert a.step==b.step==source['step']+8
    for name in ('context','replay_x','replay_y','rng'):
        assert torch.equal(a.state()[name],b.state()[name]),name
    assert all(torch.equal(x,y) for x,y in zip(before_a,a.phi))
    assert all(torch.equal(x,y) for x,y in zip(before_b,b.phi))
    return dict(non_procedure_state_equal=True,first_prediction_equal=True,procedures_frozen=True,
        paired_experience_state_equal=True,continued_global_step=a.step,prefix_check=check,
        new_student_updates=16,new_observations_per_branch=256)

def run_pair(spec,sc,record,bootstrap,rows,segments,mode,out,identity):
    started=time.perf_counter();checkpoint=Path(record['result']['checkpoint'])
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert source['identity']==identity['e18_identity'];state=source['learner_state']
    assert state['step']==spec['prefix_updates'] and state['size']==sc['replay_capacity']
    learners=[PersistentStudent.restore(copy.deepcopy(state)) for _ in range(2)]
    learners[1].phi=[p.clone() for p in bootstrap]
    equal_state(learners[0].state(),learners[1].state(),exclude=('phi',))
    restore_seconds=time.perf_counter()-started
    # Predict bypasses any procedure, so weights/state equality must imply this.
    with torch.no_grad():
        from e9_recursive_updater import student
        first=[student(l.params,l.attach(rows[0][0])) for l in learners]
        assert torch.equal(*first)
    results=[]
    for arm,learner in zip(spec['arms'],learners):
        stem=f"{record['replicate']}-{record['stream_seed']}-{record['family']}-{mode}-{arm}"
        path=out/f'{stem}.json'
        if path.exists():r=json.loads(path.read_text());assert r['identity']==identity;results.append(r);continue
        assert not (out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before rerunning'
        frozen=[p.clone() for p in learner.phi];before=learner.counts.copy();clean=[];noisy=[];normalized=[];failure=None;start=time.perf_counter()
        for t,(x,y,truth) in enumerate(rows):
            prediction=learner.predict(x);cl=float((prediction-truth).square().mean());noise=float((prediction-y).square().mean())
            if not math.isfinite(noise):failure=f'Nonfinite prediction at continuation batch {t}';break
            clean.append(cl);noisy.append(noise);normalized.append(cl/(float(truth.square().mean())+.01))
            try:learner.observe(x,y)
            except FloatingPointError as e:failure=str(e);break
        assert all(torch.equal(a,b) for a,b in zip(frozen,learner.phi))
        costs={k:v-before[k] for k,v in learner.counts.items()}
        if not failure:
            assert learner.step==spec['prefix_updates']+spec['continuation_updates']
            assert costs['updates']==spec['continuation_updates'] and costs['student_forward_examples']==len(rows)*96
        local=dict(continuation_costs=costs,global_costs=learner.costs(),seconds=time.perf_counter()-start,
            checkpoint=str(out/f'{stem}.pt'),failed=failure)
        if failure:local.update(completed_batches=len(clean))
        else:
            segs=segment_results(segments,clean,normalized,sc)
            full=[s for s in segs if not s.get('left_truncated',False)]
            local.update(clean_mse=float(np.mean(clean)),observed_mse=float(np.mean(noisy)),clean_batch_mse=clean,
                normalized_batch_mse=normalized,quarter_clean_mse=[float(np.mean(q)) for q in np.array_split(clean,4)],
                segments=segs,full_segments=len(full),recovered_full_segments=sum(s['recovery_batches'] is not None for s in full))
        torch.save(dict(identity=identity,learner_state=learner.state()),out/f'{stem}.pt')
        r=dict(identity=identity,replicate=record['replicate'],source_stream_seed=record['stream_seed'],family=record['family'],mode=mode,arm=arm,
            source_checkpoint=str(checkpoint),source_checkpoint_sha256=digest(checkpoint),paired_restore_seconds=restore_seconds,
            first_prediction_equal=True,result=local)
        path.write_text(json.dumps(r,indent=2)+'\n');results.append(r)
        print(json.dumps(dict(case=stem,mse=local.get('clean_mse'),recovered=local.get('recovered_full_segments'),failed=failure)),flush=True)
    # Data/RNG evolution must be weight-independent. Check when both executed now.
    if all(l.step==spec['prefix_updates']+spec['continuation_updates'] for l in learners):
        for name in ('context','replay_x','replay_y','rng'):assert torch.equal(learners[0].state()[name],learners[1].state()[name])
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e19.json').read_text());sc=json.loads((HERE/'protocol_e17.json').read_text())
    old=json.loads((a.runs/'e18/complete.json').read_text())
    for name,sha in old['identity'].items():assert digest(HERE/name)==sha
    identity={n:digest(HERE/n) for n in ('protocol_e19.json','e19_procedure_continuation.py')};identity['e18_identity']=old['identity']
    a.out.mkdir(parents=True,exist_ok=True)
    sources=[r for r in old['records'] if r['arm']=='online_self_applied'];assert len(sources)==8
    first=sources[0];payload=torch.load(first['result']['checkpoint'],map_location='cpu',weights_only=False)
    boot=torch.load(first['procedure_checkpoint'],map_location='cpu',weights_only=False)['phi']
    check=preflight(spec,sc,payload['learner_state'],boot)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=check),indent=2)+'\n');print('E19 matched-state preflight passed',flush=True)
    if not a.execute:return
    records=[];world_checks=[]
    for seed,new_seed in zip(spec['source_stream_seeds'],spec['new_world_seeds']):
        for family in spec['families']:
            for mode in spec['continuation_modes']:
                start=time.perf_counter()
                if mode=='same_world':rows,segments,prefix_check=continuation(sc,seed,family,spec['prefix_updates'],spec['continuation_updates'])
                else:
                    c=sc.copy();c['updates']=spec['continuation_updates'];rows,segments=make_stream(c,new_seed,family)
                    segments=[dict(s,left_truncated=False) for s in segments];prefix_check=None
                world_checks.append(dict(source_seed=seed,new_world_seed=new_seed if mode=='new_world' else None,family=family,mode=mode,
                    generation_seconds=time.perf_counter()-start,prefix_check=prefix_check))
                for rep in spec['controller_replicates']:
                    record=next(r for r in sources if r['replicate']==rep and r['stream_seed']==seed and r['family']==family)
                    checkpoint=Path(record['procedure_checkpoint']);assert digest(checkpoint)==record['procedure_checkpoint_sha256']
                    bootstrap=torch.load(checkpoint,map_location='cpu',weights_only=False)['phi']
                    records.extend(run_pair(spec,sc,record,bootstrap,rows,segments,mode,a.out,identity))
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,world_checks=world_checks,records=records),indent=2)+'\n');print('E19_COMPLETE',flush=True)

if __name__=='__main__':main()
