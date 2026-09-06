from pathlib import Path
import argparse, copy, hashlib, json, math, time
import numpy as np
import torch
from e9_recursive_updater import student, update, leaves, init_controller
from e17_persistent_student import PersistentStudent, make_stream

HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clone_tensors(xs):return [x.detach().clone() for x in xs]

def trace_batch(learner,x,y):
    attached=learner.attach(x)
    if learner.size:
        g=torch.Generator();g.set_state(learner.rng.get_state())
        ids=torch.randint(learner.size,(learner.cfg['replay_batch_size'],),generator=g)
        return torch.cat([attached,learner.replay_x[ids]]),torch.cat([y,learner.replay_y[ids]])
    return attached.clone(),y.clone()

def snapshot(learner):
    return dict(params=clone_tensors(learner.params),m=clone_tensors(learner.m),v=clone_tensors(learner.v),step=learner.step,lr=learner.lr)

def unroll(state,traces,query,old,phi,base):
    dtype=phi[0].dtype
    params=leaves([p.to(dtype) for p in state['params']]);m=[p.to(dtype) for p in state['m']];v=[p.to(dtype) for p in state['v']]
    for i,(x,y) in enumerate(traces):
        loss=(student(params,x.to(dtype))-y.to(dtype)).square().mean()
        grads=torch.autograd.grad(loss,params,create_graph=True)
        params,m,v=update(params,grads,m,v,state['step']+i+1,loss,phi,base,state['lr'])
    loss=.5*(student(params,query[0].to(dtype))-query[1].to(dtype)).square().mean()
    loss=loss+.5*(student(params,old[0].to(dtype))-old[1].to(dtype)).square().mean()
    return loss,params

def sampled_old(learner,n,g):
    ids=torch.randint(learner.size,(n,),generator=g)
    return learner.replay_x[ids].clone(),learner.replay_y[ids].clone()

def tracked_bytes(*objects):
    storages={}
    def visit(o):
        if isinstance(o,torch.Tensor):
            s=o.untyped_storage();storages[(s.data_ptr(),s.nbytes())]=s.nbytes()
        elif isinstance(o,dict):
            for v in o.values():visit(v)
        elif isinstance(o,(tuple,list)):
            for v in o:visit(v)
    for o in objects:visit(o)
    return sum(storages.values())

def preflight(cfg,stream_cfg,base,out,runs):
    c=copy.deepcopy(stream_cfg);c.update(updates=12,segment_min_batches=4,segment_max_batches=4)
    rows,_=make_stream(c,88,'conflicting_functions');phi=init_controller(base,89)
    learner=PersistentStudent(c,base,90,phi,.003);start=snapshot(learner);traces=[]
    for x,y,_ in rows[:3]:traces.append(trace_batch(learner,x,y));learner.predict(x);learner.observe(x,y)
    query=(learner.attach(rows[3][0]),rows[3][1]);old=sampled_old(learner,16,torch.Generator().manual_seed(91))
    loss,params=unroll(start,traces,query,old,leaves(phi),base)
    assert all(torch.equal(a,b) for a,b in zip(params,learner.params))
    p64=leaves([p.double() for p in phi]);analytic=float(torch.autograd.grad(unroll(start,traces,query,old,p64,base)[0],p64)[3])
    eps=1e-5;plus=leaves(p64);minus=leaves(p64)
    with torch.no_grad():plus[3].add_(eps);minus[3].sub_(eps)
    numeric=float((unroll(start,traces,query,old,plus,base)[0]-unroll(start,traces,query,old,minus,base)[0]).detach())/(2*eps)
    assert abs(analytic-numeric)<1e-5+abs(numeric)*1e-3,(analytic,numeric)
    grads=[torch.ones_like(p)*.1 for p in phi]
    zero=[torch.zeros_like(p) for p in phi]
    conventional=update(phi,grads,zero,zero,1,loss.detach(),None,base,.003)[0]
    self_applied=update(phi,grads,zero,zero,1,loss.detach(),phi,base,.003)[0]
    assert all(torch.equal(a,b) for a,b in zip(conventional,self_applied))
    copy_learner=PersistentStudent.restore(copy.deepcopy(learner.state()))
    x,y,_=rows[3];incumbent_prediction=learner.predict(x);candidate_prediction=copy_learner.predict(x)
    assert torch.equal(incumbent_prediction,candidate_prediction)
    learner.observe(x,y);copy_learner.observe(x,y)
    assert all(torch.equal(a,b) for a,b in zip(learner.params,copy_learner.params))
    import tempfile
    smoke_cfg=cfg.copy();smoke_cfg.update(rounds=2,ordinary_batches=12,trial_batches=4,retrospective_updates=3,old_query_examples=16)
    smoke_stream=copy.deepcopy(stream_cfg);smoke_stream['updates']=32
    with tempfile.TemporaryDirectory(dir=out,prefix='preflight-') as td:
        aa=run_case(smoke_cfg,smoke_stream,base,93,'conflicting_functions','online_meta_adam',phi,Path(td),'meta',{})
        bb=run_case(smoke_cfg,smoke_stream,base,93,'conflicting_functions','online_self_applied',phi,Path(td),'self',{})
        trained=torch.load(runs/'e9/12101-bootstrap.pt',map_location='cpu',weights_only=False)['phi']
        cc=run_case(smoke_cfg,smoke_stream,base,94,'input_shift','online_self_applied',trained,Path(td),'trained',{})
    assert all('failed' not in r for r in (aa,bb,cc))
    for key in ('incumbent_trial_mse','candidate_trial_mse','accepted','proposed_phi'):
        assert aa['rounds'][0][key]==bb['rounds'][0][key]
    assert aa['unique_stream_examples']==32*stream_cfg['batch_size']
    return dict(retrospective_parameters_exact=True,meta_gradient_analytic=analytic,meta_gradient_finite_difference=numeric,
        zero_controller_proposal_exact=True,branch_copy_exact=True,retrospective_updates=3,
        full_loop_first_zero_proposal_equal=True,trained_controller_replay_exact=True,
        smoke_counts=[r['costs'] for r in (aa,bb,cc)],smoke_seconds=sum(r['seconds'] for r in (aa,bb,cc)),
        note='Synthetic preflight uses three replayed updates, double-precision finite differences, and one paired branch update; not a scored replicate.')

def segment_results(segments,clean,normalized,cfg):
    results=[]
    for s in segments:
        lo,hi=s['start'],s['stop'];run=0;recovery=None
        for j,value in enumerate(normalized[lo:hi]):
            run=run+1 if value<=cfg['recovery_normalized_mse'] else 0
            if run>=cfg['recovery_consecutive_batches']:recovery=j+1;break
        results.append(dict(**s,clean_mse=float(np.mean(clean[lo:hi])),first_four_mse=float(np.mean(clean[lo:min(hi,lo+4)])),
            last_four_mse=float(np.mean(clean[max(lo,hi-4):hi])),recovery_batches=recovery,censored_at_batches=hi-lo if recovery is None else None))
    return results

def run_case(cfg,sc,base,seed,family,arm,phi,out,stem,identity):
    started=time.perf_counter();rows,segments=make_stream(sc,seed,family)
    learner=PersistentStudent(sc,base,seed,None if arm=='adam_01' else phi,.01 if arm=='adam_01' else .003)
    online=arm.startswith('online_');g=torch.Generator().manual_seed(seed+300000)
    meta_m=[torch.zeros_like(p) for p in phi];meta_v=[torch.zeros_like(p) for p in phi];accepted=0;previous_phi=None
    totals={k:0 for k in learner.counts};totals.update(meta_gradient_calls=0,outer_controller_forward_calls=0,outer_proposals=0,retrospective_updates=0,trace_preview_examples=0)
    observed=[];clean=[];normalized=[];rounds=[];peak=tracked_bytes(learner.state(),meta_m,meta_v);position=0
    log_path=out/f'{stem}-rounds.jsonl';assert not log_path.exists(),'Partial round log exists: inspect before rerunning'
    def add_delta(before,after):
        for k in learner.counts:totals[k]+=after[k]-before[k]
    def consume(model,x,y,truth,operational):
        before=model.counts.copy();prediction=model.predict(x)
        loss=float((prediction-y).square().mean());cl=float((prediction-truth).square().mean())
        if not math.isfinite(loss):raise FloatingPointError('nonfinite branch prediction')
        if operational:observed.append(loss);clean.append(cl);normalized.append(cl/(float(truth.square().mean())+.01))
        model.observe(x,y);add_delta(before,model.counts);return loss
    try:
        for round_id in range(cfg['rounds']):
            round_start=position;trace=[];saved=None;query=None;old_query=None;expected=None
            for j in range(cfg['ordinary_batches']):
                x,y,truth=rows[position]
                if online:
                    if j==cfg['ordinary_batches']-cfg['retrospective_updates']-1:saved=snapshot(learner)
                    if cfg['ordinary_batches']-cfg['retrospective_updates']-1<=j<cfg['ordinary_batches']-1:
                        trace.append(trace_batch(learner,x,y));totals['trace_preview_examples']+=len(trace[-1][0])
                    if j==cfg['ordinary_batches']-1:
                        query=(learner.attach(x).clone(),y.clone());old_query=sampled_old(learner,cfg['old_query_examples'],g)
                        expected=clone_tensors(learner.params)
                consume(learner,x,y,truth,True);position+=1
            proposal=None;candidate=None;candidate_phi=None;cm=cv=None;meta_value=None;pretrial_old=None
            if online:
                assert len(trace)==cfg['retrospective_updates']
                current_phi=leaves(learner.phi);meta_loss,restored=unroll(saved,trace,query,old_query,current_phi,base)
                assert all(torch.equal(a,b) for a,b in zip(restored,expected)),'Retrospective replay does not match actual pre-query student'
                grads=torch.autograd.grad(meta_loss,current_phi)
                norm=torch.stack([p.square().sum() for p in grads]).sum().sqrt()
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite meta-gradient')
                grads=[p.detach()*min(1.,cfg['meta_gradient_norm_clip']/(float(norm)+1e-12)) for p in grads]
                meta_value=float(meta_loss.detach())
                candidate_phi,cm,cv=update(current_phi,grads,meta_m,meta_v,accepted+1,meta_loss.detach(),
                    clone_tensors(current_phi) if arm=='online_self_applied' else None,base,cfg['meta_learning_rate'])
                candidate_phi=clone_tensors(candidate_phi)
                candidate=PersistentStudent.restore(copy.deepcopy(learner.state()));candidate.phi=clone_tensors(candidate_phi)
                pretrial_old=sampled_old(learner,cfg['old_query_examples'],g)
                totals['student_forward_examples']+=sum(len(x) for x,y in trace)+len(query[0])+len(old_query[0])
                totals['student_gradient_batches']+=len(trace);totals['updates']+=len(trace);totals['retrospective_updates']+=len(trace)
                totals['controller_forward_calls']+=len(trace)*len(learner.params);totals['meta_gradient_calls']+=1;totals['outer_proposals']+=1
                totals['outer_controller_forward_calls']+=len(current_phi) if arm=='online_self_applied' else 0
                peak=max(peak,tracked_bytes(learner.state(),candidate.state(),saved,trace,query,old_query,pretrial_old,
                    current_phi,candidate_phi,grads,meta_m,meta_v,cm,cv,expected,restored,previous_phi))
                del meta_loss,restored,grads,current_phi
            before_errors=[];after_errors=[]
            for j in range(cfg['trial_batches']):
                x,y,truth=rows[position]
                before_errors.append(consume(learner,x,y,truth,True))
                if candidate is not None:after_errors.append(consume(candidate,x,y,truth,False))
                position+=1
            row=dict(round=round_id,start_batch=round_start,stop_batch=position,operational_clean_mse=float(np.mean(clean[round_start:position])),
                meta_loss=meta_value,accepted=None,incumbent_trial_mse=float(np.mean(before_errors)),candidate_trial_mse=None)
            assert row['incumbent_trial_mse']==float(np.mean(observed[-cfg['trial_batches']:]))
            if candidate is not None:
                with torch.no_grad():
                    old_before=float((student(learner.params,pretrial_old[0])-pretrial_old[1]).square().mean())
                    old_after=float((student(candidate.params,pretrial_old[0])-pretrial_old[1]).square().mean())
                totals['student_forward_examples']+=2*len(pretrial_old[0])
                after=float(np.mean(after_errors));before=float(np.mean(before_errors))
                admit=math.isfinite(after) and math.isfinite(old_after) and after<before*(1-cfg['accept_improvement_fraction']) and old_after<=old_before*(1+cfg['retention_tolerance_fraction'])
                row.update(accepted=admit,candidate_trial_mse=after,old_sample_before=old_before,old_sample_after=old_after,
                    proposed_phi=[p.tolist() for p in candidate_phi])
                if admit:
                    previous_phi=clone_tensors(learner.phi);learner=candidate;meta_m=clone_tensors(cm);meta_v=clone_tensors(cv);accepted+=1
            assert len(clean)==position and learner.step==position
            row.update(cumulative_counts=totals.copy(),elapsed_seconds=time.perf_counter()-started);rounds.append(row)
            with log_path.open('a') as f:f.write(json.dumps(row)+'\n')
            if (round_id+1)%16==0:print(json.dumps(dict(case=stem,round=round_id+1,mse=float(np.mean(clean)),accepted=accepted)),flush=True)
        result=dict(clean_mse=float(np.mean(clean)),observed_mse=float(np.mean(observed)),
            quarter_clean_mse=[float(np.mean(q)) for q in np.array_split(clean,4)],clean_batch_mse=clean,normalized_batch_mse=normalized,
            segments=segment_results(segments,clean,normalized,sc),rounds=rounds,accepted=accepted,
            costs=totals,unique_stream_examples=position*sc['batch_size'],tracked_peak_tensor_bytes=peak,
            stored_tensor_bytes=tracked_bytes(learner.state(),meta_m,meta_v,previous_phi,g.get_state()),
            seconds=time.perf_counter()-started,checkpoint=str(out/f'{stem}.pt'))
        torch.save(dict(identity=identity,learner_state=learner.state(),meta_m=meta_m,meta_v=meta_v,previous_phi=previous_phi,
            meta_rng=g.get_state(),accepted=accepted,counts=totals),out/f'{stem}.pt')
        result['checkpoint_file_bytes']=(out/f'{stem}.pt').stat().st_size;result['trial_log_file_bytes']=log_path.stat().st_size
        return result
    except FloatingPointError as e:
        return dict(failed=str(e),batch=position,costs=totals,unique_stream_examples=position*sc['batch_size'],
            seconds=time.perf_counter()-started,completed_rounds=rounds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);cfg=json.loads((HERE/'protocol_e18.json').read_text());sc=json.loads((HERE/'protocol_e17.json').read_text())
    sc['updates']=cfg['rounds']*(cfg['ordinary_batches']+cfg['trial_batches']);base=json.loads((HERE/'protocol_e9.json').read_text())
    identity={n:digest(HERE/n) for n in ('protocol_e18.json','e18_online_procedure.py','protocol_e17.json','e17_persistent_student.py','protocol_e9.json','e9_recursive_updater.py')}
    a.out.mkdir(parents=True,exist_ok=True);checks=preflight(cfg,sc,base,a.out,a.runs)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks),indent=2)+'\n');print('E18 online procedure preflight passed',flush=True)
    if not a.execute:return
    old=json.loads((a.runs/'e9/complete.json').read_text())
    for name,value in old['identity'].items():
        if name.endswith(('.py','.json')):assert digest(HERE/name)==value
    records=[]
    for rep in cfg['controller_replicates']:
        checkpoint=a.runs/'e9'/f'{rep}-bootstrap.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['identity']==old['identity'];phi=payload['phi']
        for seed in cfg['stream_seeds']:
            for family in cfg['families']:
                for arm in cfg['arms']:
                    stem=f'{rep}-{seed}-{family}-{arm}';path=a.out/f'{stem}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
                    reused=None
                    if arm=='adam_01' and rep!=cfg['controller_replicates'][0]:
                        reused=next(r for r in records if r['replicate']==cfg['controller_replicates'][0] and r['stream_seed']==seed and r['family']==family and r['arm']==arm);result=reused['result']
                    else:result=run_case(cfg,sc,base,seed,family,arm,phi,a.out,stem,identity)
                    r=dict(identity=identity,replicate=rep,stream_seed=seed,family=family,arm=arm,
                        procedure_checkpoint=str(checkpoint) if arm!='adam_01' else None,procedure_checkpoint_sha256=digest(checkpoint) if arm!='adam_01' else None,
                        inherited_from_rep=reused['replicate'] if reused else None,new_work=not bool(reused),result=result)
                    path.write_text(json.dumps(r,indent=2)+'\n');records.append(r)
                    print(json.dumps(dict(case=stem,mse=result.get('clean_mse'),accepted=result.get('accepted'),failed=result.get('failed'))),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=cfg,identity=identity,inherited_e9=old['identity'],records=records),indent=2)+'\n');print('E18_COMPLETE',flush=True)

if __name__=='__main__':main()
