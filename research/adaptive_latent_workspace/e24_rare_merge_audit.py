from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
import e1_stream as e1
from e6_sharing import GuardedSharing,factory
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generate(spec,seed,updates=None):
    n=spec['updates'] if updates is None else updates
    wg=torch.Generator().manual_seed(seed);sg=torch.Generator().manual_seed(seed+100000);og=torch.Generator().manual_seed(seed+200000);pg=torch.Generator().manual_seed(seed+300000)
    w1=torch.randn(2,8,32,generator=wg)/math.sqrt(8);b=torch.randn(2,32,generator=wg)*.2;w2=torch.randn(2,32,4,generator=wg)/math.sqrt(32)
    means=torch.randn(4,8,generator=wg)*3;means[0]=0;means[:,0]=0
    def target(x,rare):
        y=torch.empty(len(x),4)
        for k in (0,1):
            mask=rare if k else ~rare;y[mask]=torch.tanh(x[mask]@w1[k]+b[k])@w2[k]
        return y
    def inputs(size,g,prob,context):
        rare=torch.rand(size,generator=g)<prob;x=torch.randn(size,8,generator=g)
        x[~rare]+=means[context];x[:,0]=torch.where(rare,1.,-1.)*(1+.5*x[:,0].abs())
        return x,rare
    rows=[];rare_counts=[];ctx=0
    for t in range(n):
        if t>=spec['acquisition_updates'] and (t-spec['acquisition_updates'])%spec['common_shift_batches']==0:
            ctx=(ctx+1+int(torch.randint(3,(),generator=sg)))%4
        prob=spec['rare_probability_acquisition'] if t<spec['acquisition_updates'] else spec['rare_probability_later']
        x,rare=inputs(spec['batch_size'],og,prob,ctx);clean=target(x,rare);y=clean+.03*torch.randn(len(x),4,generator=og)
        rows.append((x,y,clean));rare_counts.append(int(rare.sum()))
    probes={}
    for name,prob in [('common',0.),('rare',1.)]:
        x,rare=inputs(spec['probe_examples_per_region'],pg,prob,0);probes[name]=(x,target(x,rare))
    return rows,probes,rare_counts

@torch.no_grad()
def model_probe(model,probes):return {name:float((model(x)-y).square().mean()) for name,(x,y) in probes.items()}

@torch.no_grad()
def endpoint(learner,probes):
    operational={name:float((learner.predict(x,count=False)-y).square().mean()) for name,(x,y) in probes.items()}
    # Context learner has a context-augmented interface, not raw-input modules.
    module_scores=[model_probe(m,probes) for m in learner.models] if isinstance(learner,GuardedSharing) else []
    best={name:min(v[name] for v in module_scores) for name in probes} if module_scores else None
    return dict(operational=operational,best_persistent_module_diagnostic=best,module_scores=module_scores)

def audit_step(learner,x,y,probes,spec,enabled=True,verify_old=False):
    if not isinstance(learner,GuardedSharing):learner.observe(x,y);return [],0,0.
    refs=list(learner.models);previous=len(learner.events)
    before=[copy.deepcopy(m.state_dict()) for m in refs] if verify_old else None
    learner.observe(x,y)
    checks=[e for e in learner.events[previous:] if e['kind']=='anchor_check']
    if not enabled or not checks:return [],0,0.
    start=time.perf_counter();out=[];examples=0;state=torch.get_rng_state().clone()
    for event in checks:
        source=learner.source;old=refs[source]
        if event['merged']:candidate=learner.models[source]
        elif len(learner.models)>len(refs):candidate=learner.models[-1]
        else:candidate=learner.fast_model
        assert candidate is not None and candidate is not old
        if verify_old:assert all(torch.equal(v,old.state_dict()[k]) for k,v in before[source].items())
        old_score=model_probe(old,probes);new_score=model_probe(candidate,probes);examples+=2*sum(len(x) for x,y in probes.values())
        competent=old_score['rare']<=spec['competence_mse']
        material=competent and new_score['rare']>max(spec['competence_mse'],spec['material_loss_ratio']*old_score['rare'])
        out.append(dict(event=copy.deepcopy(event),source=source,old_probe=old_score,candidate_probe=new_score,
            source_rare_competent=competent,material_rare_loss=material,accepted_material_rare_loss=material and event['merged'],
            audited_outgoing_parameter_bytes=sum(p.numel()*p.element_size() for p in old.parameters())))
    assert torch.equal(state,torch.get_rng_state())
    return out,examples,time.perf_counter()-start

def preflight(spec,cfg):
    a,pa,ca=generate(spec,109,600);b,pb,cb=generate(spec,109,700)
    assert all(torch.equal(u,v) for r,s in zip(a,b) for u,v in zip(r,s)) and ca==cb[:600]
    assert all(torch.equal(pa[k][0],pb[k][0]) for k in pa)
    c=copy.deepcopy(cfg);c.update(module_cap=1,merge_absolute_tolerance=100.)
    g=torch.Generator().manual_seed(110);rows=[(torch.randn(32,8,generator=g),torch.ones(32,4)*(0 if i<100 else 3)) for i in range(180)]
    outcomes=[];allchecks=[]
    for enabled in (False,True):
        torch.manual_seed(111);l=GuardedSharing(c,torch.device('cpu'));predictions=[]
        for x,y in rows:
            predictions.append(l.predict(x).clone());checked,_,_=audit_step(l,x,y,pa,spec,enabled,verify_old=enabled);allchecks+=checked
        outcomes.append((l,predictions,torch.get_rng_state().clone()))
    la,xa,ra=outcomes[0];lb,xb,rb=outcomes[1]
    assert all(torch.equal(x,y) for x,y in zip(xa,xb)) and torch.equal(ra,rb) and la.costs()==lb.costs()
    assert all(torch.equal(x,y) for ma,mb in zip(la.models,lb.models) for x,y in zip(ma.parameters(),mb.parameters()))
    assert all(torch.equal(a.x,b.x) and torch.equal(a.y,b.y) for a,b in zip(la.anchors,lb.anchors))
    assert la.merges>0 and any(r['event']['merged'] for r in allchecks)
    import io
    buf=io.BytesIO();torch.save(lb,buf);buf.seek(0);restored=torch.load(buf,weights_only=False)
    assert endpoint(restored,pa)==endpoint(lb,pa)
    return dict(prefix_exact=True,audit_does_not_change_predictions_weights_anchors_counters_or_rng=True,
        outgoing_model_frozen_on_checks=True,accepted_checks_exercised=sum(r['event']['merged'] for r in allchecks),checkpoint_prediction_restore_exact=True)

def run(spec,cfg,seed,arm,rows,probes,rare_counts,out,identity):
    torch.manual_seed(seed+400000);learner=factory(arm,cfg,torch.device('cpu'));errors=[];audits=[];endpoints=[];af=0;asec=0.;failure=None;start=time.perf_counter()
    for t,(x,y,truth) in enumerate(rows):
        pred=learner.predict(x);error=float((pred-truth).square().mean())
        if not math.isfinite(error):failure=f'Nonfinite prediction at {t}';break
        errors.append(error)
        try:checks,count,seconds=audit_step(learner,x,y,probes,spec)
        except RuntimeError as e:failure=str(e);break
        for c in checks:c['observation_batch']=t+1
        audits.extend(checks);af+=count;asec+=seconds
        if (t+1)%spec['probe_every_batches']==0:
            ts=time.perf_counter();state=torch.get_rng_state().clone();e=endpoint(learner,probes);assert torch.equal(state,torch.get_rng_state())
            e['batch']=t+1;endpoints.append(e);af+=sum(len(x) for x,y in probes.values())*(1+(len(learner.models) if isinstance(learner,GuardedSharing) else 0));asec+=time.perf_counter()-ts
    elapsed=time.perf_counter()-start;stem=f'{seed}-{arm}'
    r=dict(seed=seed,arm=arm,identity=identity,failed=failure,completed_batches=len(errors),clean_batch_mse=errors,
        mean_clean_mse=float(np.mean(errors)) if errors else None,learner_costs=learner.costs(),loop_seconds_including_audit=elapsed,
        audit_seconds=asec,loop_seconds_excluding_audit_kernels=elapsed-asec,audit_forward_examples=af,probe_tensor_bytes=sum(t.numel()*t.element_size() for pair in probes.values() for t in pair),
        audits=audits,endpoints=endpoints,rare_observations=sum(rare_counts[:len(errors)]),
        rare_acquisition_observations=sum(rare_counts[:spec['acquisition_updates']]),capacity_hits=getattr(learner,'capacity_hits',0),
        merges=getattr(learner,'merges',0),events=getattr(learner,'events',[]))
    if not failure:
        assert len(errors)==spec['updates'];ts=time.perf_counter();path=out/f'{stem}.pt';torch.save(dict(identity=identity,learner=learner),path)
        r.update(checkpoint=str(path),checkpoint_bytes=path.stat().st_size,checkpoint_seconds=time.perf_counter()-ts)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e24.json').read_text());cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(json.loads((HERE/'protocol_e6.json').read_text()))
    cfg.update(module_cap=spec['module_cap'],anchor_capacity=spec['anchor_capacity']);a.out.mkdir(parents=True,exist_ok=True)
    names=('protocol_e24.json','e24_rare_merge_audit.py','protocol_e1.json','protocol_e2.json','protocol_e6.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py')
    identity={n:digest(HERE/n) for n in names};start=time.perf_counter();check=preflight(spec,cfg)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=check,seconds=time.perf_counter()-start),indent=2)+'\n');print('E24 rare merge audit preflight passed',flush=True)
    if not a.execute:return
    records=[];worlds=[]
    for seed in spec['seeds']:
        start=time.perf_counter();rows,probes,rare_counts=generate(spec,seed);worlds.append(dict(seed=seed,generation_seconds=time.perf_counter()-start,rare_counts=rare_counts))
        for arm in spec['arms']:
            stem=f'{seed}-{arm}';path=a.out/f'{stem}.json'
            if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
            else:
                assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before any restart'
                r=run(spec,cfg,seed,arm,rows,probes,rare_counts,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r);print(json.dumps(dict(seed=seed,arm=arm,mse=r['mean_clean_mse'],merges=r['merges'],checks=len(r['audits']),accepted_rare_losses=sum(c['accepted_material_rare_loss'] for c in r['audits']),failed=r['failed'])),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,worlds=worlds,records=records),indent=2)+'\n');print('E24_COMPLETE',flush=True)
if __name__=='__main__':main()
