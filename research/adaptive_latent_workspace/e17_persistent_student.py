from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
import torch
from e9_recursive_updater import student, update, leaves, init_controller

HERE = Path(__file__).resolve().parent

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_stream(cfg, seed, family):
    g = torch.Generator().manual_seed(seed)
    c = cfg['student']; k, d, h, o = cfg['contexts'], c['input_dim'], c['teacher_hidden'], c['output_dim']
    w1 = torch.randn(k,d,h,generator=g)/math.sqrt(d)
    b = torch.randn(k,h,generator=g)*.2
    w2 = torch.randn(k,h,o,generator=g)/math.sqrt(h)
    means = torch.randn(k,d,generator=g)*1.5; means[0]=0
    current = int(torch.randint(k,(),generator=g)); rows=[]; segments=[]; t=0; seen=set()
    while t < cfg['updates']:
        length=min(int(torch.randint(cfg['segment_min_batches'],cfg['segment_max_batches']+1,(),generator=g)),cfg['updates']-t)
        segments.append(dict(start=t,stop=t+length,context=current,recurring=current in seen)); seen.add(current)
        for _ in range(length):
            x=torch.randn(cfg['batch_size'],d,generator=g)
            if family=='input_shift': x=x+means[current]
            teacher=current if family=='conflicting_functions' else 0
            clean=torch.tanh(x@w1[teacher]+b[teacher])@w2[teacher]
            y=clean+.03*torch.randn(cfg['batch_size'],o,generator=g)
            rows.append((x,y,clean))
        t+=length
        current=(current+1+int(torch.randint(k-1,(),generator=g)))%k
    return rows,segments

class PersistentStudent:
    def __init__(self,cfg,base,seed,phi,lr):
        self.cfg=cfg;self.base=base;self.lr=lr;self.phi=None if phi is None else [p.detach().clone() for p in phi]
        c=cfg['student'];d=c['input_dim']+c['context_dim'];h=c['hidden_dim'];o=c['output_dim']
        g=torch.Generator().manual_seed(seed+100000)
        self.params=leaves([torch.randn(h,d,generator=g)/math.sqrt(d),torch.zeros(h),
            torch.randn(h,h,generator=g)/math.sqrt(h),torch.zeros(h),torch.randn(o,h,generator=g)/math.sqrt(h),torch.zeros(o)])
        self.m=[torch.zeros_like(p) for p in self.params];self.v=[torch.zeros_like(p) for p in self.params]
        self.context=torch.zeros(c['input_dim']+1,o)
        self.replay_x=torch.zeros(cfg['replay_capacity'],d);self.replay_y=torch.zeros(cfg['replay_capacity'],o)
        self.size=0;self.seen=0;self.step=0;self.rng=torch.Generator().manual_seed(seed+200000)
        self.counts=dict(student_forward_examples=0,student_gradient_batches=0,controller_forward_calls=0,
            observed_examples=0,replayed_examples=0,updates=0,reservoir_draws=0)
    def attach(self,x): return torch.cat([x,self.context.reshape(1,-1).expand(len(x),-1)],dim=1)
    @torch.no_grad()
    def predict(self,x):
        self.counts['student_forward_examples']+=len(x)
        return student(self.params,self.attach(x))
    def observe(self,x,y):
        attached=self.attach(x)
        if self.size:
            ids=torch.randint(self.size,(self.cfg['replay_batch_size'],),generator=self.rng)
            tx=torch.cat([attached,self.replay_x[ids]]);ty=torch.cat([y,self.replay_y[ids]])
            self.counts['replayed_examples']+=len(ids)
        else:tx,ty=attached,y
        loss=(student(self.params,tx)-ty).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite persistent student loss')
        grads=torch.autograd.grad(loss,self.params)
        self.step+=1
        self.params,self.m,self.v=update(self.params,grads,self.m,self.v,self.step,loss,self.phi,self.base,self.lr)
        self.params=leaves(self.params);self.m=[p.detach() for p in self.m];self.v=[p.detach() for p in self.v]
        self.counts['updates']+=1;self.counts['observed_examples']+=len(x)
        self.counts['student_forward_examples']+=len(tx);self.counts['student_gradient_batches']+=1
        self.counts['controller_forward_calls']+=len(self.params) if self.phi is not None else 0
        for xi,yi in zip(attached,y):
            if self.seen<len(self.replay_x): slot=self.seen
            else:slot=int(torch.randint(self.seen+1,(),generator=self.rng));self.counts['reservoir_draws']+=1
            if slot<len(self.replay_x):self.replay_x[slot]=xi;self.replay_y[slot]=yi
            self.seen+=1;self.size=min(self.seen,len(self.replay_x))
        extended=torch.cat([x,torch.ones(len(x),1)],dim=1)
        self.context.mul_(self.cfg['context_ema']).add_((extended.T@y)/len(x),alpha=1-self.cfg['context_ema'])
    def costs(self):
        tensors=self.params+self.m+self.v+[self.context,self.replay_x,self.replay_y,self.rng.get_state()]+(self.phi or [])
        return dict(**self.counts,student_parameters=sum(p.numel() for p in self.params),
            controller_parameters=sum(p.numel() for p in (self.phi or [])),stored_tensor_bytes=sum(p.numel()*p.element_size() for p in tensors),
            replay_size=self.size,optimizer_step=self.step)
    def state(self):
        return dict(params=[p.detach() for p in self.params],m=self.m,v=self.v,phi=self.phi,
            context=self.context,replay_x=self.replay_x,replay_y=self.replay_y,rng=self.rng.get_state(),
            size=self.size,seen=self.seen,step=self.step,counts=self.counts.copy(),cfg=self.cfg,base=self.base,lr=self.lr)
    @classmethod
    def restore(cls,state):
        result=cls(state['cfg'],state['base'],0,state['phi'],state['lr'])
        for k in ('m','v','context','replay_x','replay_y','size','seen','step','counts'):setattr(result,k,state[k])
        result.params=leaves(state['params']);result.rng.set_state(state['rng']);return result

def preflight(cfg,base):
    c=cfg.copy();c.update(updates=5,segment_min_batches=2,segment_max_batches=2,replay_capacity=48)
    rows,segments=make_stream(c,77,'conflicting_functions')
    z=init_controller(base,78);a=PersistentStudent(c,base,79,None,.003);b=PersistentStudent(c,base,79,z,.003)
    before=[p.clone() for p in b.phi];initial=[p.detach().clone() for p in a.params]
    for t,(x,y,_) in enumerate(rows):
        pa=a.predict(x);pb=b.predict(x);assert torch.equal(pa,pb)
        assert torch.equal(student(a.params,a.attach(x)),pa)
        a.observe(x,y);b.observe(x,y)
        assert all(torch.equal(u,v) for u,v in zip(a.params,b.params))
        assert a.step==t+1 and a.size<=48
    repeat=PersistentStudent(c,base,79,None,.003)
    for x,y,_ in rows:repeat.predict(x);repeat.observe(x,y)
    assert all(torch.equal(u,v) for u,v in zip(a.params,repeat.params))
    assert all(torch.equal(u,v) for u,v in zip(before,b.phi))
    assert any(not torch.equal(u,v) for u,v in zip(initial,a.params))
    assert a.counts['student_forward_examples']==5*32*2+4*32 and a.counts['updates']==5
    import copy
    resumed=PersistentStudent.restore(copy.deepcopy(a.state()));x,y,_=rows[0]
    assert torch.equal(a.predict(x),resumed.predict(x));a.observe(x,y);resumed.observe(x,y)
    assert all(torch.equal(u,v) for u,v in zip(a.params,resumed.params)) and a.costs()==resumed.costs()
    return dict(zero_controller_exact=True,deterministic_repeat=True,procedure_unchanged=True,
        exact_state_resume=True,boundaries=segments,costs=a.costs(),persistent_step=a.step)

def execute_case(cfg,base,rows,segments,seed,phi,lr,out,stem,identity):
    learner=PersistentStudent(cfg,base,seed,phi,lr);results=[];losses=[];clean_losses=[];normalized=[];started=time.perf_counter()
    segment_index=0
    for t,(x,y,clean) in enumerate(rows):
        prediction=learner.predict(x);loss=float((prediction-y).square().mean());cl=float((prediction-clean).square().mean())
        if not math.isfinite(loss):return dict(failed=f'nonfinite prediction at {t}',seconds=time.perf_counter()-started,costs=learner.costs(),completed_segments=results)
        losses.append(loss);clean_losses.append(cl);normalized.append(cl/(float(clean.square().mean())+.01))
        try:learner.observe(x,y)
        except FloatingPointError as e:return dict(failed=str(e),seconds=time.perf_counter()-started,costs=learner.costs(),completed_segments=results)
        if t+1==segments[segment_index]['stop']:
            s=segments[segment_index];start,stop=s['start'],s['stop'];norm=normalized[start:stop];run=0;recovery=None
            for j,value in enumerate(norm):
                run=run+1 if value<=cfg['recovery_normalized_mse'] else 0
                if run>=cfg['recovery_consecutive_batches']:recovery=j+1;break
            results.append(dict(**s,observed_mse=float(np.mean(losses[start:stop])),clean_mse=float(np.mean(clean_losses[start:stop])),
                first_four_mse=float(np.mean(clean_losses[start:min(start+4,stop)])),last_four_mse=float(np.mean(clean_losses[max(start,stop-4):stop])),
                recovery_batches=recovery,censored_at_batches=stop-start if recovery is None else None,cumulative_costs=learner.costs()))
            segment_index+=1
    torch.save(dict(identity=identity,learner_state=learner.state()),out/f'{stem}.pt')
    return dict(segments=results,observed_mse=float(np.mean(losses)),clean_mse=float(np.mean(clean_losses)),
        clean_batch_mse=clean_losses,normalized_batch_mse=normalized,seconds=time.perf_counter()-started,costs=learner.costs(),
        recovered_segments=sum(s['recovery_batches'] is not None for s in results),total_segments=len(results),
        checkpoint=str(out/f'{stem}.pt'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);cfg=json.loads((HERE/'protocol_e17.json').read_text());base=json.loads((HERE/'protocol_e9.json').read_text())
    identity={n:digest(HERE/n) for n in ('protocol_e17.json','e17_persistent_student.py','protocol_e9.json','e9_recursive_updater.py')}
    a.out.mkdir(parents=True,exist_ok=True);checks=preflight(cfg,base)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks),indent=2)+'\n');print('E17 persistent-student preflight passed',flush=True)
    if not a.execute:return
    old=json.loads((a.runs/'e9/complete.json').read_text());control=json.loads((a.runs/'e11/complete.json').read_text())
    for source in (old,control):
        for name,value in source['identity'].items():
            if name.endswith(('.py','.json')):assert digest(HERE/name)==value
    records=[]
    for rep in cfg['controller_replicates']:
        for seed in cfg['stream_seeds']:
            for family in cfg['families']:
                rows,segments=make_stream(cfg,seed,family)
                for arm in cfg['arms']:
                    stem=f'{rep}-{seed}-{family}-{arm}';path=a.out/f'{stem}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
                    checkpoint=None;phi=None;lr=.003
                    if arm.startswith('adam_'):lr={'adam_001':.001,'adam_003':.003,'adam_01':.01}[arm]
                    else:
                        suffix={'frozen_bootstrap':'bootstrap','self_applied':'self-applied','fixed_meta_adam':'fixed-meta-adam'}[arm]
                        checkpoint=a.runs/('e11' if arm=='fixed_meta_adam' else 'e9')/f'{rep}-{suffix}.pt'
                        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                        assert payload['identity']==(control if arm=='fixed_meta_adam' else old)['identity'];phi=payload['phi']
                    reused=None
                    if arm.startswith('adam_') and rep!=cfg['controller_replicates'][0]:
                        reused=next(r for r in records if r['replicate']==cfg['controller_replicates'][0] and r['stream_seed']==seed and r['family']==family and r['arm']==arm)
                        result=reused['result'];new_counts={k:0 for k in result['costs'] if k.endswith(('examples','batches','calls','updates','draws'))}
                    else:
                        assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before rerunning'
                        start=time.perf_counter()
                        try:result=execute_case(cfg,base,rows,segments,seed,phi,lr,a.out,stem,identity)
                        except FloatingPointError as e:result=dict(failed=str(e),seconds=time.perf_counter()-start)
                        new_counts=result.get('costs',{})
                    r=dict(identity=identity,replicate=rep,stream_seed=seed,family=family,arm=arm,learning_rate=lr,
                        procedure_checkpoint=str(checkpoint) if checkpoint else None,procedure_checkpoint_sha256=digest(checkpoint) if checkpoint else None,
                        inherited_from_rep=reused['replicate'] if reused else None,new_costs=new_counts,result=result)
                    path.write_text(json.dumps(r,indent=2)+'\n');records.append(r)
                    print(json.dumps(dict(rep=rep,seed=seed,family=family,arm=arm,mse=result.get('clean_mse'),recovered=result.get('recovered_segments'),segments=result.get('total_segments'),failed=result.get('failed'))),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=cfg,identity=identity,inherited_e9=old['identity'],inherited_e11=control['identity'],records=records),indent=2)+'\n');print('E17_COMPLETE',flush=True)

if __name__=='__main__':main()
