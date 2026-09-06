from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from e9_recursive_updater import student,controller,init_controller,leaves
from e17_persistent_student import PersistentStudent,make_stream
from e18_online_procedure import segment_results
from e21_evidence_assignment import PosteriorAssignment
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def schedule(k,n,seed,lo,hi):
    g=torch.Generator().manual_seed(seed);current=int(torch.randint(k,(),generator=g));out=[]
    while len(out)<n:
        length=int(torch.randint(lo,hi+1,(),generator=g));out.extend([current]*min(length,n-len(out)))
        current=(current+1+int(torch.randint(k-1,(),generator=g)))%k
    return out

def world(cfg,seed,family):
    if family in ('conflicting_functions','input_shift'):return make_stream(cfg,seed,family)
    assert family in ('coupled_shift','independent_shift','alternating_dependency')
    c=cfg['student'];k=cfg['contexts'];n=cfg['updates'];d,h,o=c['input_dim'],c['teacher_hidden'],c['output_dim']
    g=torch.Generator().manual_seed(seed);w1=torch.randn(k,d,h,generator=g)/math.sqrt(d);b=torch.randn(k,h,generator=g)*.2
    w2=torch.randn(k,h,o,generator=g)/math.sqrt(h);means=torch.randn(k,d,generator=g)*1.5;means[0]=0
    fs=schedule(k,n,seed+400000,cfg['segment_min_batches'],cfg['segment_max_batches'])
    xs=fs if family=='coupled_shift' else schedule(k,n,seed+500000,cfg['segment_min_batches'],cfg['segment_max_batches'])
    if family=='alternating_dependency':
        held=fs[0];actual=[]
        for t,f in enumerate(fs):
            if (t//512)%2:held=f
            actual.append(held)
        fs=actual
    obs_rng=torch.Generator().manual_seed(seed+300000);rows=[];segments=[];seen=set();previous=None
    for t,(f,xid) in enumerate(zip(fs,xs)):
        key=(f,xid)
        if key!=previous:
            if segments:segments[-1]['stop']=t
            segments.append(dict(start=t,stop=n,context=f*k+xid,recurring=key in seen,function_context=f,input_context=xid,
                function_changed=previous is not None and f!=previous[0],input_changed=previous is not None and xid!=previous[1]))
            seen.add(key);previous=key
        x=torch.randn(cfg['batch_size'],d,generator=obs_rng)+means[xid]
        clean=torch.tanh(x@w1[f]+b[f])@w2[f];y=clean+.03*torch.randn(cfg['batch_size'],o,generator=obs_rng)
        rows.append((x,y,clean))
    return rows,segments

class EvidenceLearner(PersistentStudent):
    def __init__(self,cfg,base,seed,spec,learned):
        super().__init__(cfg,base,seed,None,spec['student_learning_rate'])
        self.spec=spec;self.learned=learned;self.fast=torch.zeros_like(self.context);self.slow=torch.zeros_like(self.context)
        self.lagged_error=1.;self.last_gate=.5;self.gate_calls=0;self.gate_updates=0
        self.gate_params=leaves(init_controller(dict(controller_hidden=spec['gate_hidden']),seed+600000)) if learned else []
        self.gate_opt=torch.optim.Adam(self.gate_params,lr=spec['gate_learning_rate'],betas=(.9,.99)) if learned else None
        self.cached_prediction=None;self.cached_x=None
    def features(self,x):
        rms=lambda t:(t.square().mean()+1e-8).sqrt()
        return torch.stack([torch.tanh(rms(self.fast-self.slow).log()/5),torch.tanh(rms(self.fast).log()/5),
            torch.tanh(rms(self.slow).log()/5),torch.tanh(rms(x.mean(0))),
            x.new_tensor(math.tanh(math.log(self.lagged_error+1e-8)/5)),
            (self.fast*self.slow).mean()/(rms(self.fast)*rms(self.slow))]).detach()
    def predict(self,x):
        assert self.cached_prediction is None,'Prior prediction has not received its outcome'
        alpha=torch.sigmoid(controller(self.gate_params,self.features(x))) if self.learned else x.new_tensor(.5)
        self.last_gate=float(alpha.detach());context=alpha*self.fast+(1-alpha)*self.slow
        attached=torch.cat([x,context.reshape(1,-1).expand(len(x),-1)],dim=1)
        self.context=context.detach().clone();self.cached_x=x.detach().clone()
        if self.learned:self.cached_prediction=student(self.params,attached);self.gate_calls+=1
        else:
            with torch.no_grad():self.cached_prediction=student(self.params,attached)
        self.counts['student_forward_examples']+=len(x)
        return self.cached_prediction.detach()
    def observe(self,x,y):
        assert self.cached_prediction is not None and torch.equal(x,self.cached_x)
        prediction_loss=(self.cached_prediction-y).square().mean()
        if not torch.isfinite(prediction_loss):raise FloatingPointError('Nonfinite causal gate loss')
        if self.learned:
            grads=torch.autograd.grad(prediction_loss,self.gate_params)
            if not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite gate gradient')
            self.gate_opt.zero_grad(set_to_none=True)
            for p,g in zip(self.gate_params,grads):p.grad=g
            torch.nn.utils.clip_grad_norm_(self.gate_params,self.spec['gate_gradient_clip']);self.gate_opt.step()
            self.gate_updates+=1
        self.lagged_error=.95*self.lagged_error+.05*float(prediction_loss.detach())
        self.cached_prediction=None;self.cached_x=None
        # Student/replay training uses the context that actually predicted, detached.
        super().observe(x,y)
        extended=torch.cat([x,torch.ones(len(x),1)],dim=1);moment=extended.T@y/len(x)
        self.fast.mul_(self.spec['fast_decay']).add_(moment,alpha=1-self.spec['fast_decay'])
        self.slow.mul_(self.spec['slow_decay']).add_(moment,alpha=1-self.spec['slow_decay'])
        self.context=self.last_gate*self.fast+(1-self.last_gate)*self.slow
    def costs(self):
        result=super().costs();extra=[self.fast,self.slow]+self.gate_params
        if self.gate_opt:
            extra += [v for state in self.gate_opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
        result.update(gate_forward_calls=self.gate_calls,gate_gradient_calls=self.gate_updates,
            student_backward_for_gate=self.gate_updates,gate_parameters=sum(p.numel() for p in self.gate_params))
        result['stored_tensor_bytes']+=sum(p.numel()*p.element_size() for p in extra)
        return result
    def state(self):
        result=super().state();assert self.cached_prediction is None
        result['evidence']=dict(spec=self.spec,learned=self.learned,fast=self.fast,slow=self.slow,lagged_error=self.lagged_error,
            last_gate=self.last_gate,gate_calls=self.gate_calls,gate_updates=self.gate_updates,
            gate_params=[p.detach() for p in self.gate_params],gate_optimizer=self.gate_opt.state_dict() if self.gate_opt else None)
        return result
    @classmethod
    def restore(cls,state):
        e=state['evidence'];obj=cls(state['cfg'],state['base'],0,e['spec'],e['learned'])
        raw=PersistentStudent.restore(state);obj.__dict__.update(raw.__dict__)
        for name in ('fast','slow','lagged_error','last_gate','gate_calls','gate_updates'):setattr(obj,name,copy.deepcopy(e[name]))
        if e['learned']:
            obj.gate_params=leaves(e['gate_params']);obj.gate_opt=torch.optim.Adam(obj.gate_params,lr=e['spec']['gate_learning_rate'],betas=(.9,.99))
            obj.gate_opt.load_state_dict(copy.deepcopy(e['gate_optimizer']))
        return obj

def create(cfg,base,seed,spec,arm):
    c=copy.deepcopy(cfg)
    if arm in ('learned_gate','fixed_mixture'):return EvidenceLearner(c,base,seed,spec,arm=='learned_gate')
    c['context_ema']=spec['slow_decay'] if arm=='slow_prior' else spec['fast_decay']
    cls=PosteriorAssignment if arm=='fast_posterior' else PersistentStudent
    return cls(c,base,seed,None,spec['student_learning_rate'])

def preflight(spec,cfg,base):
    c=copy.deepcopy(cfg);c['updates']=12
    rows,segments=world(c,101,'conflicting_functions');old=make_stream(c,101,'conflicting_functions')
    assert all(torch.equal(a,b) for r,s in zip(rows,old[0]) for a,b in zip(r,s)) and segments==old[1]
    controls=[]
    for arm,decay,cls in [('slow_prior',.95,PersistentStudent),('fast_prior',.5,PersistentStudent),('fast_posterior',.5,PosteriorAssignment)]:
        cc=copy.deepcopy(c);cc['context_ema']=decay;original=cls(cc,base,102,None,.01);new=create(c,base,102,spec,arm)
        for x,y,_ in rows:
            assert torch.equal(original.predict(x),new.predict(x));original.observe(x,y);new.observe(x,y)
        assert all(torch.equal(a,b) for a,b in zip(original.params,new.params));controls.append(arm)
    learned=create(c,base,102,spec,'learned_gate');fixed=create(c,base,102,spec,'fixed_mixture');initial=[p.clone() for p in learned.gate_params]
    assert torch.equal(learned.predict(rows[0][0]),fixed.predict(rows[0][0]))
    learned.observe(*rows[0][:2]);fixed.observe(*rows[0][:2])
    for x,y,_ in rows[1:]:learned.predict(x);learned.observe(x,y)
    assert any(not torch.equal(a,b) for a,b in zip(initial,learned.gate_params))
    resumed=EvidenceLearner.restore(copy.deepcopy(learned.state()));x,y,_=rows[0]
    assert torch.equal(learned.predict(x),resumed.predict(x));learned.observe(x,y);resumed.observe(x,y)
    assert all(torch.equal(a,b) for a,b in zip(learned.params,resumed.params))
    assert all(torch.equal(a,b) for a,b in zip(learned.gate_params,resumed.gate_params)) and learned.costs()==resumed.costs()
    # Derivative of the current causal prediction with respect to gate output bias.
    phi=leaves([p.double() for p in learned.gate_params]);features=learned.features(x).double();params=[p.double() for p in learned.params]
    fast,slow=learned.fast.double(),learned.slow.double()
    def objective(p):
        a=torch.sigmoid(controller(p,features));z=a*fast+(1-a)*slow
        return (student(params,torch.cat([x.double(),z.reshape(1,-1).expand(len(x),-1)],1))-y.double()).square().mean()
    analytic=float(torch.autograd.grad(objective(phi),phi)[3]);eps=1e-5;plus=leaves(phi);minus=leaves(phi)
    with torch.no_grad():plus[3].add_(eps);minus[3].sub_(eps)
    numeric=float((objective(plus)-objective(minus)).detach())/(2*eps)
    assert abs(analytic-numeric)<1e-7+abs(numeric)*1e-4
    checks=[]
    for family in ('coupled_shift','independent_shift'):
        short=copy.deepcopy(c);short['updates']=128;long=copy.deepcopy(c);long['updates']=256
        a,segs=world(short,103,family);b,_=world(long,103,family)
        assert all(torch.equal(u,v) for r,s in zip(a,b) for u,v in zip(r,s))
        if family=='independent_shift':assert any(s['function_changed'] and not s['input_changed'] for s in segs) and any(s['input_changed'] and not s['function_changed'] for s in segs)
        else:assert all(s['function_changed']==s['input_changed'] for s in segs)
        checks.append(dict(family=family,exact_prefix=True,segments=segs))
    alternating=copy.deepcopy(c);alternating['updates']=1152
    _,segs=world(alternating,103,'alternating_dependency')
    changes=[s['start'] for s in segs if s['function_changed']]
    assert changes and all(512<=t<1024 for t in changes)
    return dict(exact_controls=controls,half_mixture_initial_exact=True,gate_learns=True,resume_exact=True,
        gate_derivative_analytic=analytic,gate_derivative_numeric=numeric,generator_checks=checks,
        alternating_function_changes_only_in_active_phase=True,costs=learned.costs())

def run(spec,cfg,base,seed,family,arm,rows,segments,out,identity):
    learner=create(cfg,base,seed,spec,arm);clean=[];noisy=[];norm=[];gate=[];start=time.perf_counter();failure=None
    for t,(x,y,truth) in enumerate(rows):
        prediction=learner.predict(x);cl=float((prediction-truth).square().mean());loss=float((prediction-y).square().mean())
        if not math.isfinite(loss):failure=f'Nonfinite prediction at {t}';break
        clean.append(cl);noisy.append(loss);norm.append(cl/(float(truth.square().mean())+.01))
        if isinstance(learner,EvidenceLearner):gate.append(learner.last_gate)
        try:learner.observe(x,y)
        except FloatingPointError as e:failure=str(e);break
    stem=f'{seed}-{family}-{arm}';r=dict(seed=seed,family=family,arm=arm,identity=identity,failed=failure,seconds=time.perf_counter()-start,costs=learner.costs())
    if not failure:
        assert learner.step==spec['updates'] and learner.size==cfg['replay_capacity']
        assert learner.counts['student_forward_examples']==spec['updates']*96-32
        if arm=='learned_gate':assert learner.gate_updates==spec['updates'] and learner.gate_calls==spec['updates']
        segs=segment_results(segments,clean,norm,cfg)
        r.update(clean_mse=float(np.mean(clean)),observed_mse=float(np.mean(noisy)),clean_batch_mse=clean,normalized_batch_mse=norm,
            quarter_clean_mse=[float(np.mean(q)) for q in np.array_split(clean,4)],gate_weights=gate,
            gate_quarter_means=[float(np.mean(q)) for q in np.array_split(gate,4)] if gate else None,
            segments=segs,recovered_segments=sum(s['recovery_batches'] is not None for s in segs),total_segments=len(segs),
            segments_shorter_than_criterion=sum(s['stop']-s['start']<cfg['recovery_consecutive_batches'] for s in segs))
        if family=='alternating_dependency':
            r['dependency_phases']=[dict(start=i,stop=min(i+512,len(clean)),function_changes_active=bool((i//512)%2),
                clean_mse=float(np.mean(clean[i:i+512])),gate_mean=float(np.mean(gate[i:i+512])) if gate else None)
                for i in range(0,len(clean),512)]
        torch.save(dict(identity=identity,learner_state=learner.state()),out/f'{stem}.pt');r['checkpoint']=str(out/f'{stem}.pt')
    else:r['completed_batches']=len(clean)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e22.json').read_text());cfg=json.loads((HERE/'protocol_e17.json').read_text());cfg['updates']=spec['updates']
    base=json.loads((HERE/'protocol_e9.json').read_text());names=('protocol_e22.json','e22_learned_evidence.py','protocol_e17.json','e17_persistent_student.py','e18_online_procedure.py','e21_evidence_assignment.py','protocol_e9.json','e9_recursive_updater.py')
    identity={n:digest(HERE/n) for n in names};a.out.mkdir(parents=True,exist_ok=True);check=preflight(spec,cfg,base)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=check),indent=2)+'\n');print('E22 learned evidence preflight passed',flush=True)
    if not a.execute:return
    records=[];worlds=[]
    for seed in spec['stream_seeds']:
        for family in spec['families']:
            start=time.perf_counter();rows,segments=world(cfg,seed,family);worlds.append(dict(seed=seed,family=family,generation_seconds=time.perf_counter()-start))
            for arm in spec['arms']:
                stem=f'{seed}-{family}-{arm}';path=a.out/f'{stem}.json'
                if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                else:
                    assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before restarting'
                    r=run(spec,cfg,base,seed,family,arm,rows,segments,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
                records.append(r);print(json.dumps({k:r.get(k) for k in ('seed','family','arm','clean_mse','recovered_segments','total_segments','gate_quarter_means','failed')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,worlds=worlds,records=records),indent=2)+'\n');print('E22_COMPLETE',flush=True)

if __name__=='__main__':main()
