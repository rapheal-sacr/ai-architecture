from pathlib import Path
import argparse,copy,hashlib,json,math,time
from collections import deque
import numpy as np
import torch
from torch import nn
from e6_sharing import GuardedSharing,factory
from e24_rare_merge_audit import generate
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nbytes(ts):return sum(t.numel()*t.element_size() for t in ts)
def bank(l):
    b={i:m for i,m in enumerate(l.models)}
    if l.fast_model is not None:b[l.cfg['module_cap']]=l.fast_model
    return b
def library_signature(l):return (len(l.models),l.fast_model is not None,l.merges,len(l.events))
def memory(l):
    bs=list(l.anchors)+([l.scratch] if l.scratch is not None else []);bs=[b for b in bs if b.size]
    return (torch.cat([b.x[:b.size] for b in bs]),torch.cat([b.y[:b.size] for b in bs])) if bs else (torch.empty(0,8),torch.empty(0,4))
def core_state(l):
    def buf(a):return None if a is None else dict(x=a.x[:a.size].clone(),y=a.y[:a.size].clone(),size=a.size,count=a.count,capacity=a.capacity)
    scalars={k:copy.deepcopy(v) for k,v in vars(l).items() if isinstance(v,(int,float,bool,str)) or v is None}
    return dict(scalars=scalars,models=[copy.deepcopy(m.state_dict()) for m in l.models],opts=[copy.deepcopy(o.state_dict()) for o in l.opts],
        fast=None if l.fast_model is None else copy.deepcopy(l.fast_model.state_dict()),fast_opt=None if l.fast_opt is None else copy.deepcopy(l.fast_opt.state_dict()),
        anchors=[buf(a) for a in l.anchors],scratch=buf(l.scratch),ages=list(l.ages),expected=list(l.expected_error),
        fast_scores=list(l.fast_scores),old_scores=list(l.old_scores),events=copy.deepcopy(l.events),cfg=copy.deepcopy(l.cfg),
        scratch_model_is_fast=l.scratch_model is l.fast_model,scratch_model_none=l.scratch_model is None)
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class OnlineAccess:
    def __init__(self,l,kind,spec,seed):
        self.kind=kind;self.spec=spec;self.seed=seed;self.step=0;self.max_slots=l.cfg['module_cap']+1
        self.net=None;self.opt=None;self.mean=None;self.scale=None;self.pool=None;self.labels=None;self.signature=None
        self.rng=torch.Generator().manual_seed(seed+2700000) if kind=='online_top1' else None
        self.counts=dict(target_expert_examples=0,router_train_examples=0,router_backward_batches=0,cache_refreshes=0)
        start=time.perf_counter()
        if kind=='nearest_cached':self.refresh(l)
        elif kind=='online_top1' and len(bank(l))>1:
            x,y=memory(l);self.initialize(x)
            for _ in range(spec['bootstrap_router_updates']):
                ids=torch.randint(len(x),(spec['router_bootstrap_batch'],),generator=self.rng);self.fit(l,x[ids],y[ids])
        self.bootstrap_seconds=time.perf_counter()-start
    def initialize(self,x):
        self.mean=x.mean(0);self.scale=x.std(0,unbiased=False).clamp_min(.1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed+2800000);self.net=nn.Sequential(nn.Linear(8,self.spec['router_hidden']),nn.Tanh(),nn.Linear(self.spec['router_hidden'],self.max_slots))
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.spec['router_learning_rate'])
    @torch.no_grad()
    def targets(self,l,x,y):
        b=bank(l);slots=list(b);pred=torch.stack([m(x) for m in b.values()],1);index=(pred-y[:,None,:]).square().mean(2).argmin(1)
        self.counts['target_expert_examples']+=len(x)*len(b);return torch.tensor(slots)[index]
    def refresh(self,l):
        x,y=memory(l);self.signature=library_signature(l)
        if len(bank(l))==1 or not len(x):self.pool=None;self.labels=None;self.mean=None;self.scale=None;return
        self.mean=x.mean(0);self.scale=x.std(0,unbiased=False).clamp_min(.1);self.pool=(x-self.mean)/self.scale;self.labels=self.targets(l,x,y);self.counts['cache_refreshes']+=1
    def fit(self,l,x,y):
        if len(bank(l))<2:return
        if self.net is None:
            mx,_=memory(l);self.initialize(mx if len(mx) else x)
        labels=self.targets(l,x,y);logits=self.net((x-self.mean)/self.scale);valid=torch.zeros(self.max_slots,dtype=torch.bool);valid[list(bank(l))]=True
        loss=nn.functional.cross_entropy(logits.masked_fill(~valid,-1e9),labels)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite online router loss')
        self.opt.zero_grad(set_to_none=True);loss.backward();self.opt.step();self.counts['router_train_examples']+=len(x);self.counts['router_backward_batches']+=1
    @torch.no_grad()
    def predict(self,l,x):
        b=bank(l);c=dict(expert_examples=0,router_examples=0,distance_pairs=0)
        if self.kind=='active' or len(b)==1:
            y=l.predict(x,count=False);c['expert_examples']=len(x);return y,c
        if self.kind=='nearest_cached':
            assert self.signature==library_signature(l),'Stale expert slots must refresh after observe'
            if self.pool is None:y=l.predict(x,count=False);c['expert_examples']=len(x);return y,c
            ids=self.labels[torch.cdist((x-self.mean)/self.scale,self.pool).argmin(1)];c['distance_pairs']=len(x)*len(self.pool)
        else:
            assert self.net is not None;logits=self.net((x-self.mean)/self.scale);valid=torch.zeros(self.max_slots,dtype=torch.bool);valid[list(b)]=True
            ids=logits.masked_fill(~valid,-1e9).argmax(1);c['router_examples']=len(x)
        assert set(ids.tolist()).issubset(b);y=x.new_empty(len(x),4)
        for slot,m in b.items():
            mask=ids==slot
            if mask.any():y[mask]=m(x[mask]);c['expert_examples']+=int(mask.sum())
        return y,c
    def observe(self,l,x,y):
        self.step+=1
        if self.kind=='nearest_cached':
            if self.signature!=library_signature(l) or self.step%self.spec['refresh_batches']==0:self.refresh(l)
        elif self.kind=='online_top1' and len(bank(l))>1:
            mx,my=memory(l);ids=torch.randint(len(mx),(self.spec['router_replay_batch'],),generator=self.rng)
            self.fit(l,torch.cat([x,mx[ids]]),torch.cat([y,my[ids]]))
    def tensor_bytes(self):
        ts=[t for t in (self.mean,self.scale,self.pool,self.labels) if t is not None]
        if self.net:ts+=list(self.net.parameters())
        if self.opt:ts+=[v for s in self.opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
        if self.kind=='online_top1':ts+=[self.rng.get_state()]
        return nbytes(ts)

def fresh_probes(spec,seed):
    g=torch.Generator().manual_seed(seed);w1=torch.randn(2,8,32,generator=g)/math.sqrt(8);b=torch.randn(2,32,generator=g)*.2;w2=torch.randn(2,32,4,generator=g)/math.sqrt(32);means=torch.randn(4,8,generator=g)*3;means[0]=0;means[:,0]=0
    q=torch.Generator().manual_seed(seed+2600000);out={}
    for k,name in enumerate(('common','rare')):
        n=spec['probe_examples_per_region'];x=torch.randn(n,8,generator=q)
        if k==0:x+=means[torch.randint(4,(n,),generator=q)]
        x[:,0]=(-1 if k==0 else 1)*(1+.5*x[:,0].abs());y=torch.tanh(x@w1[k]+b[k])@w2[k];out[name]=(x,y)
    return out
@torch.no_grad()
def probe(l,r,probes):
    result={};examples=0;router_examples=0;distance_pairs=0
    for name,(x,y) in probes.items():
        pred,c=r.predict(l,x);result[name]=float((pred-y).square().mean());scores=[float((m(x)-y).square().mean()) for m in bank(l).values()]
        result['privileged_best_'+name]=min(scores);examples+=c['expert_examples']+len(x)*len(bank(l));router_examples+=c['router_examples'];distance_pairs+=c['distance_pairs']
    return result,dict(expert_examples=examples,router_examples=router_examples,distance_pairs=distance_pairs)

def preflight(spec,cfg):
    g=torch.Generator().manual_seed(114);rows=[(torch.randn(32,8,generator=g),torch.ones(32,4)*(0 if i<100 else 3)) for i in range(210)]
    c=copy.deepcopy(cfg);c['merge_absolute_tolerance']=100.;results=[];checks=[]
    for kind in spec['arms']:
        torch.manual_seed(115);l=GuardedSharing(c,torch.device('cpu'));r=OnlineAccess(l,kind,spec,116)
        for x,y in rows:
            before=torch.get_rng_state().clone();prediction,_=r.predict(l,x);assert torch.equal(before,torch.get_rng_state()) and torch.isfinite(prediction).all()
            l.observe(x,y);before=torch.get_rng_state().clone();r.observe(l,x,y);assert torch.equal(before,torch.get_rng_state())
        results.append((core_state(l),torch.get_rng_state().clone()))
        if kind=='online_top1':assert r.counts['router_backward_batches']>0 and r.net is not None
        import io
        buf=io.BytesIO();torch.save((l,r,torch.get_rng_state()),buf);buf.seek(0);ll,rr,rng=torch.load(buf,weights_only=False);x,y=rows[-1]
        assert torch.equal(r.predict(l,x)[0],rr.predict(ll,x)[0]);torch.set_rng_state(rng);l.observe(x,y);r.observe(l,x,y);end_rng=torch.get_rng_state().clone()
        torch.set_rng_state(rng);ll.observe(x,y);rr.observe(ll,x,y);assert equal(core_state(l),core_state(ll)) and torch.equal(end_rng,torch.get_rng_state())
        assert torch.equal(r.predict(l,x)[0],rr.predict(ll,x)[0]) and r.counts==rr.counts
        checks.append(dict(kind=kind,restore_and_next_update_exact=True,counts=r.counts,library_events=len(l.events)))
    assert all(equal(a,results[0][0]) and torch.equal(b,results[0][1]) for a,b in results)
    return dict(core_trajectory_and_rng_exact_across_selectors=True,valid_slot_masks=True,checks=checks)

def reconstruct(spec,cfg,old_spec,source,verified,rows,out,identity):
    name=f"{source['seed']}-{source['arm']}-reconstructed.pt";path=out/name
    if path.exists():
        p=torch.load(path,weights_only=False);assert p['identity']==identity and p['metadata']['source_sha256']==verified['checkpoint_sha256'];return p
    assert digest(source['checkpoint'])==verified['checkpoint_sha256'];ts=time.perf_counter();saved=torch.load(source['checkpoint'],weights_only=False)['learner'];source_load_seconds=time.perf_counter()-ts
    ts=time.perf_counter();short,_,counts=generate(old_spec,source['seed']);prefix_generation_seconds=time.perf_counter()-ts;assert len(short)==old_spec['updates']
    assert all(torch.equal(x,y) for a,b in zip(short,rows) for x,y in zip(a,b));start=time.perf_counter();torch.manual_seed(source['seed']+400000)
    l=factory(source['arm'],cfg,torch.device('cpu'))
    for i,(x,y,truth) in enumerate(short):
        error=float((l.predict(x)-truth).square().mean());assert error==source['clean_batch_mse'][i],('Prefix error mismatch',source['seed'],source['arm'],i)
        l.observe(x,y)
    assert equal(core_state(l),core_state(saved)),('Source state mismatch',source['seed'],source['arm'])
    metadata=dict(seed=source['seed'],source_arm=source['arm'],source_sha256=verified['checkpoint_sha256'],prefix_errors_exact=True,prefix_data_exact=True,full_core_state_exact=True,source_load_seconds=source_load_seconds,prefix_generation_seconds=prefix_generation_seconds,reconstruction_seconds=time.perf_counter()-start,inherited_e24_loop_seconds=source['loop_seconds_including_audit'])
    p=dict(identity=identity,learner=l,rng=torch.get_rng_state().clone(),metadata=metadata);torch.save(p,path);print('RECONSTRUCTED '+json.dumps(metadata),flush=True);return p

def run(spec,source,kind,tail,probes,out,identity):
    l=copy.deepcopy(source['learner']);torch.set_rng_state(source['rng'].clone());start_cost=l.costs();r=OnlineAccess(l,kind,spec,source['metadata']['seed']);errors=[];endpoints=[];query_cost=dict(expert_examples=0,router_examples=0,distance_pairs=0);audit_cost=query_cost.copy();audit_seconds=0.;failure=None;start=time.perf_counter()
    for i,(x,y,truth) in enumerate(tail):
        pred,c=r.predict(l,x);err=float((pred-truth).square().mean())
        if not math.isfinite(err):failure=f'Nonfinite prediction at {i}';break
        errors.append(err)
        for k,v in c.items():query_cost[k]+=v
        try:l.observe(x,y);r.observe(l,x,y)
        except (RuntimeError,FloatingPointError) as e:failure=str(e);break
        if (i+1)%spec['probe_every_batches']==0:
            ts=time.perf_counter();rng=torch.get_rng_state().clone();p,c=probe(l,r,probes);assert torch.equal(rng,torch.get_rng_state());p['continuation_batch']=i+1;endpoints.append(p)
            for k,v in c.items():audit_cost[k]+=v
            audit_seconds+=time.perf_counter()-ts
    elapsed=time.perf_counter()-start;seed=source['metadata']['seed'];source_arm=source['metadata']['source_arm'];stem=f'{seed}-{source_arm}-{kind}';final=l.costs()
    record=dict(seed=seed,source_arm=source_arm,arm=kind,identity=identity,source=source['metadata'],failed=failure,completed_updates=len(errors),clean_batch_mse=errors,mean_clean_mse=float(np.mean(errors)) if errors else None,
        endpoints=endpoints,start_core_costs=start_cost,final_core_costs=final,new_core_forward_examples=final['forward_examples']-start_cost['forward_examples'],
        new_core_updates=final['optimizer_updates']-start_cost['optimizer_updates'],query_cost=query_cost,router_fit_cost=r.counts,router_tensor_bytes=r.tensor_bytes(),
        router_bootstrap_seconds=r.bootstrap_seconds,loop_seconds_including_audit=elapsed,audit_seconds=audit_seconds,audit_cost=audit_cost)
    if not failure:
        assert l.updates==source['learner'].updates+spec['continuation_updates'];path=out/f'{stem}.pt';torch.save(dict(identity=identity,learner=l,router=r,rng=torch.get_rng_state().clone()),path);record['checkpoint']=str(path);record['checkpoint_bytes']=path.stat().st_size
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e26.json').read_text());old_spec=json.loads((HERE/'protocol_e24.json').read_text());cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(json.loads((HERE/'protocol_e6.json').read_text()));cfg.update(module_cap=old_spec['module_cap'],anchor_capacity=old_spec['anchor_capacity'])
    names=('protocol_e26.json','e26_online_access.py','protocol_e24.json','e24_rare_merge_audit.py','protocol_e1.json','protocol_e2.json','protocol_e6.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py');identity={n:digest(HERE/n) for n in names};a.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();check=preflight(spec,cfg);(a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=check,seconds=time.perf_counter()-start),indent=2)+'\n');print('E26 online access preflight passed',flush=True)
    if not a.execute:return
    old=json.loads((a.runs/'e24/complete.json').read_text());audit=json.loads((HERE/'results/e24_audit.json').read_text());records=[];sources=[];worlds=[]
    for n,h in old['identity'].items():assert digest(HERE/n)==h
    for seed in spec['source_seeds']:
        ts=time.perf_counter();rows,_,_=generate(old_spec,seed,old_spec['updates']+spec['continuation_updates']);worlds.append(dict(seed=seed,generation_seconds=time.perf_counter()-ts));tail=rows[old_spec['updates']:];probes=fresh_probes(spec,seed)
        for source_arm in spec['source_arms']:
            src=next(r for r in old['records'] if r['seed']==seed and r['arm']==source_arm);verified=next(r for r in audit['checks'] if r['seed']==seed and r['arm']==source_arm)
            start=reconstruct(spec,cfg,old_spec,src,verified,rows,a.out,identity);sources.append(start['metadata']);reference=None;reference_rng=None
            for kind in spec['arms']:
                stem=f'{seed}-{source_arm}-{kind}';path=a.out/f'{stem}.json'
                if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                else:
                    assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before restarting'
                    r=run(spec,start,kind,tail,probes,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
                records.append(r)
                if not r['failed']:
                    final=torch.load(r['checkpoint'],weights_only=False);state=core_state(final['learner'])
                    if reference is None:reference=state;reference_rng=final['rng']
                    else:assert equal(reference,state) and torch.equal(reference_rng,final['rng']),('Core trajectory diverged across selectors',seed,source_arm,kind)
                print(json.dumps(dict(seed=seed,source_arm=source_arm,arm=kind,mse=r['mean_clean_mse'],final_probe=r['endpoints'][-1] if r['endpoints'] else None,failed=r['failed'])),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,sources=sources,worlds=worlds,records=records),indent=2)+'\n');print('E26_COMPLETE',flush=True)
if __name__=='__main__':main()
