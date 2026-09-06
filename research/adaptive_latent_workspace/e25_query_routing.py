from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from torch import nn
from e24_rare_merge_audit import generate
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bytes_of(ts):return sum(t.numel()*t.element_size() for t in ts)
def model_hash(models):
    h=hashlib.sha256()
    for m in models:
        for p in m.parameters():h.update(p.detach().numpy().tobytes())
    return h.hexdigest()

class MemoryRouter:
    def __init__(self,experts,x,y,kind,spec,seed):
        self.kind=kind;self.spec=spec;self.k=len(experts);self.output_dim=y.shape[1];self.net=None;self.opt=None;self.mean=None;self.scale=None;self.pool=None;self.labels=None
        self.fit_cost=dict(expert_target_examples=0,router_training_examples=0,router_backward_batches=0,target_seconds=0.,fit_seconds=0.,transient_training_prediction_bytes=0)
        if self.k==1:return
        start=time.perf_counter();self.mean=x.mean(0);self.scale=x.std(0,unbiased=False).clamp_min(.1);z=(x-self.mean)/self.scale
        with torch.no_grad():pred=torch.stack([m(x) for m in experts],1);labels=(pred-y[:,None,:]).square().mean(2).argmin(1)
        self.fit_cost.update(expert_target_examples=len(x)*self.k,target_seconds=time.perf_counter()-start,transient_training_prediction_bytes=bytes_of([pred]))
        if kind=='nearest_anchor':self.pool=z.clone();self.labels=labels.clone();return
        assert kind in ('learned_top1','learned_mixture')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+920000);self.net=nn.Sequential(nn.Linear(8,spec['router_hidden']),nn.Tanh(),nn.Linear(spec['router_hidden'],self.k))
        initial=model_hash([self.net]);self.opt=torch.optim.Adam(self.net.parameters(),lr=spec['router_learning_rate']);g=torch.Generator().manual_seed(seed+930000);start=time.perf_counter()
        for _ in range(spec['router_updates']):
            ids=torch.randint(len(x),(spec['router_batch_size'],),generator=g);logits=self.net(z[ids])
            loss=nn.functional.cross_entropy(logits,labels[ids]) if kind=='learned_top1' else (torch.einsum('bk,bko->bo',logits.softmax(1),pred[ids])-y[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite router fitting loss')
            self.opt.zero_grad(set_to_none=True);loss.backward();self.opt.step()
            self.fit_cost['router_training_examples']+=len(ids);self.fit_cost['router_backward_batches']+=1
        self.fit_cost['fit_seconds']=time.perf_counter()-start;self.fit_cost['parameters_changed']=model_hash([self.net])!=initial
    def tensor_bytes(self):
        ts=[t for t in (self.mean,self.scale,self.pool,self.labels) if t is not None]
        if self.net:ts+=list(self.net.parameters())
        if self.opt:ts+=[v for s in self.opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
        return bytes_of(ts)
    @torch.no_grad()
    def predict(self,x,experts):
        c=dict(expert_examples=0,router_examples=0,distance_pairs=0)
        if self.k==1:c['expert_examples']=len(x);return experts[0](x),c
        z=(x-self.mean)/self.scale
        if self.kind=='nearest_anchor':ids=self.labels[torch.cdist(z,self.pool).argmin(1)];c['distance_pairs']=len(x)*len(self.pool)
        else:
            logits=self.net(z);c['router_examples']=len(x)
            if self.kind=='learned_mixture':
                pred=torch.stack([m(x) for m in experts],1);c['expert_examples']=len(x)*self.k
                return torch.einsum('bk,bko->bo',logits.softmax(1),pred),c
            ids=logits.argmax(1)
        y=x.new_empty(len(x),self.output_dim)
        for i,m in enumerate(experts):
            mask=ids==i
            if mask.any():y[mask]=m(x[mask]);c['expert_examples']+=int(mask.sum())
        return y,c

def queries(spec,seed,law):
    wg=torch.Generator().manual_seed(seed);w1=torch.randn(2,8,32,generator=wg)/math.sqrt(8);b=torch.randn(2,32,generator=wg)*.2;w2=torch.randn(2,32,4,generator=wg)/math.sqrt(32);means=torch.randn(4,8,generator=wg)*3;means[0]=0;means[:,0]=0
    g=torch.Generator().manual_seed(seed+(900000 if law=='original_support' else 910000));shift=torch.randn(7,generator=g)*1.5;rows=[]
    for phase,prob in enumerate(spec['phase_rare_probabilities']):
        for _ in range(spec['queries_per_phase']//spec['query_batch_size']):
            n=spec['query_batch_size'];rare=torch.rand(n,generator=g)<prob;x=torch.randn(n,8,generator=g);ctx=torch.randint(4,(n,),generator=g);x[~rare]+=means[ctx[~rare]]
            x[:,0]=torch.where(rare,1.,-1.)*(1+.5*x[:,0].abs())
            if law=='shifted_support':x[:,1:]=x[:,1:]*1.5+shift
            truth=torch.empty(n,4)
            for k in (0,1):
                mask=rare if k else ~rare;truth[mask]=torch.tanh(x[mask]@w1[k]+b[k])@w2[k]
            rows.append((x,truth,rare,phase))
    return rows

def preflight(spec):
    g=torch.Generator().manual_seed(112);x=torch.randn(128,8,generator=g);y=torch.where(x[:,0:1]>0,1.,-1.).expand(-1,4)
    experts=[nn.Linear(8,4),nn.Linear(8,4)]
    with torch.no_grad():
        for i,m in enumerate(experts):m.weight.zero_();m.bias.fill_(-1 if i==0 else 1)
    for m in experts:m.requires_grad_(False)
    original=model_hash(experts);checks=[]
    for kind in ('nearest_anchor','learned_top1','learned_mixture'):
        r=MemoryRouter(experts,x,y,kind,spec,113);pred,c=r.predict(x,experts);assert torch.isfinite(pred).all() and model_hash(experts)==original
        if kind=='nearest_anchor':assert torch.equal(pred,y)
        else:assert r.fit_cost['parameters_changed'] and r.fit_cost['router_backward_batches']==spec['router_updates'] and r.fit_cost['router_training_examples']==spec['router_updates']*spec['router_batch_size']
        import io
        buf=io.BytesIO();torch.save(r,buf);buf.seek(0);restored=torch.load(buf,weights_only=False);assert torch.equal(restored.predict(x,experts)[0],pred)
        one=MemoryRouter(experts[:1],x,y,kind,spec,113);assert torch.equal(one.predict(x,experts[:1])[0],experts[0](x)) and one.tensor_bytes()==0
        checks.append(dict(kind=kind,mean_fit_error=float((pred-y).square().mean()),roundtrip_exact=True,one_expert_bypass_exact=True))
    # Finite-difference derivative of soft mixture output-bias parameter on an actual fitted dummy router.
    net=copy.deepcopy(r.net).double();z=((x[:8]-r.mean)/r.scale).double();preds=torch.stack([m(x[:8]) for m in experts],1).double();target=y[:8].double()
    def objective():return (torch.einsum('bk,bko->bo',net(z).softmax(1),preds)-target).square().mean()
    param=net[-1].bias;analytic=float(torch.autograd.grad(objective(),param)[0][0]);eps=1e-5
    with torch.no_grad():param[0]+=eps
    plus=float(objective().detach())
    with torch.no_grad():param[0]-=2*eps
    minus=float(objective().detach());numeric=(plus-minus)/(2*eps)
    assert abs(analytic-numeric)<1e-7+abs(numeric)*1e-4
    return dict(checks=checks,expert_weights_unchanged=True,mixture_gradient_analytic=analytic,mixture_gradient_numeric=numeric)

def run_queries(router,active,experts,rows):
    sq=[];flags=[];phases=[];oracle=[];all_errors=[];cost=dict(expert_examples=0,router_examples=0,distance_pairs=0);start=time.perf_counter();failure=None
    for x,y,rare,phase in rows:
        if router is None:
            with torch.no_grad():pred=active(x)
            c=dict(expert_examples=len(x),router_examples=0,distance_pairs=0)
        else:pred,c=router.predict(x,experts)
        err=(pred-y).square().mean(1)
        if not torch.isfinite(err).all():failure='Nonfinite query prediction';break
        sq.extend(err.tolist());flags.extend(rare.tolist());phases.extend([phase]*len(x))
        for k,v in c.items():cost[k]+=v
    seconds=time.perf_counter()-start
    if failure:return dict(failed=failure,mse=None,rare_mse=None,common_mse=None,completed_queries=len(sq),query_cost=cost,query_seconds=seconds,per_query_errors=sq)
    # Privileged diagnostic is separate and is never passed to a router.
    start=time.perf_counter()
    with torch.no_grad():
        for x,y,rare,phase in rows:
            e=torch.stack([(m(x)-y).square().mean(1) for m in experts],1);oracle.extend(e.min(1).values.tolist());all_errors.extend(e.tolist())
    arr=np.array(sq);mask=np.array(flags);ph=np.array(phases);e=np.array(all_errors)
    return dict(failed=failure,mse=float(arr.mean()),rare_mse=float(arr[mask].mean()),common_mse=float(arr[~mask].mean()),rare_queries=int(mask.sum()),query_examples=len(arr),
        per_query_errors=sq,phase_mse=[float(arr[ph==i].mean()) for i in range(8)],query_cost=cost,query_seconds=seconds,
        privileged_best_per_query_mse=float(np.mean(oracle)),privileged_best_single_rare=float(e[mask].mean(0).min()),privileged_best_single_common=float(e[~mask].mean(0).min()),
        diagnostic_expert_examples=len(arr)*len(experts),diagnostic_seconds=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e25.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
    names=('protocol_e25.json','e25_query_routing.py','e24_rare_merge_audit.py','protocol_e24.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py')
    identity={n:digest(HERE/n) for n in names};checks=preflight(spec);(a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks),indent=2)+'\n');print('E25 query routing preflight passed',flush=True)
    if not a.execute:return
    assert not list(a.out.glob('*-router.pt')),'Existing scored router checkpoints: inspect before restarting'
    old=json.loads((a.runs/'e24/complete.json').read_text());audit=json.loads((HERE/'results/e24_audit.json').read_text());records=[];sources=[]
    for seed in spec['source_seeds']:
        for source_arm in spec['source_arms']:
            src=next(r for r in old['records'] if r['seed']==seed and r['arm']==source_arm);verified=next(r for r in audit['checks'] if r['seed']==seed and r['arm']==source_arm)
            path=Path(src['checkpoint']);assert digest(path)==verified['checkpoint_sha256'];start=time.perf_counter();l=torch.load(path,weights_only=False)['learner'];load_seconds=time.perf_counter()-start
            experts=list(l.models);active=l.fast_model if l.fast_model is not None else l.models[l.active]
            if l.fast_model is not None:experts.append(l.fast_model)
            for m in experts:m.requires_grad_(False)
            buffers=list(l.anchors)
            if l.scratch is not None:buffers.append(l.scratch)
            x=torch.cat([b.x[:b.size] for b in buffers]);y=torch.cat([b.y[:b.size] for b in buffers]);initial_hash=model_hash(experts);initial_cost=l.costs()
            with torch.no_grad():assert torch.equal(active(x[:32]),l.predict(x[:32],count=False))
            source=dict(seed=seed,source_arm=source_arm,checkpoint=str(path),checkpoint_sha256=verified['checkpoint_sha256'],experts=len(experts),temporary_expert=l.fast_model is not None,
                memory_examples=len(x),source_load_seconds=load_seconds,inherited_learner_costs=initial_cost,inherited_e24_loop_seconds=src['loop_seconds_including_audit']);sources.append(source)
            for arm in spec['arms']:
                stem=f'{seed}-{source_arm}-{arm}';router=None if arm=='active' else MemoryRouter(experts,x,y,arm,spec,seed)
                assert model_hash(experts)==initial_hash and l.costs()==initial_cost
                fit=dict(router.fit_cost) if router else {};extra=router.tensor_bytes() if router else 0
                torch.save(dict(identity=identity,source=source,router=router),a.out/f'{stem}-router.pt')
                for law in spec['query_laws']:
                    outfile=a.out/f'{stem}-{law}.json';assert not outfile.exists(),'Do not restart scored output without inspecting existing state'
                    rows=queries(spec,seed,law);result=run_queries(router,active,experts,rows);r=dict(seed=seed,source_arm=source_arm,arm=arm,query_law=law,identity=identity,source=source,fit_cost=fit,extra_routing_tensor_bytes=extra,router_checkpoint=str(a.out/f'{stem}-router.pt'),**result)
                    outfile.write_text(json.dumps(r,indent=2)+'\n');records.append(r);print(json.dumps({k:r[k] for k in ('seed','source_arm','arm','query_law','mse','rare_mse','failed')}),flush=True)
                assert model_hash(experts)==initial_hash and l.costs()==initial_cost
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,sources=sources,records=records),indent=2)+'\n');print('E25_COMPLETE',flush=True)
if __name__=='__main__':main()
