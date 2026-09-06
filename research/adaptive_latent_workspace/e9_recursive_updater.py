from pathlib import Path
import argparse,copy,hashlib,json,math,time
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent

def student(params,x):
    x=torch.tanh(F.linear(x,params[0],params[1]))
    x=torch.tanh(F.linear(x,params[2],params[3]))
    return F.linear(x,params[4],params[5])

def controller(phi,features):return F.linear(torch.tanh(F.linear(features,phi[0],phi[1])),phi[2],phi[3]).squeeze()

def init_controller(cfg,seed,dtype=torch.float32):
    g=torch.Generator().manual_seed(seed);h=cfg['controller_hidden']
    return [torch.randn(h,6,generator=g,dtype=dtype)/math.sqrt(6),torch.zeros(h,dtype=dtype),
            torch.zeros(1,h,dtype=dtype),torch.zeros(1,dtype=dtype)]

def leaves(xs):return [x.detach().clone().requires_grad_(True) for x in xs]

def update(params,grads,mom,second,step,loss,phi,cfg,lr):
    b1,b2=cfg['beta1'],cfg['beta2'];new=[];ms=[];vs=[]
    for p,g,m,v in zip(params,grads,mom,second):
        m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g.square()
        mh=m/(1-b1**step);vh=v/(1-b2**step)
        if phi is None:mult=1.
        else:
            rms=lambda t:(t.square().mean()+1e-8).sqrt()
            features=torch.stack([torch.tanh(rms(g).log()/5),torch.tanh(rms(p).log()/5),
                (g*m).mean()/(rms(g)*rms(m)),torch.tanh((v.mean()+1e-8).sqrt().log()/5),
                loss.new_tensor(step/(step+10)),torch.tanh((loss+1e-8).log()/5)])
            mult=controller(phi,features).clamp(-2,2).exp()
        new.append(p-lr*mult*mh/(vh.sqrt()+1e-8));ms.append(m);vs.append(v)
    return new,ms,vs

def task(cfg,seed,phi,steps=None,distribution='in_distribution',meta=False,lr=None,dtype=torch.float32):
    steps=steps or cfg['inner_steps'];lr=lr or cfg['base_learning_rate'];c=cfg['student']
    g=torch.Generator().manual_seed(seed);d,h,o,th=c['input_dim'],c['hidden_dim'],c['output_dim'],c['teacher_hidden']
    tw1=torch.randn(d,th,generator=g,dtype=dtype)/math.sqrt(d)
    tb=torch.randn(th,generator=g,dtype=dtype)*.2;tw2=torch.randn(th,o,generator=g,dtype=dtype)/math.sqrt(th)
    scale=float(torch.exp(torch.empty((),dtype=dtype).uniform_(-.5,.5,generator=g)))
    params=leaves([torch.randn(h,d,generator=g,dtype=dtype)/math.sqrt(d),torch.zeros(h,dtype=dtype),
        torch.randn(h,h,generator=g,dtype=dtype)/math.sqrt(h),torch.zeros(h,dtype=dtype),
        torch.randn(o,h,generator=g,dtype=dtype)/math.sqrt(h),torch.zeros(o,dtype=dtype)])
    m=[torch.zeros_like(p) for p in params];v=[torch.zeros_like(p) for p in params];losses=[]
    def sample(n):
        x=torch.randn(n,d,generator=g,dtype=dtype)*(4. if distribution=='scaled_inputs' else 1.)
        hidden=x@tw1+tb
        clean=(torch.relu(hidden) if distribution=='relu_teacher' else torch.tanh(hidden))@tw2*scale
        return x,clean+.03*torch.randn(n,o,generator=g,dtype=dtype)
    for t in range(1,steps+1):
        x,y=sample(cfg['batch_size']);loss=(student(params,x)-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite student loss')
        losses.append(float(loss.detach()))
        grads=torch.autograd.grad(loss,params,create_graph=meta)
        params,m,v=update(params,grads,m,v,t,loss,phi,cfg,lr)
        if not meta:
            params=leaves(params);m=[z.detach() for z in m];v=[z.detach() for z in v]
    x,y=sample(cfg['query_size']);query=(student(params,x)-y).square().mean()
    return query,dict(prequential_mse=sum(losses)/len(losses),query_mse=float(query.detach()),
        inner_updates=steps,training_observations=steps*cfg['batch_size'],query_observations=cfg['query_size'])

def evaluate(cfg,phi,seeds,**kwargs):
    rows=[];start=time.perf_counter()
    frozen=None if phi is None else [p.detach() for p in phi]
    for seed in seeds:
        _,row=task(cfg,seed,frozen,**kwargs);row['seed']=seed;rows.append(row)
    return dict(rows=rows,query_mse=sum(r['query_mse'] for r in rows)/len(rows),
        prequential_mse=sum(r['prequential_mse'] for r in rows)/len(rows),seconds=time.perf_counter()-start)

def meta_loss(cfg,phi,seeds):return torch.stack([task(cfg,s,phi,meta=True,dtype=phi[0].dtype)[0] for s in seeds]).mean()

def preflight(cfg):
    c=copy.deepcopy(cfg);c['inner_steps']=3;phi=leaves(init_controller(c,91,torch.float64))
    analytic=float(torch.autograd.grad(meta_loss(c,phi,[92]),phi)[3].item())
    eps=1e-5;plus=leaves(phi);minus=leaves(phi)
    with torch.no_grad():plus[3].add_(eps);minus[3].sub_(eps)
    numeric=float((meta_loss(c,plus,[92])-meta_loss(c,minus,[92])).detach())/(2*eps)
    assert abs(analytic-numeric)<1e-5+1e-3*abs(numeric),(analytic,numeric)
    b=evaluate(c,phi,[92],dtype=torch.float64)
    # Compare same dtype for the zero-output Adam equivalence.
    aa=evaluate(c,None,[92],dtype=torch.float64)
    assert abs(aa['query_mse']-b['query_mse'])<1e-12
    return dict(meta_gradient_analytic=analytic,meta_gradient_finite_difference=numeric,adam_equivalence=True)

def run(cfg,rep,out,identity):
    start=time.perf_counter();phi=leaves(init_controller(cfg,rep));opt=torch.optim.Adam(phi,lr=cfg['bootstrap_learning_rate'])
    prefix=rep*100000;bootstrap=[]
    for t in range(cfg['bootstrap_updates']):
        seeds=[prefix+t*cfg['tasks_per_meta_update']+i for i in range(cfg['tasks_per_meta_update'])]
        loss=meta_loss(cfg,phi,seeds);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(phi,1.);opt.step()
        bootstrap.append(float(loss.detach()))
        if (t+1)%32==0:print(json.dumps(dict(replicate=rep,bootstrap_updates=t+1,meta_loss=bootstrap[-1])),flush=True)
    bootstrap_seconds=time.perf_counter()-start;frozen=leaves(phi)
    torch.save(dict(phi=[p.detach() for p in frozen],identity=identity),out/f'{rep}-bootstrap.pt')
    anchors=[prefix+10000+i for i in range(cfg['retention_validation_tasks'])]
    selection_seeds=[prefix+20000+i for i in range(cfg['baseline_selection_tasks'])]
    baseline_selection={str(lr):evaluate(cfg,None,selection_seeds,lr=lr) for lr in cfg['baseline_learning_rates']}
    best_lr=min(cfg['baseline_learning_rates'],key=lambda lr:baseline_selection[str(lr)]['query_mse'])
    rounds=[];m=[torch.zeros_like(p) for p in phi];v=[torch.zeros_like(p) for p in phi];accepted=0
    for t in range(cfg['self_rounds']):
        started=time.perf_counter();train_seeds=[prefix+30000+t*10+i for i in range(cfg['tasks_per_meta_update'])]
        loss=meta_loss(cfg,phi,train_seeds);grads=torch.autograd.grad(loss,phi)
        # The controller processes gradients of its own parameters. No outer
        # Adam optimizer proposes this update; bootstrap Adam has finished.
        candidate,cm,cv=update(phi,[g.detach() for g in grads],m,v,accepted+1,loss.detach(),
            [p.detach() for p in phi],cfg,cfg['self_learning_rate'])
        candidate=leaves(candidate);fresh=[prefix+40000+t*10+i for i in range(cfg['fresh_validation_tasks_per_round'])]
        before=evaluate(cfg,phi,fresh);after=evaluate(cfg,candidate,fresh)
        old_anchor=evaluate(cfg,phi,anchors);new_anchor=evaluate(cfg,candidate,anchors)
        accept=(after['query_mse']<before['query_mse']*(1-cfg['accept_improvement_fraction']) and
            new_anchor['query_mse']<=old_anchor['query_mse']*(1+cfg['retention_tolerance_fraction']))
        if accept:phi=candidate;m=[z.detach() for z in cm];v=[z.detach() for z in cv];accepted+=1
        row=dict(round=t,accepted=accept,training_meta_loss=float(loss.detach()),fresh_before=before,fresh_after=after,
            anchor_before=old_anchor,anchor_after=new_anchor,seconds=time.perf_counter()-started)
        rounds.append(row);print(json.dumps(dict(replicate=rep,self_round=t,accepted=accept,
            before=before['query_mse'],after=after['query_mse'])),flush=True)
    torch.save(dict(phi=[p.detach() for p in phi],identity=identity),out/f'{rep}-self-applied.pt')
    final=[]
    for distribution in cfg['final_distributions']:
        for horizon in cfg['final_horizons']:
            seeds=[prefix+60000+i for i in range(cfg['final_tasks'])]
            for name,control,lr in [('adam_default',None,cfg['base_learning_rate']),('adam_selected',None,best_lr),
                ('frozen_bootstrap',frozen,cfg['base_learning_rate']),('self_applied',phi,cfg['base_learning_rate'])]:
                result=evaluate(cfg,control,seeds,steps=horizon,distribution=distribution,lr=lr)
                result.update(arm=name,distribution=distribution,horizon=horizon);final.append(result)
                print(json.dumps(dict(replicate=rep,arm=name,distribution=distribution,horizon=horizon,
                    query_mse=result['query_mse'],seconds=result['seconds'])),flush=True)
    return dict(replicate=rep,identity=identity,bootstrap_losses=bootstrap,bootstrap_seconds=bootstrap_seconds,
        baseline_selection=baseline_selection,selected_baseline_lr=best_lr,self_rounds=rounds,accepted_self_updates=accepted,
        final=final,controller_parameters=sum(p.numel() for p in phi),total_seconds=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);cfg=json.loads((HERE/'protocol_e9.json').read_text())
    check=preflight(cfg);print(json.dumps(dict(preflight=check)),flush=True)
    if not a.execute:return
    a.out.mkdir(parents=True,exist_ok=True)
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e9.json','e9_recursive_updater.py')}
    records=[]
    for rep in cfg['replicates']:
        path=a.out/f'{rep}.json'
        if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
        else:r=run(cfg,rep,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
        records.append(r)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=cfg,identity=identity,preflight=check,records=records),indent=2)+'\n')
    print('E9_COMPLETE',flush=True)

if __name__=='__main__':main()
