"""E1: actual neural learning in a stream with hidden recurring contexts.

Run with --execute to consume the frozen protocol. No WAM/witness dependencies.
Learners see x, predict, then see y. Only OracleModules receives regime IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn

HERE=Path(__file__).resolve().parent


class TeacherWorld:
    def __init__(self,cfg,seed):
        g=torch.Generator().manual_seed(seed+100000)
        k,d,h,o=cfg['teacher_contexts'],cfg['input_dim'],cfg['teacher_hidden'],cfg['output_dim']
        self.w1=torch.randn(k,d,h,generator=g)/math.sqrt(d)
        self.b1=torch.randn(k,h,generator=g)*.2
        self.w2=torch.randn(k,h,o,generator=g)/math.sqrt(h)
        self.noise=cfg['teacher_output_noise_std']
        self.d=d;self.o=o

    def sample(self,context,n,generator,device):
        x=torch.randn(n,self.d,generator=generator)
        clean=torch.tanh(x@self.w1[context]+self.b1[context])@self.w2[context]
        y=clean+self.noise*torch.randn(n,self.o,generator=generator)
        return x.to(device),y.to(device),clean.to(device)


def net(d,h,o,device):
    return nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,o)).to(device)


class Replay:
    def __init__(self,n,d,o,device):
        self.x=torch.empty(n,d,device=device);self.y=torch.empty(n,o,device=device)
        self.size=0;self.count=0;self.capacity=n

    def add(self,x,y):
        # Uniform reservoir, not FIFO. Future labels cannot enter a past prediction.
        for xi,yi in zip(x,y):
            i=self.count if self.count<self.capacity else int(torch.randint(self.count+1,()).item())
            if i<self.capacity:self.x[i]=xi;self.y[i]=yi
            self.count+=1;self.size=min(self.count,self.capacity)

    def sample(self,n):
        ids=torch.randint(self.size,(n,),device=self.x.device)
        return self.x[ids],self.y[ids]

    def nbytes(self):return self.x.numel()*self.x.element_size()+self.y.numel()*self.y.element_size()


class Base:
    def __init__(self,cfg,device):
        self.cfg=cfg;self.device=device
        self.forward_examples=0;self.training_examples=0;self.updates=0

    def train_batch(self,model,opt,x,y):
        opt.zero_grad(set_to_none=True)
        loss=(model(x)-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();opt.step()
        self.forward_examples+=len(x);self.training_examples+=len(x);self.updates+=1

    def parameters_bytes(self):
        params=[p for m in self.models for p in m.parameters()]
        return sum(p.numel()*p.element_size() for p in params)

    def costs(self):
        state_bytes=0
        for opt in self.opts:
            for state in opt.state.values():
                state_bytes+=sum(v.numel()*v.element_size() for v in state.values() if isinstance(v,torch.Tensor))
        return dict(parameter_count=sum(p.numel() for m in self.models for p in m.parameters()),
            stored_tensor_bytes=self.parameters_bytes()+state_bytes+self.extra_bytes(),
            forward_examples=self.forward_examples,training_examples=self.training_examples,
            optimizer_updates=self.updates,module_count=len(self.models),
            module_switches=getattr(self,'switches',0))

    def extra_bytes(self):return 0


class SingleLearner(Base):
    def __init__(self,cfg,device,mode):
        super().__init__(cfg,device);self.mode=mode
        d,o=cfg['input_dim'],cfg['output_dim']
        self.moment=torch.zeros(d+1,o,device=device)
        self.context_dim=(d+1)*o if mode=='context_replay' else 0
        self.model=net(d+self.context_dim,cfg['learner_hidden'],o,device)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=cfg['learning_rate'])
        self.models=[self.model];self.opts=[self.opt]
        self.replay=None if mode=='online' else Replay(cfg['replay_capacity'],d+self.context_dim,o,device)

    def features(self,x):
        if not self.context_dim:return x
        return torch.cat([x,self.moment.flatten().expand(len(x),-1)],dim=1)

    @torch.no_grad()
    def predict(self,x,count=True):
        if count:self.forward_examples+=len(x)
        return self.model(self.features(x))

    def observe(self,x,y):
        features=self.features(x).detach()
        train_x,train_y=features,y
        if self.replay is not None and self.replay.size:
            past_x,past_y=self.replay.sample(len(x))
            train_x=torch.cat([features,past_x]);train_y=torch.cat([y,past_y])
        self.train_batch(self.model,self.opt,train_x,train_y)
        if self.replay is not None:self.replay.add(features,y)
        if self.context_dim:
            with torch.no_grad():
                augmented=torch.cat([x,torch.ones(len(x),1,device=self.device)],dim=1)
                moment=augmented.T@y/len(x)
                self.moment.lerp_(moment,1-self.cfg['moment_decay'])

    def extra_bytes(self):
        return self.moment.numel()*self.moment.element_size()+(self.replay.nbytes() if self.replay else 0)


class ModuleBank(Base):
    def __init__(self,cfg,device):
        super().__init__(cfg,device)
        self.models=[];self.opts=[];self.ages=[];self.active=0;self.switches=0
        self.novelty_count=0;self.capacity_hits=0;self.events=[]
        self.add_model()

    def add_model(self):
        c=self.cfg;m=net(c['input_dim'],c['learner_hidden'],c['output_dim'],self.device)
        self.models.append(m);self.opts.append(torch.optim.Adam(m.parameters(),lr=c['learning_rate']))
        self.ages.append(0);return len(self.models)-1

    @torch.no_grad()
    def predict(self,x,count=True):
        if count:self.forward_examples+=len(x)
        return self.models[self.active](x)

    def observe(self,x,y):
        # These are observed outcomes, available only after the scored prediction.
        with torch.no_grad():
            losses=torch.stack([(m(x)-y).square().mean() for m in self.models])
            self.forward_examples+=len(self.models)*len(x)
            best=int(losses.argmin().item())
            old=self.active
            if losses[best]<self.cfg['switch_ratio']*losses[self.active]:
                self.active=best
            novel=bool(losses.min()>self.cfg['novelty_mse_threshold'])
            self.novelty_count=self.novelty_count+1 if novel else 0
        if self.novelty_count>=self.cfg['novelty_patience'] and self.ages[self.active]>=self.cfg['module_min_age']:
            if len(self.models)<self.cfg['module_cap']:
                self.active=self.add_model()
                self.events.append(dict(update=self.updates,kind='create',module=self.active))
            else:self.capacity_hits+=1
            self.novelty_count=0
        if self.active!=old:self.switches+=1
        self.train_batch(self.models[self.active],self.opts[self.active],x,y)
        self.ages[self.active]+=1


class OracleModules(ModuleBank):
    def __init__(self,cfg,device):
        super().__init__(cfg,device)
        while len(self.models)<cfg['teacher_contexts']:self.add_model()

    def set_context(self,context):
        if self.active!=context:self.switches+=1
        self.active=context

    def observe(self,x,y):
        self.train_batch(self.models[self.active],self.opts[self.active],x,y)
        self.ages[self.active]+=1


def make_learner(arm,cfg,device):
    if arm=='module_bank':return ModuleBank(cfg,device)
    if arm=='oracle_modules':return OracleModules(cfg,device)
    return SingleLearner(cfg,device,arm)


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def run_arm(cfg,seed,arm,device):
    torch.manual_seed(seed);world=TeacherWorld(cfg,seed)
    learner=make_learner(arm,cfg,device)
    generator=torch.Generator().manual_seed(seed+200000)
    segments=[];seen=set();total_seconds=0.
    for segment,context in enumerate(cfg['context_order']):
        if isinstance(learner,OracleModules):learner.set_context(context)
        errors=[];clean_errors=[];sync(device);start=time.perf_counter()
        for batch in range(cfg['batches_per_context']):
            x,y,clean=world.sample(context,cfg['batch_size'],generator,device)
            prediction=learner.predict(x)
            errors.append(float((prediction-y).square().mean().item()))
            clean_errors.append(float((prediction-clean).square().mean().item()))
            learner.observe(x,y)
        sync(device);elapsed=time.perf_counter()-start;total_seconds+=elapsed
        # Independent diagnostic input/outcome stream; never observed by learner.
        test_g=torch.Generator().manual_seed(seed+300000+segment)
        tx,_,ty=world.sample(context,1024,test_g,device)
        heldout=float((learner.predict(tx,count=False)-ty).square().mean().item())
        mean=lambda a:sum(a)/len(a)
        row=dict(segment=segment,context=context,returning=context in seen,
            prequential_mse=mean(errors),clean_prequential_mse=mean(clean_errors),
            first_32_batch_mse=mean(errors[:32]),last_32_batch_mse=mean(errors[-32:]),
            heldout_current_mse=heldout,wall_seconds=elapsed,**learner.costs())
        segments.append(row);seen.add(context)
    return dict(seed=seed,arm=arm,segments=segments,wall_seconds=total_seconds,
                costs=learner.costs(),events=getattr(learner,'events',[]),
                capacity_hits=getattr(learner,'capacity_hits',0))


def preflight(cfg,device):
    # Tests interface separation, actual parameter updates, and disjoint sample RNG.
    torch.manual_seed(9);learner=SingleLearner(cfg,device,'context_replay')
    x=torch.randn(32,cfg['input_dim'],device=device);y=torch.randn(32,cfg['output_dim'],device=device)
    before=learner.predict(x).clone()
    assert torch.equal(before,learner.predict(x))
    learner.observe(x,y)
    assert not torch.equal(before,learner.predict(x))
    assert not hasattr(learner,'set_context')
    assert not hasattr(ModuleBank(cfg,device),'set_context')
    world=TeacherWorld(cfg,9)
    a=world.sample(0,32,torch.Generator().manual_seed(1),device)[0]
    b=world.sample(0,32,torch.Generator().manual_seed(2),device)[0]
    assert not torch.equal(a,b)
    return dict(causal_interface=True,gradient_updates=True,task_id_interface_isolated=True,
                independent_diagnostic_rng=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--out',type=Path,default=HERE/'runs/e1')
    args=parser.parse_args();torch.set_num_threads(1);device=torch.device(args.device)
    cfg=json.loads((HERE/'protocol_e1.json').read_text())
    checks=preflight(cfg,device);print(json.dumps(dict(preflight=checks)),flush=True)
    if not args.execute:return
    args.out.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256((HERE/'protocol_e1.json').read_bytes()).hexdigest()
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            path=args.out/f'{seed}-{arm}.json'
            if path.exists():
                r=json.loads(path.read_text())
                if r['protocol_sha256']!=protocol_hash or r['script_sha256']!=script_hash:
                    raise RuntimeError('Existing output identity mismatch; use a new experiment directory')
            else:
                r=run_arm(cfg,seed,arm,device)
                r.update(protocol_sha256=protocol_hash,script_sha256=script_hash,torch_version=torch.__version__,device=str(device))
                path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            print(json.dumps(dict(seed=seed,arm=arm,mean_mse=sum(s['prequential_mse'] for s in r['segments'])/len(r['segments']),
                                  returning_mse=sum(s['first_32_batch_mse'] for s in r['segments'] if s['returning'])/4,
                                  parameters=r['costs']['parameter_count'],modules=r['costs']['module_count'],seconds=r['wall_seconds'])),flush=True)
    summary=dict(protocol=cfg,protocol_sha256=protocol_hash,script_sha256=script_hash,preflight=checks,records=records)
    (args.out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('E1_COMPLETE '+str(args.out/'complete.json'),flush=True)


if __name__=='__main__':main()
