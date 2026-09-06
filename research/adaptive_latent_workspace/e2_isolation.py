"""Temporary adaptation, protected persistent modules, inferred context.

Uses E1's task and baselines without editing its frozen implementation.
"""
from __future__ import annotations
from collections import deque
import argparse,copy,hashlib,json
from pathlib import Path
import torch
import e1_stream as e1

HERE=Path(__file__).resolve().parent


class IsolatedAdaptation(e1.ModuleBank):
    def __init__(self,cfg,device):
        super().__init__(cfg,device)
        self.expected_error=[.1];self.fast_model=None;self.fast_opt=None;self.fast_age=0
        self.fast_scores=deque(maxlen=cfg['recent_score_window'])
        self.old_scores=deque(maxlen=cfg['recent_score_window'])
        self.peak_bytes=0;self.peak_parameters=0

    @torch.no_grad()
    def predict(self,x,count=True):
        if count:self.forward_examples+=len(x)
        m=self.fast_model if self.fast_model is not None else self.models[self.active]
        return m(x)

    def observe(self,x,y):
        old=self.active
        with torch.no_grad():
            losses=torch.stack([(m(x)-y).square().mean() for m in self.models])
            self.forward_examples+=len(x)*len(self.models)
            thresholds=torch.tensor([max(self.cfg['reuse_error_floor'],self.cfg['reuse_error_ratio']*v)
                                      for v in self.expected_error],device=self.device)
            # Immature models do not gain broad applicability from high startup loss.
            acceptable=losses<thresholds
            for i,age in enumerate(self.ages):
                if age<self.cfg['module_min_age'] and i!=self.active:acceptable[i]=False
            ids=torch.nonzero(acceptable,as_tuple=False).flatten()
            best=int(losses.argmin())
        if len(ids):
            self.active=int(ids[losses[ids].argmin()])
            self.fast_model=None;self.fast_opt=None;self.fast_age=0
            self.fast_scores.clear();self.old_scores.clear()
            self.train_batch(self.models[self.active],self.opts[self.active],x,y)
            self.ages[self.active]+=1
            decay=self.cfg['error_ema_decay']
            self.expected_error[self.active]=decay*self.expected_error[self.active]+(1-decay)*float(losses[self.active])
        elif self.fast_model is None and self.ages[self.active]<self.cfg['module_min_age']:
            self.train_batch(self.models[self.active],self.opts[self.active],x,y)
            self.ages[self.active]+=1
            decay=self.cfg['error_ema_decay']
            self.expected_error[self.active]=decay*self.expected_error[self.active]+(1-decay)*float(losses[self.active])
        else:
            if self.fast_model is None:
                self.fast_model=copy.deepcopy(self.models[best])
                self.fast_opt=torch.optim.Adam(self.fast_model.parameters(),lr=self.cfg['learning_rate'])
                self.fast_age=0;self.fast_scores.clear();self.old_scores.clear()
                self.events.append(dict(update=self.updates,kind='temporary',source=best))
            with torch.no_grad():
                fast_loss=float((self.fast_model(x)-y).square().mean())
                self.forward_examples+=len(x)
            self.fast_scores.append(fast_loss);self.old_scores.append(float(losses.min()))
            self.train_batch(self.fast_model,self.fast_opt,x,y);self.fast_age+=1
            commit=(self.fast_age>=self.cfg['temporary_min_age'] and
                    sum(self.fast_scores)<self.cfg['commit_improvement_ratio']*sum(self.old_scores))
            if commit:
                if len(self.models)<self.cfg['module_cap']:
                    self.models.append(self.fast_model);self.opts.append(self.fast_opt)
                    self.ages.append(self.fast_age);self.expected_error.append(sum(self.fast_scores)/len(self.fast_scores))
                    self.active=len(self.models)-1
                    self.events.append(dict(update=self.updates,kind='commit',module=self.active))
                    self.fast_model=None;self.fast_opt=None;self.fast_age=0
                else:self.capacity_hits+=1
        if self.active!=old:self.switches+=1
        c=self.costs();self.peak_bytes=max(self.peak_bytes,c['stored_tensor_bytes'])
        self.peak_parameters=max(self.peak_parameters,c['parameter_count'])

    def costs(self):
        c=super().costs()
        if self.fast_model is not None:
            c['parameter_count']+=sum(p.numel() for p in self.fast_model.parameters())
            c['stored_tensor_bytes']+=sum(p.numel()*p.element_size() for p in self.fast_model.parameters())
            for state in self.fast_opt.state.values():
                c['stored_tensor_bytes']+=sum(v.numel()*v.element_size() for v in state.values() if isinstance(v,torch.Tensor))
        c['peak_tensor_bytes']=max(self.peak_bytes,c['stored_tensor_bytes'])
        c['peak_parameter_count']=max(self.peak_parameters,c['parameter_count'])
        c['temporary_active']=self.fast_model is not None
        return c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--fresh',action='store_true');parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);device=torch.device('cpu')
    spec=json.loads((HERE/'protocol_e2.json').read_text())
    cfg=json.loads((HERE/spec['base_protocol']).read_text());cfg.update(spec)
    original_factory=e1.make_learner
    e1.make_learner=lambda arm,c,d: IsolatedAdaptation(c,d) if arm=='isolated_adaptation' else original_factory(arm,c,d)
    # Preflight gradient and API behavior, train-only dummy data.
    torch.manual_seed(1);learner=IsolatedAdaptation(cfg,device)
    x=torch.randn(32,8);y=torch.randn(32,4);before=learner.predict(x).clone()
    learner.observe(x,y)
    assert not torch.equal(before,learner.predict(x)) and not hasattr(learner,'set_context')
    print('E2 preflight passed',flush=True)
    if not args.execute:return
    seeds=spec['fresh_seeds'] if args.fresh else spec['development_seeds']
    identity={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
              for name in ('protocol_e1.json','protocol_e2.json','e1_stream.py','e2_isolation.py')}
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    for seed in seeds:
        for arm in spec['arms']:
            p=args.out/f'{seed}-{arm}.json'
            if p.exists():
                r=json.loads(p.read_text());assert r['identity']==identity
            else:
                r=e1.run_arm(cfg,seed,arm,device);r['identity']=identity
                r['split']='fresh' if args.fresh else 'development'
                p.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            print(json.dumps(dict(seed=seed,arm=arm,mean_mse=sum(s['prequential_mse'] for s in r['segments'])/8,
                returning_mse=sum(s['first_32_batch_mse'] for s in r['segments'] if s['returning'])/4,
                modules=r['costs']['module_count'],seconds=r['wall_seconds'])),flush=True)
    (args.out/'complete.json').write_text(json.dumps(dict(protocol=cfg,identity=identity,records=records),indent=2)+'\n')
    print('E2_COMPLETE '+str(args.out/'complete.json'),flush=True)


if __name__=='__main__':main()
