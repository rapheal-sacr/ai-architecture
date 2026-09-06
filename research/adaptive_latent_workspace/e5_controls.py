from pathlib import Path
import argparse,hashlib,json,copy
import torch
import e1_stream as e1
from e2_isolation import IsolatedAdaptation

HERE=Path(__file__).resolve().parent

class OverwriteDuringIdentification(IsolatedAdaptation):
    def observe(self,x,y):
        previous=len(self.events)
        super().observe(x,y)
        for event in self.events[previous:]:
            if event['kind']=='temporary':self.temporary_source=event['source']
        if self.fast_model is not None:
            self.models[self.temporary_source].load_state_dict(self.fast_model.state_dict())
            self.opts[self.temporary_source].load_state_dict(copy.deepcopy(self.fast_opt.state_dict()))

class ActiveFirstIsolation(IsolatedAdaptation):
    def observe(self,x,y):
        if self.fast_model is None:
            with torch.no_grad():
                loss=float((self.models[self.active](x)-y).square().mean())
                self.forward_examples+=len(x)
            threshold=max(self.cfg['reuse_error_floor'],self.cfg['reuse_error_ratio']*self.expected_error[self.active])
            if loss<threshold:
                self.train_batch(self.models[self.active],self.opts[self.active],x,y)
                self.ages[self.active]+=1
                decay=self.cfg['error_ema_decay']
                self.expected_error[self.active]=decay*self.expected_error[self.active]+(1-decay)*loss
                c=self.costs();self.peak_bytes=max(self.peak_bytes,c['stored_tensor_bytes'])
                self.peak_parameters=max(self.peak_parameters,c['parameter_count'])
                return
        super().observe(x,y)

def factory(arm,cfg,device):
    classes={'isolated_adaptation':IsolatedAdaptation,'overwrite_during_identification':OverwriteDuringIdentification,
             'active_first_isolation':ActiveFirstIsolation}
    if arm in classes:return classes[arm](cfg,device)
    if arm in ('wide_context_replay','compact_wide_replay'):
        cfg=cfg.copy();cfg['learner_hidden']=cfg['wide_hidden']
        if arm=='compact_wide_replay':cfg['replay_capacity']=cfg['compact_replay_capacity']
        return e1.SingleLearner(cfg,device,'context_replay')
    return e1.SingleLearner(cfg,device,arm)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);device=torch.device('cpu')
    spec=json.loads((HERE/'protocol_e5.json').read_text())
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(spec)
    cfg['context_order']*=spec['cycles_of_e1_order']
    # Gradient/cost preflight uses a separate, non-evaluation teacher seed.
    for name in spec['arms']:
        torch.manual_seed(21);learner=factory(name,cfg,device)
        x=torch.randn(32,8);y=torch.randn(32,4);before=learner.predict(x).clone()
        learner.observe(x,y);assert not torch.equal(before,learner.predict(x))
        assert not hasattr(learner,'set_context')
    print('E5 preflight passed',flush=True)
    if not a.execute:return
    e1.make_learner=factory
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
        ('protocol_e1.json','protocol_e2.json','protocol_e5.json','e1_stream.py','e2_isolation.py','e5_controls.py')}
    a.out.mkdir(parents=True,exist_ok=True);records=[]
    for seed in spec['seeds']:
        for arm in spec['arms']:
            path=a.out/f'{seed}-{arm}.json'
            if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
            else:
                r=e1.run_arm(cfg,seed,arm,device);r['identity']=identity
                path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            print(json.dumps(dict(seed=seed,arm=arm,mse=sum(s['prequential_mse'] for s in r['segments'])/len(r['segments']),
                seconds=r['wall_seconds'],modules=r['costs']['module_count'])),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=cfg,identity=identity,records=records),indent=2)+'\n')
    print('E5_COMPLETE',flush=True)

if __name__=='__main__':main()
