from pathlib import Path
import argparse,copy,hashlib,json,pickle,subprocess,sys,time
import torch
import e1_stream as e1
from e5_controls import ActiveFirstIsolation

HERE=Path(__file__).resolve().parent

class GuardedSharing(ActiveFirstIsolation):
    def __init__(self,cfg,device,merge=True):
        self.anchors=[];self.scratch=None;self.scratch_model=None;self.merges=0;self.merge_enabled=merge
        super().__init__(cfg,device)
        self.anchors=[self.new_buffer()]

    def new_buffer(self):
        return e1.Replay(self.cfg['anchor_capacity'],self.cfg['input_dim'],self.cfg['output_dim'],self.device)

    def extra_bytes(self):
        return sum(a.nbytes() for a in self.anchors)+(self.scratch.nbytes() if self.scratch is not None else 0)

    def train_batch(self,model,opt,x,y):
        if model is self.fast_model:
            if model is not self.scratch_model:
                self.scratch_model=model;self.scratch=self.new_buffer()
                self.source=next(e['source'] for e in reversed(self.events) if e['kind']=='temporary')
            anchor=self.anchors[self.source]
            target_buffer=self.scratch
        else:
            index=next(i for i,m in enumerate(self.models) if m is model)
            anchor=self.anchors[index];target_buffer=anchor
        if anchor.size:
            ax,ay=anchor.sample(min(len(x),anchor.size))
            tx=torch.cat([x,ax]);ty=torch.cat([y,ay])
        else:tx,ty=x,y
        super().train_batch(model,opt,tx,ty)
        target_buffer.add(x,y)

    def observe(self,x,y):
        n=len(self.models)
        super().observe(x,y)
        grew=len(self.models)>n
        pending=(self.fast_model is not None and len(self.models)==self.cfg['module_cap'] and
            self.fast_age>=self.cfg['temporary_min_age'] and self.fast_age%32==0 and
            sum(self.fast_scores)<self.cfg['commit_improvement_ratio']*sum(self.old_scores))
        if grew or pending:
            source=self.source;anchor=self.anchors[source]
            candidate=self.models[-1] if grew else self.fast_model
            with torch.no_grad():
                old=float((self.models[source](anchor.x[:anchor.size])-anchor.y[:anchor.size]).square().mean())
                new=float((candidate(anchor.x[:anchor.size])-anchor.y[:anchor.size]).square().mean())
                self.forward_examples+=2*anchor.size
            tolerance=max(self.cfg['merge_absolute_tolerance'],self.cfg['merge_relative_tolerance']*old)
            merge=self.merge_enabled and anchor.size>0 and new<=old+tolerance
            self.events.append(dict(update=self.updates,kind='anchor_check',old_mse=old,new_mse=new,merged=merge))
            if merge:
                if grew:
                    self.models[source]=self.models.pop();self.opts[source]=self.opts.pop()
                    self.ages[source]=self.ages.pop();self.expected_error[source]=self.expected_error.pop()
                else:
                    self.models[source]=self.fast_model;self.opts[source]=self.fast_opt
                    self.ages[source]=self.fast_age;self.expected_error[source]=sum(self.fast_scores)/len(self.fast_scores)
                    self.fast_model=None;self.fast_opt=None;self.fast_age=0
                self.active=source;self.merges+=1
                anchor.add(self.scratch.x[:self.scratch.size],self.scratch.y[:self.scratch.size])
            elif grew:self.anchors.append(self.scratch)
            if grew or merge:self.scratch=None;self.scratch_model=None
        elif self.fast_model is None:
            self.scratch=None;self.scratch_model=None
        c=self.costs();self.peak_bytes=max(self.peak_bytes,c['stored_tensor_bytes'])
        self.peak_parameters=max(self.peak_parameters,c['parameter_count'])

    def costs(self):
        c=super().costs();c['merges']=self.merges
        return c

def factory(arm,cfg,device):
    if arm=='guarded_sharing':return GuardedSharing(cfg,device,True)
    if arm=='local_replay_isolation':return GuardedSharing(cfg,device,False)
    if arm=='active_first_isolation':return ActiveFirstIsolation(cfg,device)
    return e1.SingleLearner(cfg,device,arm)

def external_run(cfg,spec,seed,arm,x,y):
    torch.manual_seed(seed+100000);learner=factory(arm,cfg,torch.device('cpu'))
    blocks=[];weighted=0.;n=0;start=time.perf_counter()
    for i in range(0,len(x),32):
        bx=x[i:i+32];by=y[i:i+32]
        loss=float((learner.predict(bx)-by).square().mean());assert torch.isfinite(torch.tensor(loss))
        weighted+=loss*len(bx);n+=len(bx);learner.observe(bx,by)
        if n>=10000 or i+len(bx)==len(x):
            blocks.append(dict(end_observation=i+len(bx),observations=n,mse=weighted/n,**learner.costs()))
            weighted=0.;n=0
    return dict(seed=seed,arm=arm,blocks=blocks,mse=sum(b['mse']*b['observations'] for b in blocks)/len(x),
        wall_seconds=time.perf_counter()-start,costs=learner.costs(),events=getattr(learner,'events',[]),
        capacity_hits=getattr(learner,'capacity_hits',0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e6.json').read_text());ext=json.loads((HERE/'protocol_e4.json').read_text())
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(spec)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip();assert head==ext['source_commit']
    sys.path.insert(0,str(a.repo))
    from lop.slowly_changing_regression.slowly_changing_regression import generate_problem_data
    # Exercise creation, replay and merge branches on train-only dummy data.
    torch.manual_seed(31);check_cfg=cfg.copy();check_cfg.update(module_cap=1,merge_absolute_tolerance=100.)
    check=GuardedSharing(check_cfg,torch.device('cpu'))
    for i in range(180):
        bx=torch.randn(32,8);by=torch.ones(32,4)*(0 if i<100 else 3)
        check.predict(bx);check.observe(bx,by)
    assert check.updates==180 and check.extra_bytes()>0 and not hasattr(check,'set_context')
    assert check.merges>0 and len(check.models)==1
    print('E6 preflight passed',flush=True)
    if not a.execute:return
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
        ('protocol_e1.json','protocol_e2.json','protocol_e4.json','protocol_e6.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py')}
    identity['external_commit']=head;a.out.mkdir(parents=True,exist_ok=True);e1.make_learner=factory;records=[]
    for case in spec['cases']:
        c=cfg.copy();c['context_order']=c['context_order']*spec['abrupt_cycles']
        if case=='external':c.update(input_dim=20,output_dim=1)
        for seed in spec['seeds']:
            if case=='external':
                path=a.out/f'generated-{seed}.pickle'
                if not path.exists():
                    torch.manual_seed(seed);generate_problem_data(data_file=str(path),num_data_points=spec['external_observations'],**ext['generator_parameters'])
                with path.open('rb') as f:x,y,_=pickle.load(f)
                x=x[:spec['external_observations']];y=y[:spec['external_observations']]
                stream_hash=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
            for arm in spec['arms']:
                path=a.out/f'{case}-{seed}-{arm}.json'
                if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                else:
                    r=external_run(c,spec,seed,arm,x,y) if case=='external' else e1.run_arm(c,seed,arm,torch.device('cpu'))
                    if case=='external':r['stream_sha256']=stream_hash
                    else:r['mse']=sum(s['prequential_mse'] for s in r['segments'])/len(r['segments'])
                    r.update(case=case,identity=identity);path.write_text(json.dumps(r,indent=2)+'\n')
                records.append(r)
                print(json.dumps(dict(case=case,seed=seed,arm=arm,mse=r['mse'],seconds=r['wall_seconds'],costs=r['costs'])),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E6_COMPLETE',flush=True)

if __name__=='__main__':main()
