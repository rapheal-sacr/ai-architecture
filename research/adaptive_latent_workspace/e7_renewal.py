from pathlib import Path
import argparse,hashlib,json,pickle,subprocess,sys
import torch
import e1_stream as e1
import e6_sharing as harness

HERE=Path(__file__).resolve().parent

class RenewalLearner(e1.SingleLearner):
    def __init__(self,cfg,device,replay,renew):
        self.gnt=None;self.replacements=0;self.saturation=0.;self.renew=renew
        super().__init__(cfg,device,'replay' if replay else 'online')
        from lop.utils.AdamGnT import AdamGnT
        from lop.algos.gnt import GnT
        self.opt=AdamGnT(self.model.parameters(),lr=cfg['learning_rate']);self.opts=[self.opt]
        if renew:
            self.gnt=GnT(self.model,hidden_activation='tanh',opt=self.opt,
                replacement_rate=cfg['replacement_rate'],decay_rate=cfg['decay_rate'],
                maturity_threshold=cfg['maturity_threshold'],util_type=cfg['utility_type'],accumulate=False)

    def train_batch(self,model,opt,x,y):
        opt.zero_grad(set_to_none=True)
        h1=model[1](model[0](x));h2=model[3](model[2](h1));out=model[4](h2)
        loss=(out-y).square().mean();assert torch.isfinite(loss)
        loss.backward();opt.step()
        self.saturation=float(torch.cat([h1.detach(),h2.detach()],dim=1).abs().gt(.9).float().mean())
        if self.gnt is not None:
            opt.zero_grad(set_to_none=True);self.gnt.gen_and_test([h1.detach(),h2.detach()])
            self.replacements+=sum(int((age==0).sum()) for age in self.gnt.ages)
        self.forward_examples+=len(x);self.training_examples+=len(x);self.updates+=1

    def extra_bytes(self):
        n=super().extra_bytes()
        if self.gnt is not None:
            for name in ('util','bias_corrected_util','ages','mean_feature_act'):
                n+=sum(v.numel()*v.element_size() for v in getattr(self.gnt,name))
        return n

    def costs(self):
        c=super().costs();c.update(feature_replacements=self.replacements,last_training_saturation=self.saturation)
        return c

def factory(arm,cfg,device):
    if arm=='context_replay':return e1.SingleLearner(cfg,device,arm)
    return RenewalLearner(cfg,device,replay=arm.startswith('replay'),renew='cbp' in arm)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e7.json').read_text());ext=json.loads((HERE/'protocol_e4.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip();assert head==spec['source_commit']
    sys.path.insert(0,str(a.repo))
    from lop.slowly_changing_regression.slowly_changing_regression import generate_problem_data
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(spec);cfg.update(input_dim=20,output_dim=1)
    # Preflight deliberately raises replacement rate/maturity speed only for dummy data.
    c=cfg.copy();c.update(replacement_rate=.1,maturity_threshold=2)
    torch.manual_seed(44);check=factory('replay_cbp_gnt',c,torch.device('cpu'))
    for i in range(20):
        x=torch.randn(32,20);y=torch.randn(32,1);check.predict(x);check.observe(x,y)
    assert check.replacements>0 and check.updates==20 and torch.isfinite(check.predict(x)).all()
    assert check.costs()['stored_tensor_bytes']>check.parameters_bytes()
    print('E7 renewal/optimizer/replay preflight passed',flush=True)
    if not a.execute:return
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
        ('protocol_e1.json','protocol_e4.json','protocol_e7.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py','e7_renewal.py')}
    identity['external_commit']=head;a.out.mkdir(parents=True,exist_ok=True);harness.factory=factory;records=[]
    for seed in spec['seeds']:
        path=a.out/f'generated-{seed}.pickle'
        if not path.exists():
            torch.manual_seed(seed);generate_problem_data(data_file=str(path),num_data_points=spec['observations'],**ext['generator_parameters'])
        with path.open('rb') as f:x,y,_=pickle.load(f)
        x=x[:spec['observations']];y=y[:spec['observations']]
        stream_hash=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
        for arm in spec['arms']:
            path=a.out/f'{seed}-{arm}.json'
            if path.exists():r=json.loads(path.read_text());assert r['identity']==identity and r['stream_sha256']==stream_hash
            else:
                r=harness.external_run(cfg,spec,seed,arm,x,y);r.update(identity=identity,stream_sha256=stream_hash)
                path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            print(json.dumps(dict(seed=seed,arm=arm,mse=r['mse'],seconds=r['wall_seconds'],costs=r['costs'])),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E7_COMPLETE',flush=True)

if __name__=='__main__':main()
