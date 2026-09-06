from pathlib import Path
import argparse,hashlib,json,math
import torch
import e1_stream as e1
from e2_isolation import IsolatedAdaptation

HERE=Path(__file__).resolve().parent
OriginalWorld=e1.TeacherWorld

class GradualWorld(OriginalWorld):
    def sample(self,context,n,generator,device):
        # Hidden smooth interpolation; no context or blend coefficient reaches learner.
        progress=context/8;low=int(progress)%len(self.w1);high=(low+1)%len(self.w1);fraction=progress%1
        x=torch.randn(n,self.d,generator=generator)
        a=torch.tanh(x@self.w1[low]+self.b1[low])@self.w2[low]
        b=torch.tanh(x@self.w1[high]+self.b1[high])@self.w2[high]
        clean=(1-fraction)*a+fraction*b
        y=clean+self.noise*torch.randn(n,self.o,generator=generator)
        return x.to(device),y.to(device),clean.to(device)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if not args.execute:raise SystemExit('Use --execute after inspecting the frozen protocol')
    torch.set_num_threads(1);device=torch.device('cpu')
    spec=json.loads((HERE/'protocol_e3.json').read_text())
    base=json.loads((HERE/'protocol_e1.json').read_text());base.update(json.loads((HERE/'protocol_e2.json').read_text()))
    original_factory=e1.make_learner
    e1.make_learner=lambda a,c,d:IsolatedAdaptation(c,d) if a=='isolated_adaptation' else original_factory(a,c,d)
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
              ('protocol_e1.json','protocol_e2.json','protocol_e3.json','e1_stream.py','e2_isolation.py','e3_stress.py')}
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    for case,settings in spec['cases'].items():
        cfg=base.copy();cfg['batches_per_context']=settings['batches_per_context']
        cfg['teacher_output_noise_std']=settings.get('noise_std',base['teacher_output_noise_std'])
        cfg['teacher_contexts']=settings.get('contexts',base['teacher_contexts'])
        cfg['context_order']=settings.get('context_order',base['context_order'])
        if case=='capacity':cfg['context_order']=list(range(settings['contexts']))*settings['cycles']
        if case=='long':cfg['context_order']=base['context_order']*settings['cycles_of_e1_order']
        if case=='gradual':cfg['context_order']=list(range(settings['segments']))
        e1.TeacherWorld=GradualWorld if case=='gradual' else OriginalWorld
        for seed in spec['seeds']:
            for arm in spec['arms']:
                path=args.out/f'{case}-{seed}-{arm}.json'
                if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                else:
                    r=e1.run_arm(cfg,seed,arm,device);r.update(case=case,identity=identity,configuration=cfg)
                    path.write_text(json.dumps(r,indent=2)+'\n')
                records.append(r)
                print(json.dumps(dict(case=case,seed=seed,arm=arm,
                    mse=sum(s['prequential_mse'] for s in r['segments'])/len(r['segments']),
                    seconds=r['wall_seconds'],modules=r['costs']['module_count'],capacity_hits=r['capacity_hits'])),flush=True)
    (args.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E3_COMPLETE '+str(args.out/'complete.json'),flush=True)

if __name__=='__main__':main()
