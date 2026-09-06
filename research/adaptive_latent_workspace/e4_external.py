from pathlib import Path
import argparse,hashlib,json,pickle,subprocess,sys,time
import torch
import e1_stream as e1
from e2_isolation import IsolatedAdaptation

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e4.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
    assert head==spec['source_commit']
    sys.path.insert(0,str(a.repo))
    from lop.slowly_changing_regression.slowly_changing_regression import generate_problem_data
    if not a.execute:return
    a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()))
    cfg.update(input_dim=20,output_dim=1,batch_size=spec['batch_size'])
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
        ('protocol_e1.json','protocol_e2.json','protocol_e4.json','e1_stream.py','e2_isolation.py','e4_external.py')}
    identity['external_commit']=head;records=[]
    for seed in spec['seeds']:
        datafile=a.out/f'generated-{seed}.pickle'
        if not datafile.exists():
            torch.manual_seed(seed)
            generate_problem_data(data_file=str(datafile),num_data_points=spec['observations'],**spec['generator_parameters'])
        # This pickle is generated locally by the pinned source immediately above.
        with datafile.open('rb') as f:x,y,_=pickle.load(f)
        x=x[:spec['observations']];y=y[:spec['observations']]
        stream_hash=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
        assert len(x)==spec['observations'] and torch.isfinite(y).all()
        for arm in spec['arms']:
            path=a.out/f'{seed}-{arm}.json'
            if path.exists():
                r=json.loads(path.read_text());assert r['identity']==identity and r['stream_sha256']==stream_hash
            else:
                torch.manual_seed(seed+100000)
                learner=IsolatedAdaptation(cfg,torch.device('cpu')) if arm=='isolated_adaptation' else e1.make_learner(arm,cfg,torch.device('cpu'))
                block=[];errors=[];started=time.perf_counter();weighted=0.;n=0
                for i in range(0,len(x),spec['batch_size']):
                    bx=x[i:i+spec['batch_size']];by=y[i:i+spec['batch_size']]
                    loss=float((learner.predict(bx)-by).square().mean())
                    assert torch.isfinite(torch.tensor(loss))
                    weighted+=loss*len(bx);n+=len(bx)
                    learner.observe(bx,by)
                    # Some batches straddle 10k boundaries. Store exact sample extent.
                    if n>=10000 or i+len(bx)==len(x):
                        block.append(dict(end_observation=i+len(bx),observations=n,mse=weighted/n,**learner.costs()))
                        errors.append((weighted,n));weighted=0.;n=0
                r=dict(seed=seed,arm=arm,identity=identity,stream_sha256=stream_hash,blocks=block,
                    mse=sum(e for e,n in errors)/sum(n for e,n in errors),wall_seconds=time.perf_counter()-started,
                    costs=learner.costs(),capacity_hits=getattr(learner,'capacity_hits',0),events=getattr(learner,'events',[]))
                path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            print(json.dumps({k:r[k] for k in ('seed','arm','mse','wall_seconds','capacity_hits')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E4_COMPLETE',flush=True)

if __name__=='__main__':main()
