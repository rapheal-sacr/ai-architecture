from pathlib import Path
import argparse,hashlib,json,pickle,sys,subprocess,time
import torch
import e1_stream as e1
from e6_sharing import factory
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stream_hash(x,y):
    h=hashlib.sha256();h.update(x.numpy().tobytes());h.update(y.numpy().tobytes());return h.hexdigest()

def extend(x,y,teacher,spec,seed,params):
    prefix=spec['prefix_observations'];total=spec['total_observations'];flip=params['flip_after'];slow=params['num_flipping_bits']
    assert prefix%flip==0 and total%flip==0
    x=x[:prefix];y=y[:prefix];state=x[-1,:slow].clone();g=torch.Generator().manual_seed(seed+spec['extension_seed_offset'])
    future=torch.empty(total-prefix,params['num_inputs']);targets=torch.empty(total-prefix,1)
    with torch.no_grad():
        for i in range(0,len(future),flip):
            bit=int(torch.randint(slow,(1,),generator=g));state[bit]=1-state[bit]
            future[i:i+flip,:slow]=state
            future[i:i+flip,slow:]=torch.randint(2,(flip,params['num_inputs']-slow),generator=g,dtype=torch.float32)
            targets[i:i+flip],_=teacher.predict(x=future[i:i+flip])
    return torch.cat([x,future]),torch.cat([y,targets])

def run(cfg,spec,seed,arm,x,y,prior,out):
    torch.manual_seed(seed+100000);learner=factory(arm,cfg,torch.device('cpu'));blocks=[];total_loss=0.;weighted=0.;n=0
    start=time.perf_counter();prefix_check=None
    log_path=out/f'{seed}-{arm}-blocks.jsonl';assert not log_path.exists(),'Partial log exists; inspect before restarting'
    for i in range(0,len(x),32):
        bx=x[i:i+32];by=y[i:i+32];loss=float((learner.predict(bx)-by).square().mean())
        assert torch.isfinite(torch.tensor(loss));total_loss+=loss*len(bx);weighted+=loss*len(bx);n+=len(bx);learner.observe(bx,by)
        end=i+len(bx)
        if n>=10000 or end in (spec['prefix_observations'],len(x)):
            row=dict(end_observation=end,observations=n,mse=weighted/n,elapsed_seconds=time.perf_counter()-start,**learner.costs())
            blocks.append(row)
            with log_path.open('a') as f:f.write(json.dumps(row)+'\n')
            weighted=0.;n=0
        if end==spec['prefix_observations']:
            # E6 computes its prefix mean from weighted block means, so use the
            # identical summation convention for the exact restoration check.
            mse=sum(b['mse']*b['observations'] for b in blocks)/end
            prefix_check=dict(mse_equal=mse==prior['mse'],costs_equal=learner.costs()==prior['costs'],mse=mse,prior_mse=prior['mse'])
            assert prefix_check['mse_equal'] and prefix_check['costs_equal'],prefix_check
            print(json.dumps(dict(seed=seed,arm=arm,prefix_check=prefix_check)),flush=True)
        if end%1000000==0:print(json.dumps(dict(seed=seed,arm=arm,observations=end,cumulative_mse=total_loss/end,seconds=time.perf_counter()-start)),flush=True)
    return dict(seed=seed,arm=arm,blocks=blocks,prefix_check=prefix_check,mse=sum(b['mse']*b['observations'] for b in blocks)/len(x),
        wall_seconds=time.perf_counter()-start,costs=learner.costs(),events=getattr(learner,'events',[]),capacity_hits=getattr(learner,'capacity_hits',0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e13.json').read_text());oldspec=json.loads((HERE/'protocol_e6.json').read_text());ext=json.loads((HERE/'protocol_e4.json').read_text())
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(oldspec);cfg.update(input_dim=20,output_dim=1)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()==ext['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.repo,text=True).strip()
    sys.path.insert(0,str(a.repo));a.out.mkdir(parents=True,exist_ok=True)
    names=['protocol_e1.json','protocol_e2.json','protocol_e4.json','protocol_e6.json','protocol_e13.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py','e13_amortization.py']
    identity={n:digest(HERE/n) for n in names};identity['external_commit']=ext['source_commit'];records=[]
    for seed in spec['seeds'] if a.execute else spec['seeds'][:1]:
        source=a.runs/'e6'/f'generated-{seed}.pickle'
        with source.open('rb') as f:x,y,teacher=pickle.load(f)
        old={arm:json.loads((a.runs/'e6'/f'external-{seed}-{arm}.json').read_text()) for arm in spec['arms']}
        for record in old.values():
            assert all(identity[k]==v for k,v in record['identity'].items())
            assert stream_hash(x[:spec['prefix_observations']],y[:spec['prefix_observations']])==record['stream_sha256']
        trial=spec.copy()
        if not a.execute:trial['total_observations']=trial['prefix_observations']+20000
        x,y=extend(x,y,teacher,trial,seed,ext['generator_parameters'])
        first=spec['prefix_observations'];slow=ext['generator_parameters']['num_flipping_bits'];flip=ext['generator_parameters']['flip_after']
        assert (x[first,:slow]!=x[first-1,:slow]).sum()==1
        assert torch.equal(x[first,:slow],x[first+flip-1,:slow])
        assert (x[first+flip,:slow]!=x[first,:slow]).sum()==1
        if not a.execute:
            print('E13 exact-prefix/data-continuation preflight passed',flush=True);return
        sha=stream_hash(x,y)
        for arm in spec['arms']:
            path=a.out/f'{seed}-{arm}.json'
            if path.exists():r=json.loads(path.read_text());assert r['identity']==identity and r['stream_sha256']==sha
            else:
                r=run(cfg,spec,seed,arm,x,y,old[arm],a.out);r.update(identity=identity,stream_sha256=sha,prefix_source_sha256=digest(source))
                path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E13_COMPLETE',flush=True)
if __name__=='__main__':main()
