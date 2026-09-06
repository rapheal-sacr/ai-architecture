from pathlib import Path
import argparse,copy,hashlib,json,time
import torch
from e17_persistent_student import PersistentStudent,make_stream
import e20_context_capacity as assay
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class PosteriorAssignment(PersistentStudent):
    def observe(self,x,y):
        extended=torch.cat([x,torch.ones(len(x),1)],dim=1)
        posterior=self.context.clone()
        posterior.mul_(self.cfg['context_ema']).add_((extended.T@y)/len(x),alpha=1-self.cfg['context_ema'])
        self.context=posterior.clone()
        super().observe(x,y)
        # The inherited context update is redundant work; do not apply it twice.
        self.context=posterior

def create(sc,base,seed,mode,width,lr):
    assert width==16 and mode in ('prior_assignment','posterior_assignment')
    cls=PersistentStudent if mode=='prior_assignment' else PosteriorAssignment
    return cls(copy.deepcopy(sc),base,seed,None,lr)

def preflight(spec,sc,base):
    checks=[]
    for decay in spec['context_decays']:
        c=copy.deepcopy(sc);c.update(context_ema=decay,updates=8,segment_min_batches=4,segment_max_batches=4)
        rows,_=make_stream(c,98,'conflicting_functions')
        original=PersistentStudent(c,base,99,None,.01);prior=create(c,base,99,'prior_assignment',16,.01);post=create(c,base,99,'posterior_assignment',16,.01)
        assert not hasattr(prior,'set_regime') and not hasattr(post,'set_regime')
        assert torch.equal(prior.predict(rows[0][0]),post.predict(rows[0][0]))
        # Balance the separately charged initial prediction check for exact counters.
        original.predict(rows[0][0])
        for x,y,_ in rows:
            assert torch.equal(original.predict(x),prior.predict(x));post.predict(x)
            original.observe(x,y);prior.observe(x,y);post.observe(x,y)
            assert all(torch.equal(a,b) for a,b in zip(original.params,prior.params))
            assert torch.equal(prior.context,post.context)
        assert prior.counts==post.counts and prior.costs()==post.costs()
        # Before reservoir capacity, the final inserted block has its exact posterior.
        assert torch.equal(post.replay_x[post.size-32:post.size,8:],post.context.reshape(1,-1).expand(32,-1))
        assert not torch.equal(prior.replay_x[:prior.size,8:],post.replay_x[:post.size,8:])
        checks.append(dict(decay=decay,prior_exact=True,context_trajectory_exact=True,posterior_assignment_verified=True,costs=post.costs()))
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e21.json').read_text());sc=json.loads((HERE/'protocol_e17.json').read_text())
    sc['updates']=spec['updates'];base=json.loads((HERE/'protocol_e9.json').read_text())
    names=('protocol_e21.json','e21_evidence_assignment.py','e20_context_capacity.py','protocol_e17.json','e17_persistent_student.py','e18_online_procedure.py','protocol_e9.json','e9_recursive_updater.py')
    identity={n:digest(HERE/n) for n in names};a.out.mkdir(parents=True,exist_ok=True)
    checks=preflight(spec,sc,base);(a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks),indent=2)+'\n')
    print('E21 evidence timing preflight passed',flush=True)
    if not a.execute:return
    assay.create=create;records=[];worlds=[]
    for seed in spec['stream_seeds']:
        for family in spec['families']:
            start=time.perf_counter();rows,segments=make_stream(sc,seed,family);worlds.append(dict(seed=seed,family=family,generation_seconds=time.perf_counter()-start))
            for decay in spec['context_decays']:
                c=copy.deepcopy(sc);c['context_ema']=decay;directory=a.out/f'ema-{decay}';directory.mkdir(parents=True,exist_ok=True)
                for mode in spec['assignment_modes']:
                    stem=f'{seed}-{family}-{mode}-width16';path=directory/f'{stem}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                    else:
                        assert not (directory/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before restarting'
                        r=assay.run(spec,c,base,seed,family,mode,16,rows,segments,directory,identity);r['context_decay']=decay
                        path.write_text(json.dumps(r,indent=2)+'\n')
                    records.append(r);print(json.dumps({k:r.get(k) for k in ('seed','family','context_mode','context_decay','clean_mse','recovered_segments','total_segments','failed')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,worlds=worlds,records=records),indent=2)+'\n');print('E21_COMPLETE',flush=True)

if __name__=='__main__':main()
