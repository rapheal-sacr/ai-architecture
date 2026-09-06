from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from e17_persistent_student import PersistentStudent,make_stream
from e18_online_procedure import segment_results
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class OracleContextDiagnostic(PersistentStudent):
    def set_regime(self,index):
        self.context.zero_();self.context.reshape(-1)[index]=1

def create(sc,base,seed,mode,width,lr):
    cfg=copy.deepcopy(sc);cfg['student']['hidden_dim']=width
    cls=OracleContextDiagnostic if mode=='oracle_onehot_diagnostic' else PersistentStudent
    return cls(cfg,base,seed,None,lr)

def preflight(spec,sc,base):
    c=copy.deepcopy(sc);c.update(updates=8,segment_min_batches=4,segment_max_batches=4)
    rows,segments=make_stream(c,96,'conflicting_functions')
    a=PersistentStudent(c,base,97,None,.01);b=create(c,base,97,'inferred_ema',16,.01)
    for x,y,_ in rows:
        assert torch.equal(a.predict(x),b.predict(x));a.observe(x,y);b.observe(x,y)
    assert all(torch.equal(x,y) for x,y in zip(a.params,b.params)) and not hasattr(b,'set_regime')
    checks=[]
    for width in spec['hidden_widths']:
        oracle=create(c,base,97,'oracle_onehot_diagnostic',width,.01)
        for i,(x,y,_) in enumerate(rows):
            context=next(s['context'] for s in segments if s['start']<=i<s['stop'])
            oracle.set_regime(context);code=oracle.attach(x)[:,8:]
            assert torch.equal(code.sum(1),torch.ones(len(x))) and code[:,context].eq(1).all()
            assert oracle.predict(x).shape==(32,2);oracle.observe(x,y)
        stored=oracle.replay_x[:oracle.size,8:]
        assert stored[:,:4].sum(1).eq(1).all() and stored[:,4:].eq(0).all()
        assert oracle.step==8 and oracle.size==256
        checks.append(dict(width=width,costs=oracle.costs(),replay_codes_onehot=True))
    return dict(inferred_width16_exact=True,inferred_has_no_regime_interface=True,checks=checks)

def run(spec,sc,base,seed,family,mode,width,rows,segments,out,identity):
    learner=create(sc,base,seed,mode,width,spec['learning_rate']);clean=[];noisy=[];normalized=[];failure=None;start=time.perf_counter();j=0
    for t,(x,y,truth) in enumerate(rows):
        if t>=segments[j]['stop']:j+=1
        if mode=='oracle_onehot_diagnostic':learner.set_regime(segments[j]['context'])
        prediction=learner.predict(x);cl=float((prediction-truth).square().mean());loss=float((prediction-y).square().mean())
        if not math.isfinite(loss):failure=f'Nonfinite prediction at {t}';break
        clean.append(cl);noisy.append(loss);normalized.append(cl/(float(truth.square().mean())+.01))
        try:learner.observe(x,y)
        except FloatingPointError as e:failure=str(e);break
    stem=f'{seed}-{family}-{mode}-width{width}'
    result=dict(seed=seed,family=family,context_mode=mode,hidden_width=width,identity=identity,failed=failure,
        seconds=time.perf_counter()-start,costs=learner.costs(),checkpoint=str(out/f'{stem}.pt'))
    if not failure:
        assert learner.step==spec['updates'] and learner.size==sc['replay_capacity']
        assert learner.counts['student_forward_examples']==spec['updates']*96-32
        segs=segment_results(segments,clean,normalized,sc)
        result.update(clean_mse=float(np.mean(clean)),observed_mse=float(np.mean(noisy)),clean_batch_mse=clean,
            quarter_clean_mse=[float(np.mean(q)) for q in np.array_split(clean,4)],normalized_batch_mse=normalized,segments=segs,
            recovered_segments=sum(s['recovery_batches'] is not None for s in segs),total_segments=len(segs))
    else:result['completed_batches']=len(clean)
    torch.save(dict(identity=identity,learner_state=learner.state()),out/f'{stem}.pt');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e20.json').read_text());sc=json.loads((HERE/'protocol_e17.json').read_text())
    sc['updates']=spec['updates'];base=json.loads((HERE/'protocol_e9.json').read_text())
    identity={n:digest(HERE/n) for n in ('protocol_e20.json','e20_context_capacity.py','protocol_e17.json','e17_persistent_student.py','protocol_e9.json','e9_recursive_updater.py','e18_online_procedure.py')}
    a.out.mkdir(parents=True,exist_ok=True);check=preflight(spec,sc,base)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=check),indent=2)+'\n');print('E20 context/capacity preflight passed',flush=True)
    if not a.execute:return
    records=[];worlds=[]
    for seed in spec['stream_seeds']:
        for family in spec['families']:
            start=time.perf_counter();rows,segments=make_stream(sc,seed,family)
            worlds.append(dict(seed=seed,family=family,generation_seconds=time.perf_counter()-start))
            for mode in spec['context_modes']:
                for width in spec['hidden_widths']:
                    stem=f'{seed}-{family}-{mode}-width{width}';path=a.out/f'{stem}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                    else:
                        assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before restarting'
                        r=run(spec,sc,base,seed,family,mode,width,rows,segments,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
                    records.append(r);print(json.dumps({k:r.get(k) for k in ('seed','family','context_mode','hidden_width','clean_mse','recovered_segments','total_segments','failed')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,worlds=worlds,records=records),indent=2)+'\n');print('E20_COMPLETE',flush=True)

if __name__=='__main__':main()
