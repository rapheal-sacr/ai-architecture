from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from e17_persistent_student import PersistentStudent
from e21_evidence_assignment import PosteriorAssignment
from e22_learned_evidence import world
from e18_online_procedure import segment_results
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class ConditionalEvidence(PersistentStudent):
    def __init__(self,cfg,base,seed,lr,ridge):
        super().__init__(cfg,base,seed,None,lr)
        self.gram=torch.zeros(self.context.shape[0],self.context.shape[0]);self.cross=torch.zeros_like(self.context)
        self.ridge=ridge;self.solves=0
    def observe(self,x,y):
        z=torch.cat([x,torch.ones(len(x),1)],1);decay=self.cfg['context_ema']
        self.gram.mul_(decay).add_(z.T@z/len(x),alpha=1-decay)
        self.cross.mul_(decay).add_(z.T@y/len(x),alpha=1-decay)
        posterior=torch.linalg.solve(self.gram+self.ridge*torch.eye(len(self.gram)),self.cross)
        self.solves+=1
        if not torch.isfinite(posterior).all():raise FloatingPointError('Nonfinite coefficient descriptor')
        self.context=posterior.clone();super().observe(x,y)
        # Match E21 post-outcome assignment; discard inherited redundant context update.
        self.context=posterior
    def costs(self):
        c=super().costs();c['stored_tensor_bytes']+=self.gram.numel()*self.gram.element_size()+self.cross.numel()*self.cross.element_size()
        c.update(matrix_solves=self.solves,gram_updates=self.solves,solve_dimension=len(self.gram));return c
    def state(self):
        s=super().state();s['conditional_evidence']=dict(gram=self.gram,cross=self.cross,ridge=self.ridge,solves=self.solves);return s
    @classmethod
    def restore(cls,s):
        e=s['conditional_evidence'];obj=cls(s['cfg'],s['base'],0,s['lr'],e['ridge'])
        raw=PersistentStudent.restore(s);obj.__dict__.update(raw.__dict__)
        obj.gram=e['gram'].clone();obj.cross=e['cross'].clone();obj.solves=e['solves'];return obj

def create(cfg,base,seed,spec,kind,decay):
    c=copy.deepcopy(cfg);c['context_ema']=decay
    if kind=='raw_cross_moment':return PosteriorAssignment(c,base,seed,None,spec['student_learning_rate'])
    assert kind=='ridge_coefficients';return ConditionalEvidence(c,base,seed,spec['student_learning_rate'],spec['ridge'])

def preflight(spec,cfg,base):
    c=copy.deepcopy(cfg);c['updates']=12;rows,_=world(c,103,'independent_shift');checks=[]
    for decay in spec['context_decays']:
        c['context_ema']=decay
        raw=create(c,base,104,spec,'raw_cross_moment',decay);old=PosteriorAssignment(c,base,104,None,.01)
        ridge=create(c,base,104,spec,'ridge_coefficients',decay)
        assert torch.equal(raw.predict(rows[0][0]),ridge.predict(rows[0][0]));raw.counts['student_forward_examples']=0;ridge.counts['student_forward_examples']=0
        last_residual=None
        for x,y,_ in rows:
            assert torch.equal(raw.predict(x),old.predict(x));raw.observe(x,y);old.observe(x,y)
            before=ridge.context.clone();prediction=ridge.predict(x);assert torch.equal(before,ridge.context)
            ridge.observe(x,y)
            residual=(ridge.gram+spec['ridge']*torch.eye(9))@ridge.context-ridge.cross
            last_residual=float(residual.abs().max());assert last_residual<1e-5
            assert torch.equal(ridge.replay_x[max(0,ridge.size-len(x)):ridge.size,8:],ridge.context.reshape(1,-1).expand(min(len(x),ridge.size),-1))
        assert all(torch.equal(a,b) for a,b in zip(raw.params,old.params)) and raw.costs()==old.costs()
        resumed=ConditionalEvidence.restore(copy.deepcopy(ridge.state()));x,y,_=rows[0]
        assert torch.equal(ridge.predict(x),resumed.predict(x));ridge.observe(x,y);resumed.observe(x,y)
        assert all(torch.equal(a,b) for a,b in zip(ridge.params,resumed.params))
        assert torch.equal(ridge.gram,resumed.gram) and torch.equal(ridge.cross,resumed.cross)
        assert torch.equal(ridge.replay_x,resumed.replay_x) and torch.equal(ridge.rng.get_state(),resumed.rng.get_state())
        assert ridge.costs()==resumed.costs() and ridge.solves==13
        checks.append(dict(decay=decay,exact_raw_control=True,resume_exact=True,post_outcome_annotations=True,normal_equation_residual=last_residual,costs=ridge.costs()))
    # Algebra check only: coefficients are invariant for a full-rank noiseless linear model without ridge.
    g=torch.Generator().manual_seed(105);w=torch.randn(9,2,generator=g,dtype=torch.float64);coeff=[];raws=[]
    for shift,scale in [(0.,1.),(3.,.5),(-2.,2.)]:
        x=torch.randn(128,8,generator=g,dtype=torch.float64)*scale+shift;z=torch.cat([x,torch.ones(128,1,dtype=torch.float64)],1);y=z@w
        gram=z.T@z/len(z);cross=z.T@y/len(z);beta=torch.linalg.solve(gram,cross)
        assert torch.allclose(beta,w,atol=1e-10,rtol=1e-10);coeff.append(float((beta-w).abs().max()));raws.append(cross)
    assert not torch.allclose(raws[0],raws[1])
    return dict(controls=checks,linear_invariance_max_errors=coeff,raw_moments_change_under_shift=True,algebra_only_not_nonlinear_invariance=True)

def run(spec,cfg,base,seed,family,kind,decay,rows,segments,out,identity):
    l=create(cfg,base,seed,spec,kind,decay);clean=[];noisy=[];norm=[];changes=[];magnitudes=[];failure=None;start=time.perf_counter()
    for t,(x,y,truth) in enumerate(rows):
        prediction=l.predict(x);cl=float((prediction-truth).square().mean());loss=float((prediction-y).square().mean())
        if not math.isfinite(loss):failure=f'Nonfinite prediction at {t}';break
        clean.append(cl);noisy.append(loss);norm.append(cl/(float(truth.square().mean())+.01));before=l.context.clone()
        try:l.observe(x,y)
        except (FloatingPointError,torch.linalg.LinAlgError) as e:failure=str(e);break
        changes.append(float((l.context-before).square().mean()));magnitudes.append(float(l.context.square().mean()))
    elapsed=time.perf_counter()-start;stem=f'{seed}-{family}-{kind}-{decay}'
    r=dict(seed=seed,family=family,descriptor=kind,decay=decay,identity=identity,failed=failure,seconds=elapsed,costs=l.costs(),completed_batches=len(clean),
        clean_batch_mse=clean,observed_batch_mse=noisy,normalized_batch_mse=norm,descriptor_squared_changes=changes,descriptor_mean_squares=magnitudes)
    if not failure:
        assert l.step==4096 and l.size==512 and l.counts['student_forward_examples']==393184
        if kind=='ridge_coefficients':assert l.solves==4096
        segs=segment_results(segments,clean,norm,cfg)
        r.update(clean_mse=float(np.mean(clean)),observed_mse=float(np.mean(noisy)),quarter_clean_mse=[float(np.mean(q)) for q in np.array_split(clean,4)],
            segments=segs,recovered_segments=sum(s['recovery_batches'] is not None for s in segs),total_segments=len(segs),
            segments_shorter_than_criterion=sum(s['stop']-s['start']<cfg['recovery_consecutive_batches'] for s in segs))
        if family=='alternating_dependency':r['dependency_phases']=[dict(start=i,stop=i+512,function_changes_active=bool((i//512)%2),clean_mse=float(np.mean(clean[i:i+512]))) for i in range(0,len(clean),512)]
        torch.save(dict(identity=identity,learner_state=l.state()),out/f'{stem}.pt');r['checkpoint']=str(out/f'{stem}.pt')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e23.json').read_text());cfg=json.loads((HERE/'protocol_e17.json').read_text());cfg['updates']=spec['updates'];base=json.loads((HERE/'protocol_e9.json').read_text())
    names=('protocol_e23.json','e23_conditional_evidence.py','e22_learned_evidence.py','e21_evidence_assignment.py','e17_persistent_student.py','e18_online_procedure.py','protocol_e17.json','protocol_e9.json','e9_recursive_updater.py')
    identity={n:digest(HERE/n) for n in names};a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();checks=preflight(spec,cfg,base)
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks,seconds=time.perf_counter()-start),indent=2)+'\n');print('E23 conditional evidence preflight passed',flush=True)
    if not a.execute:return
    records=[];worlds=[]
    for seed in spec['stream_seeds']:
        for family in spec['families']:
            start=time.perf_counter();rows,segments=world(cfg,seed,family);worlds.append(dict(seed=seed,family=family,generation_seconds=time.perf_counter()-start))
            for decay in spec['context_decays']:
                for kind in spec['descriptors']:
                    stem=f'{seed}-{family}-{kind}-{decay}';path=a.out/f'{stem}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity
                    else:
                        assert not (a.out/f'{stem}.pt').exists(),'Unrecorded checkpoint: inspect before any restart'
                        r=run(spec,cfg,base,seed,family,kind,decay,rows,segments,a.out,identity);path.write_text(json.dumps(r,indent=2)+'\n')
                    records.append(r);print(json.dumps({k:r.get(k) for k in ('seed','family','descriptor','decay','clean_mse','recovered_segments','total_segments','failed')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,worlds=worlds,records=records),indent=2)+'\n');print('E23_COMPLETE',flush=True)
if __name__=='__main__':main()
