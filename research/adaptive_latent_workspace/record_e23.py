from pathlib import Path
import json,hashlib,shutil,statistics as st
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e23_conditional_evidence import ConditionalEvidence
from e21_evidence_assignment import PosteriorAssignment
HERE=Path(__file__).resolve().parent;RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e23')
d=json.loads((RUNS/'complete.json').read_text());rs=d['records'];spec=d['protocol'];assert len(rs)==80 and not any(r['failed'] for r in rs)
for n,h in d['identity'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
states={};checks=[]
for r in rs:
    path=Path(r['checkpoint']);p=torch.load(path,weights_only=False);s=p['learner_state'];assert p['identity']==d['identity']
    cls=ConditionalEvidence if r['descriptor']=='ridge_coefficients' else PosteriorAssignment;l=cls.restore(s)
    assert s['step']==4096 and s['size']==512 and l.costs()==r['costs'] and all(torch.isfinite(t).all() for t in s['params'])
    assert len(r['clean_batch_mse'])==4096 and r['costs']['student_forward_examples']==393184
    residual=None
    if r['descriptor']=='ridge_coefficients':
        assert l.solves==4096 and torch.isfinite(l.gram).all() and torch.isfinite(l.cross).all()
        residual=float(((l.gram+l.ridge*torch.eye(9))@l.context-l.cross).abs().max());assert residual<1e-5
    states[r['seed'],r['family'],r['decay'],r['descriptor']]=s
    checks.append(dict(seed=r['seed'],family=r['family'],decay=r['decay'],descriptor=r['descriptor'],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),normal_equation_residual=residual,restored_costs_exact=True))
pairs=0
for seed in spec['stream_seeds']:
    for family in spec['families']:
        for decay in spec['context_decays']:
            a=states[seed,family,decay,'raw_cross_moment'];b=states[seed,family,decay,'ridge_coefficients']
            assert torch.equal(a['replay_x'][:,:8],b['replay_x'][:,:8]) and torch.equal(a['replay_y'],b['replay_y']) and torch.equal(a['rng'],b['rng']);pairs+=1
summary=[]
for family in spec['families']:
    for decay in spec['context_decays']:
        for kind in spec['descriptors']:
            cases=[r for r in rs if r['family']==family and r['decay']==decay and r['descriptor']==kind]
            summary.append(dict(family=family,decay=decay,descriptor=kind,mse=st.mean(r['clean_mse'] for r in cases),seconds=st.mean(r['seconds'] for r in cases),
                recovered=sum(r['recovered_segments'] for r in cases),segments=sum(r['total_segments'] for r in cases),costs=cases[0]['costs'],
                short_segments=sum(r['segments_shorter_than_criterion'] for r in cases),quarter_mse=np.mean([r['quarter_clean_mse'] for r in cases],0).tolist()))
lookup={(r['family'],r['decay'],r['descriptor']):r for r in summary}
report=['# E23: regularized relationship evidence gives a modest scoped repair','',
'All 80 frozen cases completed without divergence. Four fresh worlds per family are paired across raw cross-moments versus regularized linear coefficients and context decays 0.95 versus 0.5. Predictions use preceding evidence; both descriptors annotate training and replay after the output arrives. The same 738-parameter student, 512-example reservoir, observations and update rule persist throughout. No gate, hidden regimes, extra model width or offline meta-training is supplied.', '',
'| Family | Decay | Descriptor | Mean clean MSE | Recovered / all segments | Mean seconds |', '|---|---:|---|---:|---:|---:|']
for r in summary:report.append(f"| {r['family']} | {r['decay']} | {r['descriptor']} | {r['mse']:.6f} | {r['recovered']}/{r['segments']} | {r['seconds']:.3f} |")
report+=['','## Paired improvement and counterexamples',''];comparison=[]
for family in spec['families']:
    for decay in spec['context_decays']:
        raw=lookup[family,decay,'raw_cross_moment'];ridge=lookup[family,decay,'ridge_coefficients']
        wins=0;losses=[]
        for seed in spec['stream_seeds']:
            a=next(r for r in rs if r['seed']==seed and r['family']==family and r['decay']==decay and r['descriptor']=='raw_cross_moment')
            b=next(r for r in rs if r['seed']==seed and r['family']==family and r['decay']==decay and r['descriptor']=='ridge_coefficients')
            wins+=b['clean_mse']<a['clean_mse']
            if b['clean_mse']>=a['clean_mse']:losses.append(seed)
        v=dict(family=family,decay=decay,relative_error=ridge['mse']/raw['mse']-1,wins=wins,losses=losses,time_ratio=ridge['seconds']/raw['seconds']);comparison.append(v)
        report.append(f"- {family}, decay {decay}: {v['relative_error']:+.1%} mean error; {wins}/4 wins; {v['time_ratio']:.3f} times runtime. Counterexample seeds: {losses or 'none in these four' }.")
report+=['',
'The representation change improves every paired independent and alternating world at both decays, with a modest gain on top of the stronger fast/post-outcome rule. Input-shift and conflicting-function counterexamples remain. The result supports regularized coefficients as a scoped evidence descriptor; it does not establish invariant task identity or an optimal timescale. Recovery can worsen even where whole-stream error improves, as the individual records show.', '',
'In noiseless full-rank linear data, unregularized coefficients remove the input-distribution dependence of cross-moments. The preflight confirms that algebra. Scored teachers are nonlinear, so these coefficients remain distribution-weighted projections, and ridge introduces bias. Descriptor scale and orientation also change. This experiment cannot attribute the entire gain solely to covariance normalization or claim general invariance.', '',
'## Costs and audit','']
rawcases=[r for r in rs if r['descriptor']=='raw_cross_moment'];ridgecases=[r for r in rs if r['descriptor']=='ridge_coefficients'];ratio=st.mean(r['seconds'] for r in ridgecases)/st.mean(r['seconds'] for r in rawcases)
report += [f'Each learner consumes 131,072 observations, 4,096 student updates and 393,184 student forward examples. Ridge adds 4,096 nine-by-nine solves and Gram updates, plus 396 persistent tensor bytes: {ridgecases[0]["costs"]["stored_tensor_bytes"]:,} versus {rawcases[0]["costs"]["stored_tensor_bytes"]:,}. Its mean local runtime is {ratio:.3f} times raw/post-outcome execution. Both implementations incur the inherited redundant context calculation. There is no gate backward pass or bootstrap.', '',
'Every final checkpoint matches its frozen source identity and restored counters, has 4,096 persistent steps and a 512-example reservoir. All 40 paired raw reservoir input/target/RNG checks match exactly; contextual annotations may differ. All 40 final coefficient descriptors satisfy their regularized normal equations within the recorded numerical tolerance. Preflight tests exact raw E21 controls, causal initial predictions, post-outcome storage, full trajectory restoration and the linear algebra invariance claim.', '',
'All censored recovery segments, including those shorter than the three-batch criterion, are preserved. The twenty shared worlds are the independent seed/family units; repeated learner exposures are not additional acquired data. These local loop times exclude data generation, preflight and checkpoint serialization, and are not complete deployment latency. Full process RAM, peak solver workspace, complete FLOPs, energy, rare semantic recall and autonomous planning horizon remain unmeasured.', '',
'This is established adaptive-regression machinery, credited to prior work including ALPaCA, not a novelty claim. It is a stronger fixed baseline for this subproblem, not autonomous recursive improvement. No result here repairs the world-model search failure or establishes safe compression. The next experiment, E24, returns to the outstanding rare-memory audit rather than continuing to tune this family.']
for n in ('complete','preflight'):shutil.copyfile(RUNS/f'{n}.json',HERE/f'results/e23_{n}.json')
(HERE/'results/e23_summary.json').write_text(json.dumps(dict(rows=summary,comparisons=comparison),indent=2)+'\n')
(HERE/'results/e23_audit.json').write_text(json.dumps(dict(identity=d['identity'],checks=checks,paired_raw_reservoir_and_rng_checks=pairs),indent=2)+'\n')
(HERE/'E23-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.5));labels=['Conflicts','Input','Coupled','Independent','Alternating']
for i,decay in enumerate(spec['context_decays']):
    for j,family in enumerate(spec['families']):
        v=next(r for r in comparison if r['family']==family and r['decay']==decay)
        axes[0].bar(j+(i-.5)*.35,100*v['relative_error'],.35,label=f'decay {decay}' if j==0 else None,color=['#777777','#0072B2'][i])
    axes[1].plot(range(5),[lookup[f,decay,'ridge_coefficients']['recovered']/lookup[f,decay,'ridge_coefficients']['segments'] for f in spec['families']],marker='o',label=f'ridge {decay}')
    axes[1].plot(range(5),[lookup[f,decay,'raw_cross_moment']['recovered']/lookup[f,decay,'raw_cross_moment']['segments'] for f in spec['families']],marker='x',ls='--',label=f'raw {decay}')
axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel('Ridge change in mean MSE (%)');axes[0].set_title('Modest gain, not universal invariance');axes[0].legend()
axes[1].set_ylabel('Recovered / all observed segments');axes[1].set_title('Recovery still depends strongly on evidence decay');axes[1].legend(fontsize=8)
for ax in axes:ax.set_xticks(range(5),labels,rotation=20)
fig.suptitle(f'E23 • four paired seeds per family • ridge costs {ratio:.2f}× local time and +396 tensor bytes');fig.tight_layout();fig.savefig(HERE/'results/e23_conditional_evidence.png',dpi=160)
print('\n'.join(report[27:42]));print('AUDIT_OK',len(checks),'pairs',pairs,'wins',sum(c['wins'] for c in comparison),'of40','runtime_ratio',ratio)
