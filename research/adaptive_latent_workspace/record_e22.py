from pathlib import Path
import json,hashlib,shutil,statistics as st
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e22_learned_evidence import EvidenceLearner
from e17_persistent_student import PersistentStudent
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e22')
d=json.loads((RUNS/'complete.json').read_text());rs=d['records'];spec=d['protocol']
assert len(rs)==100 and not any(r['failed'] for r in rs)
for n,h in d['identity'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
checks=[];states={}
for r in rs:
    p=Path(r['checkpoint']);packed=torch.load(p,weights_only=False);s=packed['learner_state']
    assert packed['identity']==d['identity'] and s['step']==4096 and s['size']==512
    assert all(torch.isfinite(t).all() for t in s['params'])
    assert len(r['clean_batch_mse'])==4096 and r['costs']['student_forward_examples']==393184
    restore=EvidenceLearner.restore(s) if 'evidence' in s else PersistentStudent.restore(s)
    assert restore.costs()==r['costs']
    if r['arm']=='learned_gate':
        assert s['evidence']['gate_updates']==4096 and all(torch.isfinite(t).all() for t in s['evidence']['gate_params'])
        assert any(torch.count_nonzero(t) for t in s['evidence']['gate_params'][2:])
    states[(r['seed'],r['family'],r['arm'])]=s
    checks.append(dict(seed=r['seed'],family=r['family'],arm=r['arm'],checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),step=s['step'],restored_costs_exact=True))
pairs=0
for seed in spec['stream_seeds']:
    for fam in spec['families']:
        ref=states[(seed,fam,'slow_prior')]
        for arm in spec['arms'][1:]:
            s=states[(seed,fam,arm)]
            # Stored contextual annotations may differ; the actual sampled experiences must not.
            assert torch.equal(s['replay_x'][:,:8],ref['replay_x'][:,:8])
            assert torch.equal(s['replay_y'],ref['replay_y']) and torch.equal(s['rng'],ref['rng'])
            pairs+=1
summary=[]
for family in spec['families']:
    for arm in spec['arms']:
        cases=[r for r in rs if r['family']==family and r['arm']==arm]
        summary.append(dict(family=family,arm=arm,mean_mse=st.mean(r['clean_mse'] for r in cases),
            recovered=sum(r['recovered_segments'] for r in cases),segments=sum(r['total_segments'] for r in cases),
            short_segments=sum(r['segments_shorter_than_criterion'] for r in cases),seconds=st.mean(r['seconds'] for r in cases),
            costs=cases[0]['costs'],quarter_mse=np.mean([r['quarter_clean_mse'] for r in cases],0).tolist(),
            gate_quarters=np.mean([r['gate_quarter_means'] for r in cases],0).tolist() if arm in ('fixed_mixture','learned_gate') else None))
lookup={(r['family'],r['arm']):r for r in summary}
report=['# E22: learned evidence allocation does not repair the tradeoff','',
'All 100 frozen cases completed without divergence: four fresh worlds in each of five families, paired across five arms, with 4,096 updates and 131,072 observations per run. The 65-parameter gate actually learns online from each recorded prediction after its output arrives. Student weights, gate, optimizer moments, context and bounded replay persist across hidden changes. There are no oracle boundaries or family inputs.', '',
'| Family | Arm | Mean clean MSE | Recovered / all segments | Mean seconds |',
'|---|---|---:|---:|---:|']
for r in summary:report.append(f"| {r['family']} | {r['arm']} | {r['mean_mse']:.6f} | {r['recovered']}/{r['segments']} | {r['seconds']:.3f} |")
report+=['','## Direct falsification','']
for family in spec['families']:
    candidate=lookup[family,'learned_gate'];comps=[]
    for arm in ('slow_prior','fast_prior','fast_posterior','fixed_mixture'):
        control=lookup[family,arm];wins=sum(next(r for r in rs if r['seed']==s and r['family']==family and r['arm']=='learned_gate')['clean_mse']<next(r for r in rs if r['seed']==s and r['family']==family and r['arm']==arm)['clean_mse'] for s in spec['stream_seeds'])
        comps.append(f"{arm}: {candidate['mean_mse']/control['mean_mse']-1:+.1%} error, {wins}/4 wins")
    report.append(f"- {family}: "+'; '.join(comps)+'.')
report+=['',
'The strong fast/prior control removes the post-outcome assignment difference and still defeats the learned gate on every conflicting, independent and alternating world. The weak slow/prior reference alone would have hidden that failure. The gate also loses to slow/prior on every input-shift world. Coupled shifts supply a small scoped gain over fast/prior in some worlds, without a general improvement.', '',
'## Persistent adaptation and its boundary','',
'| Family | Gate mean, first quarter | Second | Third | Fourth |', '|---|---:|---:|---:|---:|']
for family in spec['families']:
    q=lookup[family,'learned_gate']['gate_quarters'];report.append('| '+family+' | '+' | '.join(f'{x:.3f}' for x in q)+' |')
phase={}
for arm in spec['arms']:
    cases=[r for r in rs if r['family']=='alternating_dependency' and r['arm']==arm]
    phase[arm]=[dict(phase=i,function_changes_active=bool(i%2),mean_mse=st.mean(r['dependency_phases'][i]['clean_mse'] for r in cases),mean_gate=st.mean(r['dependency_phases'][i]['gate_mean'] for r in cases) if arm in ('fixed_mixture','learned_gate') else None) for i in range(8)]
report+=['',
'Fast evidence receives increasing weight on conflicting functions and slow evidence on input shift. That movement is compatible with learning a persistent timescale preference; it does not establish rapid detection of each relationship change. The alternating family changes its dependency regime every 512 batches within the same student lifetime. Its gate trajectories and all eight phase errors are retained in the summary and plot. Inference about slow adaptation or saturation is provisional: these results do not isolate a unique cause from representation, gate features, local objective and student/gate coadaptation.', '',
'## Work, audit and unmeasured quantities','']
for arm in spec['arms']:
    cases=[r for r in rs if r['arm']==arm];report.append(f"- {arm}: mean {st.mean(r['seconds'] for r in cases):.3f} seconds; {cases[0]['costs']['stored_tensor_bytes']:,} counted learner tensor bytes.")
ratio=st.mean(r['seconds'] for r in rs if r['arm']=='learned_gate')/st.mean(r['seconds'] for r in rs if r['arm']=='fast_prior')
report += ['',f'Learned-gate execution costs {ratio:.2f} times fast/prior time. All arms use 393,184 student forward examples and 4,096 student updates per run. The gate adds 4,096 gate forwards and 4,096 extra backward passes through the student into 65 gate parameters, plus gate optimizer state and two evidence summaries. Equal student forward counts therefore do not mean equal training work. There is no offline gate bootstrap. Data generation, preflight and checkpoint serialization are outside timed learner loops and recorded separately where available; full end-to-end deployment latency was not measured.', '',
'All 100 final states have the expected persistent step and bounded replay, finite weights, matching source identity and exactly reproduced restored cost counters. All 80 control comparisons have identical raw reservoir inputs, targets and RNG state; contextual annotations can differ. The preflight checks exact fixed controls, initial half-mixture equality, causal gate derivatives by finite differences, actual gate parameter changes, exact resume and independent generator clocks. Checkpoint hashes are recorded in the audit.', '',
'Independent/alternating evaluation segments use the union of input and function changes, so some last fewer than the three batches required by the recovery criterion. Their counts and all censored segments remain in the data. Recovery totals across different families are not directly comparable. Four world seeds remain a small sample; no significance or universal bound is claimed. Full peak autograd memory, process RAM, FLOPs, energy, semantic recall and autonomous task horizon were not measured.', '',
'The proposed gate is rejected as a sufficient repair of the evidence-allocation problem. Its implementation is real continual procedure adaptation, but it is neither recursive self-application nor demonstrated general improvement. No claimed success warrants the optional keep/revert gate continuation yet. The broader architecture remains unvalidated.']
for name in ('complete','preflight'):shutil.copyfile(RUNS/f'{name}.json',HERE/f'results/e22_{name}.json')
(HERE/'results/e22_audit.json').write_text(json.dumps(dict(source_identity=d['identity'],cases=checks,paired_raw_reservoir_and_rng_checks=pairs),indent=2)+'\n')
(HERE/'results/e22_summary.json').write_text(json.dumps(dict(rows=summary,alternating_phases=phase),indent=2)+'\n')
(HERE/'E22-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,3,figsize=(16,4.8));colors=['#777777','#0072B2','#009E73','#CC79A7','#D55E00']
for ai,arm in enumerate(spec['arms']):
    axes[0].plot(range(5),[lookup[f,arm]['mean_mse']/lookup[f,'fast_prior']['mean_mse'] for f in spec['families']],marker='o',label=arm,color=colors[ai])
    axes[2].plot(range(1,9),[p['mean_mse'] for p in phase[arm]],marker='o',color=colors[ai])
for i,f in enumerate(spec['families']):
    cases=[r for r in rs if r['family']==f and r['arm']=='learned_gate']
    a=np.mean([r['gate_weights'] for r in cases],0).reshape(32,128).mean(1)
    axes[1].plot(np.arange(32)*128+64,a,label=f)
axes[0].axhline(1,color='black',lw=.7);axes[0].set_xticks(range(5),['Conflicts','Input','Coupled','Independent','Alternating'],rotation=25)
axes[0].set_ylabel('Mean MSE / fast-prior mean (lower is better)');axes[0].legend(fontsize=8);axes[0].set_title('Learned gate does not dominate controls')
axes[1].set_xlabel('Online update');axes[1].set_ylabel('Fast evidence weight');axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=7);axes[1].set_title('Gate movement is not sufficient evidence')
for i in range(2,9,2):axes[2].axvspan(i-.5,i+.5,alpha=.08,color='black')
axes[2].set_xlabel('512-update phase (shaded: function changes)');axes[2].set_ylabel('Mean prequential clean MSE');axes[2].set_title('Changing dependencies within one lifetime')
fig.suptitle('E22 • four paired seeds per family • 100 complete cases');fig.tight_layout();fig.savefig(HERE/'results/e22_learned_evidence.png',dpi=160)
print('\n'.join(report[:35]));print('AUDIT_OK',len(checks),'pairs',pairs,'runtime_ratio',ratio)
