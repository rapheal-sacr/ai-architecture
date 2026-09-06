from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
d=json.loads((RUNS/'e21/complete.json').read_text());rs=d['records'];assert len(rs)==32
assert not any(r['failed'] for r in rs)
shutil.copyfile(RUNS/'e21/complete.json',HERE/'results/e21_complete.json')
shutil.copyfile(RUNS/'e21-preflight/preflight.json',HERE/'results/e21_preflight.json')
report=['# E21: faster evidence repairs much of the conflict failure but hurts input shift','',
'Every arm uses actual preceding observations for prediction; none receives hidden regimes. All have the same 738-parameter student and 512-example replay. The experiment crosses context decay 0.95 versus 0.5 with training/replay assignment before versus after incorporating the current observed outputs. Predictions always precede those outputs, even in the posterior-assignment arm. Four fresh worlds per family are paired across the four cells.', '',
'| Family | Context decay | Assignment | Mean clean MSE | Recovered segments | Mean seconds |',
'|---|---:|---|---:|---:|---:|']
summary=[]
for family in d['protocol']['families']:
    for decay in d['protocol']['context_decays']:
        for mode in d['protocol']['assignment_modes']:
            cases=[r for r in rs if r['family']==family and r['context_decay']==decay and r['context_mode']==mode]
            row=dict(family=family,context_decay=decay,assignment=mode,mean_mse=st.mean(r['clean_mse'] for r in cases),
                recovered_segments=sum(r['recovered_segments'] for r in cases),total_segments=sum(r['total_segments'] for r in cases),
                mean_seconds=st.mean(r['seconds'] for r in cases),costs=cases[0]['costs'],
                quarters=[st.mean(r['quarter_clean_mse'][i] for r in cases) for i in range(4)])
            summary.append(row);report.append(f"| {family} | {decay} | {mode} | {row['mean_mse']:.6f} | {row['recovered_segments']}/{row['total_segments']} | {row['mean_seconds']:.3f} |")
report+=['','## Non-oracle gain and its counterexample','']
for family in d['protocol']['families']:
    pair={(r['context_decay'],r['assignment']):r for r in summary if r['family']==family}
    reference=pair[(.95,'prior_assignment')];candidate=pair[(.5,'posterior_assignment')]
    wins=0
    for seed in d['protocol']['stream_seeds']:
        r={ (x['context_decay'],x['context_mode']):x for x in rs if x['family']==family and x['seed']==seed}
        wins+=r[(.5,'posterior_assignment')]['clean_mse']<r[(.95,'prior_assignment')]['clean_mse']
    report.append(f"For {family}, faster posterior assignment changes mean MSE by {candidate['mean_mse']/reference['mean_mse']-1:+.1%} relative to slow prior assignment, winning {wins}/4 fresh worlds. Its runtime ratio is {candidate['mean_seconds']/reference['mean_seconds']:.3f} with identical student-example counts and stored learner tensor bytes.")
report+=['',
'The factor comparison separates a large recency effect from a smaller assignment-timing effect in the conflicting-function family. A stale context estimate is therefore a major avoidable limitation of the previous implementation, not evidence that more shared weights are intrinsically necessary. Posterior assignment uses current outputs only after their prediction has been scored; its gain is evaluated on future prequential behavior, not lower training loss.', '',
'The input-shift counterexample prevents installing faster forgetting as a universal fix. Rapidly changing the contextual representation can add noise or unnecessary variation when the conditional function remains stable. Both task types require a criterion for allocating evidence across timescales; a fixed universally fast or slow context does not settle that problem.', '',
'## Costs and limits','',
'Every arm receives 131,072 observations, runs 4,096 real updates and 393,184 student forward examples, and retains 71,328 counted learner tensor bytes. No controller or meta-training is used. The posterior adapter incurs a redundant cross-moment calculation, with its actual elapsed time included. Raw data are shared across paired conditions; repeated learner exposures are not independent data acquisition. Full process RAM, complete FLOPs and energy are unmeasured.', '',
'The preflight verifies exact prior-path equality with E17, equal persistent context trajectories between assignment modes at a fixed decay, correct posterior features in replay, and equal counters. Every prediction remains causal. The current-label contribution to a training batch summary could still create a training shortcut; held-out prequential behavior is the relevant test. Recovery uses the frozen E17 criterion and retains all censored segments.', '',
'This is a concrete inexpensive repair for a scoped inference failure, with an explicit negative transfer result. It is not learned timescale allocation, semantic compression, general reasoning, novel inference theory, reliable recursive improvement or long-horizon closed-loop success. The broader architecture still needs those mechanisms and independent falsification.']
(HERE/'E21-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e21_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'.join(report))
