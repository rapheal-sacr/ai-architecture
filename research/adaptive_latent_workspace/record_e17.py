from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e17/complete.json').read_text());rs=data['records'];assert len(rs)==96
assert not any('failed' in r['result'] for r in rs)
shutil.copyfile(RUNS/'e17/complete.json',HERE/'results/e17_complete.json')
shutil.copyfile(RUNS/'e17-preflight/preflight.json',HERE/'results/e17_preflight.json')
report=['# E17: reset-trained procedures transfer modestly, with higher execution cost','',
'Students, optimizer moments, inferred context and 512-example reservoir replay persist across 2,048 updates (65,536 observed examples). Unannounced regimes last 16–64 batches and recur unpredictably. One family changes the nonlinear target function; the other changes input distributions under one fixed function. Four fresh stream seeds are paired across two inherited procedure-bootstrap replicates and six procedures. The learned procedures themselves are frozen throughout this test.','',
'| Procedure replicate | Family | Arm | Mean clean MSE | Recovered segments | Mean seconds |','|---|---|---|---:|---:|---:|']
summary=[]
for rep in data['protocol']['controller_replicates']:
    for family in data['protocol']['families']:
        for arm in data['protocol']['arms']:
            cases=[r for r in rs if r['replicate']==rep and r['family']==family and r['arm']==arm]
            results=[r['result'] for r in cases];segs=[s for r in results for s in r['segments']]
            row=dict(replicate=rep,family=family,arm=arm,mean_clean_mse=st.mean(r['clean_mse'] for r in results),
                recovered_segments=sum(r['recovered_segments'] for r in results),total_segments=len(segs),
                mean_seconds=st.mean(r['seconds'] for r in results),
                mean_fraction_consumed_until_recovery_or_censoring=st.mean((s['recovery_batches'] or s['stop']-s['start'])/(s['stop']-s['start']) for s in segs),
                costs=results[0]['costs'])
            summary.append(row)
            report.append(f"| {rep} | {family} | {arm} | {row['mean_clean_mse']:.6f} | {row['recovered_segments']}/{row['total_segments']} | {row['mean_seconds']:.3f} |")
report+=['',
'The self-applied procedure has lower family-average MSE than the equal-extra-meta controller in all four replicate/family aggregates, but the margins are small and aggregate direction hides stream-level reversals. The independent bootstrap count remains two. The same four worlds appear under both procedure replicates; the repeated fixed arms are reused outputs, not new replications.','']
for rep in data['protocol']['controller_replicates']:
    for family in data['protocol']['families']:
        counts={other:0 for other in ('frozen_bootstrap','fixed_meta_adam','adam_01')}
        for seed in data['protocol']['stream_seeds']:
            pair={r['arm']:r['result'] for r in rs if r['replicate']==rep and r['family']==family and r['stream_seed']==seed}
            for other in counts:counts[other]+=pair['self_applied']['clean_mse']<pair[other]['clean_mse']
        report.append(f"Replicate {rep}, {family}: self-application wins {counts['frozen_bootstrap']}/4 streams against freezing, {counts['fixed_meta_adam']}/4 against conventional meta-updates, and {counts['adam_01']}/4 against fixed Adam 0.01.")
report+=['',
'For conflicting functions, the self-applied procedures recover in 115/228 segments each. The second frozen bootstrap recovers in 121/228. Lower average error therefore does not imply faster or more reliable recovery in each setting. In the input-shift family the stronger fixed and learned procedures recover in 227/228 segments; the assay separates the families much more sharply than it separates those procedures.', '',
'## Recovery criterion and its limits','',
'Recovery is completion of three consecutive batches whose clean prequential MSE divided by mean squared clean target plus 0.01 is at most 0.2. The clean targets, boundaries and recurrence labels are used only by the recorder. Segments that do not reach the threshold remain right-censored at their actual end. No failed segment is excluded from the denominator, and no uncensored mean recovery time is asserted. The criterion is a declared operational choice, not universal competence. Returning-segment recovery may include relearning; it is not a certificate of retained knowledge.', '',
'This supplies a first censored recovery assay for the candidate bottleneck. Averaging the fraction of each segment consumed before recovery or its end, the two self-applied procedures use 82.63% and 81.99% under conflicting functions, versus 11.06% and 10.97% under input shift. An unrecovered segment contributes its full observed lifetime; this is a capped burden statistic, not an uncensored recovery-time estimate. Whole-stream error can improve while most useful time remains spent below the competence threshold. It does not prove one universal quantity bounds every AI system or isolate an information-theoretic minimum.', '',
'## Work, storage and scope','',
'All arms consume the same 65,536 observations and use 2,048 real student updates, 65,504 replayed examples and 196,576 student forward examples per run. Learned procedures add 12,288 per-tensor controller calls. They cost about 6.06 seconds versus 2.67 seconds for fixed Adam on this local runtime, before inherited procedure training/selection work. Their modest MSE gains therefore do not establish total-resource efficiency. The three fixed rates are all reported; no final-data optimizer selection is claimed as an agent capability.', '',
'Stored tensor counts include student parameters, optimizer moments, replay, context, controller and RNG state. Python metadata, full process RAM, complete FLOPs and energy are not measured. Existing E9 bootstrap/self-trial and E11 conventional-meta costs remain inherited costs; see their reports. Of 96 reported combinations, 24 fixed-optimizer cases are reused across bootstrap replicates, leaving 72 newly executed stream runs. Standalone and execution preflights are additional dummy work.', '',
'No final task diverged. Passing this transfer assay does not mean the procedure learns continuously here: it is frozen after E9/E11 while the student retains state. E18 is separately frozen to test actual online procedure changes with persistent students. There is still no closed-loop planner, semantic memory compression, broad reasoning or novelty result in E17.']
(HERE/'E17-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e17_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('\n'.join(report))
