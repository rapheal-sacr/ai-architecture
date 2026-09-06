from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
d=json.loads((RUNS/'e19/complete.json').read_text());rs=d['records'];assert len(rs)==32
assert not any(r['result']['failed'] for r in rs)
shutil.copyfile(RUNS/'e19/complete.json',HERE/'results/e19_complete.json')
shutil.copyfile(RUNS/'e19-preflight/preflight.json',HERE/'results/e19_preflight.json')
report=['# E19: isolate the procedure from the selected student state','',
'Each pair starts from exactly the same final E18 self-applied student, optimizer moments, context, replay and RNG. Only the procedure differs: keep the final updater or restore its E9 bootstrap predecessor. Both procedures are frozen while the student learns for another 4,096 batches (131,072 examples), reaching 6,400 global updates. Every original E18 self-applied case is included.', '',
'| Replicate | Source seed | Family | Continuation | Keep MSE | Revert MSE | Keep / revert recovered full segments |',
'|---|---|---|---|---:|---:|---|']
summary=[]
for family in d['protocol']['families']:
    for mode in d['protocol']['continuation_modes']:
        for rep in d['protocol']['controller_replicates']:
            for seed in d['protocol']['source_stream_seeds']:
                pair={r['arm']:r['result'] for r in rs if r['family']==family and r['mode']==mode and r['replicate']==rep and r['source_stream_seed']==seed}
                a,b=pair['keep_final_procedure'],pair['revert_bootstrap_procedure']
                assert a['full_segments']==b['full_segments']
                row=dict(replicate=rep,source_seed=seed,family=family,mode=mode,keep_mse=a['clean_mse'],revert_mse=b['clean_mse'],
                    keep_recovery=a['recovered_full_segments'],revert_recovery=b['recovered_full_segments'],full_segments=a['full_segments'],
                    keep_quarters=a['quarter_clean_mse'],revert_quarters=b['quarter_clean_mse']);summary.append(row)
                report.append(f"| {rep} | {seed} | {family} | {mode} | {a['clean_mse']:.6f} | {b['clean_mse']:.6f} | {a['recovered_full_segments']} / {b['recovered_full_segments']} of {a['full_segments']} |")
report+=['','## What keeping the learned procedure changes','']
for family in d['protocol']['families']:
    for mode in d['protocol']['continuation_modes']:
        rows=[r for r in summary if r['family']==family and r['mode']==mode]
        n=sum(r['keep_mse']<r['revert_mse'] for r in rows)
        report.append(f"{family}, {mode}: keeping the final procedure wins {n}/4 paired cases; mean MSE is {st.mean(r['keep_mse'] for r in rows):.6f} versus {st.mean(r['revert_mse'] for r in rows):.6f} after reverting.")
report+=['',
'Input-shift transfer has a horizon-dependent tradeoff. On new worlds, keeping the procedure has worse first-quarter mean MSE (0.057436 versus 0.044890), but better last-quarter MSE (0.010928 versus 0.013720). That late benefit has not repaid its initial deficit over this complete continuation. On same-world input shift, keeping is better throughout the quarters. Thus the procedure specializes to a learning/retention tradeoff, rather than simply becoming uniformly better or uniformly worse.', '',
'A keep/revert difference is caused by the procedure intervention within this fixed-start comparison, including its later interaction with the retained student and replay. It cannot be explained solely by starting with different selected student weights. However, a benefit confined to the original world is specialization; it does not establish a universally better learning algorithm. Reverting can also disrupt useful student/procedure co-adaptation, which this test does not separate from state-independent procedure quality.', '',
'The two procedure bootstraps share each of the two source/new worlds. Four cases per family/mode are dependent crossed cases, not four independent full-pipeline replications. Censored recovery and error are reported separately; improvements in average MSE must not hide unrecovered segments.', '',
'## Continuation and cost checks','',
'All four original-world prefix checks match every observation, noisy target and clean target exactly, as well as clipped segment metadata. The originally sampled latent segment continues across the former stop; no artificial switch is forced. The first partial continuation segment is marked left-truncated and retained in per-batch error, but separated from full-segment recovery summaries. No failed full segment is omitted.', '',
'For fresh worlds, seeds 19101/19102 replace the generating teachers while all old learner and memory state persists. Clean targets and segment labels remain recorder-only. Starting non-procedure state and first predictions are exact matches. Final procedure tensors are asserted unchanged, and paired experience/context/RNG state is checked to remain identical despite different student weights.', '',
'| Arm | Mean seconds | Continuation student examples | Gradient batches | Controller calls | Final stored tensor bytes |',
'|---|---:|---:|---:|---:|---:|']
for arm in d['protocol']['arms']:
    cases=[r['result'] for r in rs if r['arm']==arm];c=cases[0]['continuation_costs']
    report.append(f"| {arm} | {st.mean(r['seconds'] for r in cases):.3f} | {c['student_forward_examples']} | {c['student_gradient_batches']} | {c['controller_forward_calls']} | {cases[0]['global_costs']['stored_tensor_bytes']} |")
report+=['',
'There are 32 newly executed learner continuations. World data are shared across paired arms/replicates, so repeated learner exposures are not independent data acquisition. Dataset-generation time is stored in `world_checks`; each pair\'s restoration time is stored twice for reference and must be charged only once. The separate first-prediction check adds 64 student examples per pair (1,024 total), outside continuation counters. Prefix regeneration checks data without retraining source students. Standalone and execution preflights are additional dummy work.', '',
'All E9 bootstrap and E18 student/procedure-trial work remains inherited. Equal continuation work does not make the accumulated learned procedure free. Timings overlap the E20 CPU experiment, so they are descriptive. Complete FLOPs, process RAM and energy remain unmeasured.', '',
'This is a necessary causal attribution test in full-feedback supervised streams. It does not demonstrate semantic compression, general reasoning, closed-loop counterfactual validity, novelty or open-ended recursive improvement. E18\'s original failures and extra costs remain in the record regardless of this continuation.']
(HERE/'E19-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e19_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'.join(report))
