from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e18/complete.json').read_text());rs=data['records'];assert len(rs)==32
shutil.copyfile(RUNS/'e18/complete.json',HERE/'results/e18_complete.json')
shutil.copyfile(RUNS/'e18-preflight/preflight.json',HERE/'results/e18_preflight.json')
failures=[r for r in rs if 'failed' in r['result']]
if failures:
    (HERE/'E18-RESULT.md').write_text('# E18: failed cases retained\n\n'+json.dumps(failures,indent=2)+'\n')
    raise SystemExit('Failures present; inspect all failures before interpreting aggregates.')
report=['# E18: online self-application helps input shift but fails conflicting-function transfer','',
'This implements a bounded online self-application loop. Every learned arm starts from an E9 bootstrap procedure; student weights, optimizer moments, context and reservoir memory then persist throughout 2,304 batches (73,728 observations). Every 48 batches, online arms can propose a procedure change from an eight-update retrospective gradient and test it on sixteen subsequent batches. All trial predictions in the operational record come from the incumbent until admission. No hindsight replacement with the winning branch occurs.', '',
'| Procedure replicate | Stream seed | Family | Arm | Clean MSE | Admitted / 48 | Recovered segments | Seconds |',
'|---|---|---|---|---:|---:|---:|---:|']
summary=[]
for rep in data['protocol']['controller_replicates']:
    for seed in data['protocol']['stream_seeds']:
        for family in data['protocol']['families']:
            for arm in data['protocol']['arms']:
                record=next(r for r in rs if r['replicate']==rep and r['stream_seed']==seed and r['family']==family and r['arm']==arm);r=record['result']
                assert r['unique_stream_examples']==73728 and len(r['rounds'])==48
                recovered=sum(s['recovery_batches'] is not None for s in r['segments'])
                row=dict(replicate=rep,stream_seed=seed,family=family,arm=arm,clean_mse=r['clean_mse'],accepted=r['accepted'],
                    recovered_segments=recovered,total_segments=len(r['segments']),seconds=r['seconds'],costs=r['costs'],
                    stored_tensor_bytes=r['stored_tensor_bytes'],tracked_peak_tensor_bytes=r['tracked_peak_tensor_bytes'],quarter_clean_mse=r['quarter_clean_mse'])
                summary.append(row);admit=str(r['accepted']) if arm.startswith('online_') else '—'
                report.append(f"| {rep} | {seed} | {family} | {arm} | {r['clean_mse']:.6f} | {admit} | {recovered}/{len(r['segments'])} | {r['seconds']:.3f} |")
report+=['','## Comparisons fixed before the test','']
wins={}
for family in data['protocol']['families']:
    wins[family]={other:0 for other in ('frozen_bootstrap','online_meta_adam','adam_01')}
    for rep in data['protocol']['controller_replicates']:
        for seed in data['protocol']['stream_seeds']:
            pair={r['arm']:r['result'] for r in rs if r['replicate']==rep and r['stream_seed']==seed and r['family']==family}
            for other in wins[family]:wins[family][other]+=pair['online_self_applied']['clean_mse']<pair[other]['clean_mse']
    w=wins[family]
    report.append(f"For {family}, self-application has lower whole-stream error in {w['frozen_bootstrap']}/4 cases against freezing, {w['online_meta_adam']}/4 against conventional meta-updates and {w['adam_01']}/4 against fixed Adam 0.01.")
report+=['',
'Across the four conflicting-function cases, mean self-applied MSE is 0.181483 versus 0.179397 frozen and 0.178784 with conventional meta-updates. The early mean advantage over freezing reverses during the later stream. Pooled recovery is 99/250 segments for self-application, 104/250 frozen, 96/250 conventional meta-updates and 138/250 fixed Adam. These dependent pooled segments are descriptive, not independent trials. The fixed optimizer can have worse overall MSE yet recover more often under the declared criterion.', '',
'Under input shift, mean self-applied MSE is 0.013362 versus 0.014497 frozen and 0.014073 conventional meta-updates. All arms recover in 250/250 pooled segments. The self-applied advantage grows in later quarters here, which supports a narrow online learning benefit. It does not repair the conflicting-function counterexample.', '',
'These four cases per family cross two inherited procedure bootstraps with two fresh worlds, not four independent full-pipeline replications. Segments and proposal rounds are dependent. Accepted proposals establish that the fixed admission rule selected them on subsequent observations; they do not establish a positive long-run improvement rate. Whole-stream, quarter and recovery results must be inspected separately.', '',
'The retention guard uses 128 old observed examples and can overlap replay training. It does not certify rare knowledge or arbitrary future queries. Retrospective meta-training deliberately uses already-observed data; subsequent trial labels do not exist in the proposal inputs. Exact replay of the actual pre-query student parameters is asserted at every scored proposal. The float64 preflight meta-gradient is -0.0638097403777455 versus -0.06380974037767384 by finite difference.', '',
'## Charged work','',
'| Arm | Mean seconds | Student forward examples | Student gradient batches | Controller calls | Meta-gradient calls | Stored tensor bytes (range) | Largest recorded tensor footprint (range) |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in data['protocol']['arms']:
    cases=[r['result'] for r in rs if r['arm']==arm and r['new_work']];c=cases[0]['costs']
    assert all(r['costs']['student_forward_examples']==c['student_forward_examples'] for r in cases)
    footprints=[max(r['stored_tensor_bytes'],r['tracked_peak_tensor_bytes']) for r in cases]
    report.append(f"| {arm} | {st.mean(r['seconds'] for r in cases):.3f} | {c['student_forward_examples']} | {c['student_gradient_batches']} | {c['controller_forward_calls']+c['outer_controller_forward_calls']} | {c['meta_gradient_calls']} | {min(r['stored_tensor_bytes'] for r in cases)}–{max(r['stored_tensor_bytes'] for r in cases)} | {min(footprints)}–{max(footprints)} |")
report+=['',
'Each online case also runs 768 duplicate trial student updates and 384 retrospective student updates, in addition to the 2,304 operational stream updates. Conventional and self-applied online cases receive the same student work and proposal/admission opportunities; self-application adds four outer-controller calls per proposal. The three fixed/frozen/online costs must not be called equal. The historical E9 bootstrap cost is inherited by every learned procedure, while identical Adam cases are reused across procedure replicates (28 newly executed cases, 32 reported combinations). Standalone and execution preflight smoke work is separately retained.', '',
'The reported footprint is the maximum of final stored tensors and the runtime\'s explicitly tracked tensor sample. The original runtime marker omits the final meta-RNG serialization, so its raw baseline value can be lower than final stored state; that raw field is preserved in the JSON. Neither measure includes all autograd intermediates, Python objects or full process RAM. The runner also retains unused outer buffers in fixed/frozen controls. These are partial implementation measurements, not an optimal memory comparison or a complete peak-memory measurement. Checkpoint and full trial-log file sizes are separately recorded. Student-example counts and meta-gradient calls are not complete FLOPs. Local wall times may overlap other research, and no energy or fixed-time comparison was performed.', '',
'Online self-application takes about 1.73 times frozen-procedure time and 3.82 times fixed-Adam time here, plus inherited bootstrap work. Its input-shift accuracy benefit is not a demonstration of cost-matched efficiency. Conventional and self-applied online inference times are close, making their different transfer behavior the more direct mechanism comparison.', '',
'## What this does and does not establish','',
'Actual online procedure proposals, self-application and persistent student state now operate in the same restricted experiment. The measured comparisons determine their value; neither this implementation nor accepted generations prove efficient or open-ended recursive improvement. The controller family, evaluator and memory policy remain fixed.', '',
'A remaining attribution problem is that admission adopts both the changed procedure and its already-adapted student branch. The observed benefit belongs to that package. E18 does not isolate how much persists because the procedure itself became better, independently of the selected student trajectory. A continuation must hold student/optimizer/context/replay state fixed while keeping or reverting procedure weights, then evaluate fresh subsequent observations. Until then, the input-shift gain must not be described as proof of a generally improved learning algorithm.', '',
'Both branches can consume the same labels here because this is full-feedback supervised prediction. In a closed-loop environment different actions change subsequent observations; this experiment does not solve that counterfactual problem. It also does not implement a general reasoning workspace, semantic memory compression, reliable rare-fact recall or architectural novelty. Learned optimizers, replay, model copies and admission have prior art. The overall research goal remains unachieved.']
(HERE/'E18-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e18_summary.json').write_text(json.dumps(dict(rows=summary,wins=wins),indent=2)+'\n')
print('\n'.join(report))
