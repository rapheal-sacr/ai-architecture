from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e6/complete.json').read_text())
shutil.copyfile(RUNS/'e6/complete.json',HERE/'results/e6_complete.json')
rows=[]
for case in data['protocol']['cases']:
    for arm in data['protocol']['arms']:
        rs=[r for r in data['records'] if r['case']==case and r['arm']==arm]
        rows.append(dict(case=case,arm=arm,mse=st.mean(r['mse'] for r in rs),seconds=st.mean(r['wall_seconds'] for r in rs),
            late_mse=st.mean(st.mean(b['mse'] for b in r['blocks'][-10:]) if case=='external' else
                st.mean(s['prequential_mse'] for s in r['segments'][-8:]) for r in rs),
            merges=st.mean(r['costs'].get('merges',0) for r in rs),modules=st.mean(r['costs']['module_count'] for r in rs),
            tensor_bytes=st.mean(r['costs']['stored_tensor_bytes'] for r in rs),
            peak_tensor_bytes=max(r['costs'].get('peak_tensor_bytes',r['costs']['stored_tensor_bytes']) for r in rs),
            forward_examples=st.mean(r['costs']['forward_examples'] for r in rs),
            training_examples=st.mean(r['costs']['training_examples'] for r in rs)))
(HERE/'results/e6_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E6: local replay and guarded sharing','',
'The frozen screen adds local observed-example replay to E5 active-first isolation, then ablates a sample-based merge rule. All arms receive the same stream per seed; the external case uses the pinned official generator.','',
'| Case | Arm | MSE | Late MSE | Seconds | Modules | Merges | Tensor bytes |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['case']} | {r['arm']} | {r['mse']:.6f} | {r['late_mse']:.6f} | {r['seconds']:.2f} | {r['modules']:.1f} | {r['merges']:.1f} | {r['tensor_bytes']:.0f} |")
report+=['','Late MSE is the final quarter of abrupt-context segments or the final ten external blocks. These are different declared windows; compare arms within a case.','']
for case in data['protocol']['cases']:
    rr={r['arm']:r for r in rows if r['case']==case};g=rr['guarded_sharing'];ref=rr['context_replay'];local=rr['local_replay_isolation']
    report.append(f"For {case}, guarded sharing changes error by {100*(g['mse']/ref['mse']-1):+.1f}% and runtime by {100*(g['seconds']/ref['seconds']-1):+.1f}% versus context replay. Relative to local replay without merging, its error changes {100*(g['mse']/local['mse']-1):+.1f}%.")
report += ['',
'On abrupt contexts, the sample test admits no merges in these runs. Rehearsal therefore adds work without reducing module storage. The protected architecture still requires separate models for these conflicting conditional functions.','',
'In the external stream, a successful anchor check is only evidence about the retained sample. It cannot establish preserved performance on omitted past inputs. Compare actual prequential and late performance above; the count of successful merges is not a count of useful improvements.','',
'All replay observations, merge checks and temporary model state are charged. Capacity hits inherited from E2 count commit requests at a full bank; a later merge can recover capacity, so they are not a count of permanently lost tasks. Tensor storage excludes Python overhead, gradients and transient activations. Timing remains indicative local CPU time.','',
'No rare-fact recovery or distribution-expansion assay has been run for these merges. This is not verified semantic compression or a complete memory solution. No general-reasoning or closed-loop agent result is supplied by E6.','',
'A train-only preflight checked replay and a forced full-capacity merge with deliberately permissive test settings. The scored protocol retains its frozen, stricter thresholds. Implementation review added full-capacity merge checks before the protocol was committed and before any scored E6 run.']
(HERE/'E6-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
