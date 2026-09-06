from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e3/complete.json').read_text())
shutil.copyfile(RUNS/'e3/complete.json',HERE/'results/e3_complete.json')
rows=[]
for case in data['protocol']['cases']:
    for arm in data['protocol']['arms']:
        rs=[r for r in data['records'] if r['case']==case and r['arm']==arm]
        rows.append(dict(case=case,arm=arm,mse=st.mean(s['prequential_mse'] for r in rs for s in r['segments']),
            clean_mse=st.mean(s['clean_prequential_mse'] for r in rs for s in r['segments']),
            seconds=st.mean(r['wall_seconds'] for r in rs),modules=st.mean(r['costs']['module_count'] for r in rs),
            capacity_hits=st.mean(r['capacity_hits'] for r in rs),
            last_eighth_mse=st.mean(s['prequential_mse'] for r in rs for s in r['segments'][-max(1,len(r['segments'])//8):]),
            peak_tensor_bytes=max(r['costs'].get('peak_tensor_bytes',r['costs']['stored_tensor_bytes']) for r in rs)))
(HERE/'results/e3_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
report=['# E3: stress results and boundaries','',
'All five frozen stress cases completed on two seeds each. This is an exploratory screen, not a statistically powered generality claim. Prediction occurs before each target is observed. Settings are unchanged from E2.','',
'| Case | Arm | MSE | Final eighth MSE | Seconds | Modules | Capacity hits | Peak tensor bytes |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['case']} | {r['arm']} | {r['mse']:.6f} | {r['last_eighth_mse']:.6f} | {r['seconds']:.2f} | {r['modules']:.1f} | {r['capacity_hits']:.0f} | {r['peak_tensor_bytes']} |")
report += ['',
'**Supported in this assay:** the four-context million-observation stream retains a substantial prediction advantage. Noise and short context lifetimes do not reverse the whole-stream error advantage on these two seeds. This establishes neither adversarial-noise robustness nor rapid reliable identification of all short contexts.','',
'**Failed general cost claim:** under gradual interpolation the candidate creates one module and makes exactly the same scored predictions as online learning, with approximately 23% extra runtime. The novelty gate treats the changing function as one adaptable context. It supplies no demonstrable retained history of intermediate functions.','',
'**Unsolved capacity:** sixteen contexts exhaust eight slots. Thousands of subsequent updates satisfy the commit condition but cannot acquire a persistent slot. Lower whole-stream prediction error comes with approximately 45% more runtime than context replay and twice online runtime. In the final eighth, the candidate is worse than context replay (0.06142 versus 0.05770 MSE), so its average advantage does not establish a sustainable long-run advantage. Capacity hits count updates, not distinct rejected tasks. An unbounded archive would evade this test rather than solve its resource constraint.','',
'**Cost caveat:** forward-example counts include all expert scoring, but are not FLOPs when network widths differ. Stored tensor bytes omit Python containers, gradients and transient forward/backward activations. Peak tensor bytes include the temporary model and optimizer, not peak process RAM. Timing includes the stream generator and learner work, excludes held-out diagnostics, and is an indicative local CPU measurement.','',
'**Still unmeasured:** episodic compression with rare-fact recovery, amortized module consolidation, shared learning between related contexts, general reasoning and closed-loop long-horizon performance. A million supervised observations is not a million-step successful autonomous task.']
(HERE/'E3-RESULT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(rows,indent=2))
