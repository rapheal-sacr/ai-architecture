from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
d=json.loads((RUNS/'e20/complete.json').read_text());rs=d['records'];assert len(rs)==32
assert not any(r['failed'] for r in rs)
shutil.copyfile(RUNS/'e20/complete.json',HERE/'results/e20_complete.json')
shutil.copyfile(RUNS/'e20-preflight/preflight.json',HERE/'results/e20_preflight.json')
report=['# E20: context information versus shared model capacity','',
'A two-by-two diagnostic crosses inferred versus explicitly privileged context with hidden width 16 versus 64. All cells use the same persistent learner, replay budget, fixed Adam rate 0.01 and 4,096-batch streams. Four fresh worlds per family are paired across cells. The oracle receives the true hidden regime before prediction, encoded one-hot within the same 18-component context interface. The real inferred path receives no regime labels.', '',
'| Seed | Family | Inferred, width 16 | Inferred, width 64 | Oracle, width 16 | Oracle, width 64 |',
'|---|---|---:|---:|---:|---:|']
for seed in d['protocol']['stream_seeds']:
    for family in d['protocol']['families']:
        pair={(r['context_mode'],r['hidden_width']):r for r in rs if r['seed']==seed and r['family']==family}
        report.append(f"| {seed} | {family} | " + ' | '.join(f"{pair[(m,w)]['clean_mse']:.6f}" for m,w in [('inferred_ema',16),('inferred_ema',64),('oracle_onehot_diagnostic',16),('oracle_onehot_diagnostic',64)])+' |')
report+=['','## Recovery and measured work','',
'| Family | Context | Width | Mean MSE | Recovered segments | Parameters | Stored tensor bytes | Mean seconds |',
'|---|---|---:|---:|---:|---:|---:|---:|']
summary=[]
for family in d['protocol']['families']:
    for mode in d['protocol']['context_modes']:
        for width in d['protocol']['hidden_widths']:
            cases=[r for r in rs if r['family']==family and r['context_mode']==mode and r['hidden_width']==width]
            row=dict(family=family,context_mode=mode,width=width,mean_mse=st.mean(r['clean_mse'] for r in cases),
                recovered_segments=sum(r['recovered_segments'] for r in cases),total_segments=sum(r['total_segments'] for r in cases),
                mean_seconds=st.mean(r['seconds'] for r in cases),costs=cases[0]['costs']);summary.append(row)
            report.append(f"| {family} | {mode} | {width} | {row['mean_mse']:.6f} | {row['recovered_segments']}/{row['total_segments']} | {row['costs']['student_parameters']} | {row['costs']['stored_tensor_bytes']} | {row['mean_seconds']:.3f} |")
report+=['','## Interpretation boundary','']
for family in d['protocol']['families']:
    pair={(r['context_mode'],r['width']):r for r in summary if r['family']==family}
    initial=pair[('inferred_ema',16)]['mean_mse']
    oracle_gain=1-pair[('oracle_onehot_diagnostic',16)]['mean_mse']/initial
    width_gain=1-pair[('inferred_ema',64)]['mean_mse']/initial
    report.append(f"For {family}, supplying oracle context at width 16 changes mean MSE by {-oracle_gain:.1%}; widening under inferred context changes it by {-width_gain:.1%}.")
report+=['',
'The oracle intervention changes both information and its encoding. A large improvement locates a bottleneck in the current inference/assignment/interface package; it does not measure an information-theoretic identification lower bound. In particular, E17 stores the preceding inferred context with replay examples, whereas the oracle stores the correct context. Both action-time identification and assigning observations to retained conditional models can contribute. The test does not distinguish those two effects.', '',
'The wider model uses more weights and arithmetic. Equal student-example counts are not equal FLOPs. Each cell uses 393,184 student forward examples and 4,096 gradient batches; the cost table charges stored weights, optimizer moments, replay, context and RNG tensors. Dataset-generation time is separately recorded. Timings overlap E19 and are descriptive, with no full process RAM or energy measurement.', '',
'Recovery uses the unchanged E17 evaluation-only criterion and retains every censored full segment. Four worlds are independent seed choices within each family, while the paired conditions and within-world segments are dependent. No significance claim or optimal learning-rate tuning is made.', '',
'The oracle is never supplied to a deployed candidate and cannot count as autonomous context inference. This diagnostic does not establish memory compression, general reasoning, novel architecture design, reliable recursive improvement or efficient long-horizon control. It constrains the next architecture: a stronger inference/memory-assignment mechanism must earn its gain from actual observed evidence and repay its cost.']
(HERE/'E20-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e20_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'.join(report))
