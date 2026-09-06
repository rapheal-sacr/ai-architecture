from pathlib import Path
import json,shutil,statistics as st
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e15/complete.json').read_text());learned=json.loads((RUNS/'e14/complete.json').read_text())
shutil.copyfile(RUNS/'e15/complete.json',HERE/'results/e15_complete.json')
report=['# E15: accurate dynamics do not repair the high-gravity search failure','',
'The same cross-entropy planner receives an explicitly privileged predictor using the known transition and reward equations. A separate preflight matches real environment transition targets within 1.20e-7 maximum normalized error over 100 random-action steps at each gravity. This predictor is a diagnostic reference; E14 learners do not receive it.','',
'| Seed | Stage / gravity | Learned replay, h=20 | Accurate model, h=1 | Accurate model, h=20 | Accurate model, h=60 |','|---|---|---:|---:|---:|---:|']
rows=[]
for seed in data['protocol']['seeds']:
    learning=next(r for r in learned['records'] if r['seed']==seed and r['arm']=='replay_mpc20')
    for stage,gravity in enumerate(data['protocol']['gravity_sequence']):
        pair={r['horizon']:r for r in data['records'] if r['seed']==seed and r['stage']==stage}
        row=dict(seed=seed,stage=stage,gravity=gravity,learned_h20=learning['stages'][stage]['evaluation']['mean_return'],oracle_h1=pair[1]['mean_return'],oracle_h20=pair[20]['mean_return'],oracle_h60=pair[60]['mean_return']);rows.append(row)
        report.append(f"| {seed} | {stage} / {gravity:g} | {row['learned_h20']:.2f} | {row['oracle_h1']:.2f} | {row['oracle_h20']:.2f} | {row['oracle_h60']:.2f} |")
report+=['',
'Higher return, closer to zero, is better. At gravity 30 the accurate-model twenty-step planner remains poor (about -1603/-1586); sixty steps remains poor (about -1613/-1594). Thus learned prediction error is not the sole explanation of the high-gravity failure. The search, action proposal family, horizon and objective still impose a limitation even when the tested dynamics/reward are accurate. This does not prove that the control task is impossible.','',
'The accurate-model sixty-step planner exceeds the twenty-step return in only one of eight paired seed/stage settings. It uses exactly three times the model examples with the same population and two CEM iterations. Increasing the action-sequence dimension while keeping the search population fixed can make search harder; this is not a general rejection of longer planning.','',
'At ordinary gravity, the first learned-model seed can approach the accurate-model return, whereas the second has a large initial gap. Learning/model error and search limitations can coexist; identifying one does not remove the other.','',
'## Cost and scope','',
'| Horizon | Mean seconds per four-episode evaluation | Model examples per evaluation | Actual environment steps per evaluation |','|---|---:|---:|---:|']
for h in data['protocol']['horizons']:
    rs=[r for r in data['records'] if r['horizon']==h];report.append(f"| {h} | {st.mean(r['seconds'] for r in rs):.3f} | {rs[0]['oracle_model_examples']} | {rs[0]['actual_environment_steps']} |")
report+=['',
'The scored diagnostic uses 19,200 real environment steps. Its standalone preflight and repeated execution preflight each use 300 further steps; their exact wall time was not retained. Formula evaluations are cheaper than a learned neural model and cannot be advertised as learned-model efficiency. GPU experiments overlapped the CPU work, so timings are local descriptive measurements.','',
'This uses existing development seed/family choices and four evaluation episodes per setting. No parameters learn, no memory compresses, and no self-improvement occurs in the oracle reference. Its role is to falsify the explanation that more accurate memory or a larger dynamics model alone would solve E14. A stronger action-proposal/search method must be tested before that architectural addition is justified.']
(HERE/'E15-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e15_summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
