from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data=json.loads((RUNS/'e13/complete.json').read_text())
shutil.copyfile(RUNS/'e13/complete.json',HERE/'results/e13_complete.json')
report=['# E13: does consolidation repay its cost over five million observations?','',
'The original E6 first million observations and teacher are preserved exactly. The extension continues the same declared Markov slow-bit flips and independent fast bits with a separate seed. Recalling the bulk generator at a larger length would change the original prefix and is deliberately not used. Each learner reprocesses the prefix from the original initialization; that work is charged.','',
'| Seed | Arm | Exact prefix error / counters | Full-stream MSE | Final ten blocks MSE | Seconds | Forward examples | Stored tensor bytes | Merges |',
'|---|---|---|---:|---:|---:|---:|---:|---:|']
rows=[];windows=[]
for r in data['records']:
    late=r['blocks'][-10:];late_mse=sum(b['mse']*b['observations'] for b in late)/sum(b['observations'] for b in late)
    row=dict(seed=r['seed'],arm=r['arm'],mse=r['mse'],late_mse=late_mse,seconds=r['wall_seconds'],costs=r['costs'],prefix_check=r['prefix_check']);rows.append(row)
    c=r['costs'];check=r['prefix_check'];report.append(f"| {r['seed']} | {r['arm']} | {check['mse_equal']} / {check['costs_equal']} | {r['mse']:.6f} | {late_mse:.6f} | {r['wall_seconds']:.1f} | {c['forward_examples']} | {c['stored_tensor_bytes']} | {c.get('merges',0)} |")
    # Interior million boundaries need not coincide with emitted blocks. Use
    # the last complete block before each boundary and expose exact endpoints.
    start=0
    for target in range(1000000,data['protocol']['total_observations']+1,1000000):
        end=max(b['end_observation'] for b in r['blocks'] if b['end_observation']<=target)
        bs=[b for b in r['blocks'] if start<b['end_observation']<=end]
        windows.append(dict(seed=r['seed'],arm=r['arm'],start_observation=start,end_observation=end,
            mse=sum(b['mse']*b['observations'] for b in bs)/sum(b['observations'] for b in bs),
            cumulative_mse=sum(b['mse']*b['observations'] for b in r['blocks'] if b['end_observation']<=end)/end,
            cumulative_seconds=bs[-1]['elapsed_seconds'],cumulative_forward_examples=bs[-1]['forward_examples']))
        start=end
report+=['','## Error by observed block interval','',
'Interior intervals are approximately one million observations because the recorded blocks do not necessarily end exactly on that boundary. The table gives actual endpoints, with no interpolation. The original million and final five million are exact boundaries.','',
'| Seed | Arm | Start (exclusive) | End (inclusive) | Interval MSE | Cumulative MSE | Cumulative seconds |','|---|---|---:|---:|---:|---:|---:|']
for r in windows:report.append(f"| {r['seed']} | {r['arm']} | {r['start_observation']} | {r['end_observation']} | {r['mse']:.6f} | {r['cumulative_mse']:.6f} | {r['cumulative_seconds']:.1f} |")
report+=['','## Cost and accuracy comparison','']
for seed in data['protocol']['seeds']:
    pair={r['arm']:r for r in rows if r['seed']==seed};a=pair['guarded_sharing'];b=pair['context_replay']
    report.append(f"Seed {seed}: guarded sharing / context replay ratios are {a['mse']/b['mse']:.4f} for cumulative MSE, {a['late_mse']/b['late_mse']:.4f} for late MSE, {a['costs']['forward_examples']/b['costs']['forward_examples']:.4f} for forward examples, and {a['seconds']/b['seconds']:.4f} for elapsed time.")
report+=['',
'A lower late error is not retrospective repayment of earlier error or work. Accuracy and work are reported separately rather than combined with an invented exchange rate. No unobserved break-even point is treated as measured.','',
'The final ten blocks can be shorter than exactly 100,000 observations; weighted means use the actual recorded sizes. Complete blocks retain all cost counters, including model growth, updates, replay and merge-related work. CPU work overlaps independent GPU experiments, so operation counts are stronger evidence than local wall-time comparisons.','',
'This continues two already studied streams. It is not a fresh external replication, a rare-fact preservation test, semantic compression proof, closed-loop reasoning agent or recursive improvement result. Passing anchor checks only establishes those sampled checks.']
(HERE/'E13-RESULT.md').write_text('\n'.join(report)+'\n')
(HERE/'results/e13_summary.json').write_text(json.dumps(dict(rows=rows,windows=windows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
