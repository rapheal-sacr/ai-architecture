from pathlib import Path
import json, shutil, statistics as st

HERE = Path(__file__).resolve().parent
RUNS = Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs')
data = json.loads((RUNS/'e16/complete.json').read_text())
assert len(data['records']) == 48
shutil.copyfile(RUNS/'e16/complete.json', HERE/'results/e16_complete.json')
shutil.copyfile(RUNS/'e16-preflight/preflight.json', HERE/'results/e16_preflight.json')
report = ['# E16: temporal action blocking does not repair high-gravity control', '',
    'The sixty-step planner searches 60, 12 or 6 coefficients, each held constant for 1, 5 or 10 imagined timesteps. All variants replan after every real step and use 64 candidates, two search iterations and eight elites. Each four-episode case uses exactly 6,144,000 model examples. Accurate-model block-one cases reuse E15; learned-model cases restore matching E14 replay checkpoints without further learning.', '',
    '| Seed | Stage / gravity | Predictor | Block 1 | Block 5 | Block 10 |',
    '|---|---|---|---:|---:|---:|']
rows = []
for seed in data['protocol']['seeds']:
    for stage, gravity in enumerate(data['protocol']['gravity_sequence']):
        for mode in data['protocol']['modes']:
            pair = {r['block']: r for r in data['records'] if r['seed'] == seed and r['stage'] == stage and r['mode'] == mode}
            assert set(pair) == {1, 5, 10}
            for r in pair.values():
                assert r['new_model_examples'] + r['inherited_model_examples'] == 6144000
            row = dict(seed=seed, stage=stage, gravity=gravity, mode=mode,
                       returns={str(b): pair[b]['mean_return'] for b in (1, 5, 10)})
            rows.append(row)
            report.append(f"| {seed} | {stage} / {gravity:g} | {mode} | " + ' | '.join(f"{pair[b]['mean_return']:.2f}" for b in (1, 5, 10)) + ' |')
report += ['', 'Higher return is better. These are paired development cases, with four episodes each, not sixteen independent replications.', '']
for mode in data['protocol']['modes']:
    rs = [r for r in rows if r['mode'] == mode]
    wins = {b: sum(r['returns'][str(b)] > r['returns']['1'] for r in rs) for b in (5, 10)}
    report.append(f"For `{mode}`, blocks five and ten improve respectively {wins[5]}/8 and {wins[10]}/8 paired stage means against block one.")
report += ['',
    'At high gravity, five-step blocks yield small accurate-model improvements but returns remain about -1601/-1586; ten-step blocks are worse than block one in both seeds. Both blocked learned-model versions are worse in both high-gravity cases. Thus this simple temporal representation does not repair the failure. This does not rule out feedback policies, better temporal bases, larger populations or other search algorithms.', '',
    'Several ordinary-gravity accurate-model cases benefit substantially, whereas learned-model outcomes are mixed. The changed proposal family also changes visited states and model-error exposure. These results therefore do not isolate one-step prediction error as the sole cause of any learned-model gap.', '',
    '## Charged work', '',
    '| Predictor | Block | Mean seconds per case | New environment steps | Inherited environment steps |',
    '|---|---:|---:|---:|---:|']
for mode in data['protocol']['modes']:
    for b in (1, 5, 10):
        rs = [r for r in data['records'] if r['mode'] == mode and r['block'] == b]
        report.append(f"| {mode} | {b} | {st.mean(r['seconds'] for r in rs):.3f} | {sum(r['new_environment_steps'] for r in rs)} | {sum(r['inherited_environment_steps'] for r in rs)} |")
report += ['',
    'New scored work is 32,000 real steps and 245,760,000 model examples. The reused E15 cases account for 6,400 real steps and 49,152,000 model examples already paid for there; they are not additional research work. Standalone and execution preflights add synthetic predictor calls but no real environment steps. Learned checkpoint restoration time is retained in each case. Equal model-example counts are not equal FLOPs between formula and neural predictors; compare blocks within a predictor. Timings overlapped a GPU experiment and are descriptive, with no energy or fixed-time claim.', '',
    'No proposal parameters learn, no memory compresses and no recursive self-improvement occurs. This is a fixed search control. Prior POPLIN implementations already learn policies to guide action/parameter search, so a subsequent learned proposal memory must beat existing ideas and these fixed controls before a novelty or efficiency claim is justified.']
(HERE/'E16-RESULT.md').write_text('\n'.join(report) + '\n')
(HERE/'results/e16_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('\n'.join(report))
