# E16: temporal action blocking does not repair high-gravity control

The sixty-step planner searches 60, 12 or 6 coefficients, each held constant for 1, 5 or 10 imagined timesteps. All variants replan after every real step and use 64 candidates, two search iterations and eight elites. Each four-episode case uses exactly 6,144,000 model examples. Accurate-model block-one cases reuse E15; learned-model cases restore matching E14 replay checkpoints without further learning.

| Seed | Stage / gravity | Predictor | Block 1 | Block 5 | Block 10 |
|---|---|---|---:|---:|---:|
| 14101 | 0 / 10 | oracle_diagnostic | -245.25 | -207.33 | -176.98 |
| 14101 | 0 / 10 | frozen_learned_replay | -308.09 | -241.91 | -402.55 |
| 14101 | 1 / 30 | oracle_diagnostic | -1612.58 | -1600.98 | -1642.80 |
| 14101 | 1 / 30 | frozen_learned_replay | -1663.76 | -1680.37 | -1735.63 |
| 14101 | 2 / 5 | oracle_diagnostic | -119.37 | -90.17 | -72.84 |
| 14101 | 2 / 5 | frozen_learned_replay | -141.06 | -92.09 | -383.41 |
| 14101 | 3 / 10 | oracle_diagnostic | -177.73 | -179.29 | -179.08 |
| 14101 | 3 / 10 | frozen_learned_replay | -179.19 | -210.67 | -195.95 |
| 14102 | 0 / 10 | oracle_diagnostic | -265.62 | -198.49 | -171.07 |
| 14102 | 0 / 10 | frozen_learned_replay | -847.67 | -993.08 | -799.63 |
| 14102 | 1 / 30 | oracle_diagnostic | -1593.84 | -1586.45 | -1660.38 |
| 14102 | 1 / 30 | frozen_learned_replay | -1647.99 | -1653.42 | -1684.95 |
| 14102 | 2 / 5 | oracle_diagnostic | -144.81 | -137.88 | -138.13 |
| 14102 | 2 / 5 | frozen_learned_replay | -1004.03 | -957.46 | -505.67 |
| 14102 | 3 / 10 | oracle_diagnostic | -278.05 | -234.80 | -210.02 |
| 14102 | 3 / 10 | frozen_learned_replay | -377.79 | -240.73 | -239.40 |

Higher return is better. These are paired development cases, with four episodes each, not sixteen independent replications.

For `oracle_diagnostic`, blocks five and ten improve respectively 7/8 and 5/8 paired stage means against block one.
For `frozen_learned_replay`, blocks five and ten improve respectively 4/8 and 3/8 paired stage means against block one.

At high gravity, five-step blocks yield small accurate-model improvements but returns remain about -1601/-1586; ten-step blocks are worse than block one in both seeds. Both blocked learned-model versions are worse in both high-gravity cases. Thus this simple temporal representation does not repair the failure. This does not rule out feedback policies, better temporal bases, larger populations or other search algorithms.

Several ordinary-gravity accurate-model cases benefit substantially, whereas learned-model outcomes are mixed. The changed proposal family also changes visited states and model-error exposure. These results therefore do not isolate one-step prediction error as the sole cause of any learned-model gap.

## Charged work

| Predictor | Block | Mean seconds per case | New environment steps | Inherited environment steps |
|---|---:|---:|---:|---:|
| oracle_diagnostic | 1 | 8.143 | 0 | 6400 |
| oracle_diagnostic | 5 | 8.325 | 6400 | 0 |
| oracle_diagnostic | 10 | 8.418 | 6400 | 0 |
| frozen_learned_replay | 1 | 9.492 | 6400 | 0 |
| frozen_learned_replay | 5 | 9.432 | 6400 | 0 |
| frozen_learned_replay | 10 | 9.454 | 6400 | 0 |

New scored work is 32,000 real steps and 245,760,000 model examples. The reused E15 cases account for 6,400 real steps and 49,152,000 model examples already paid for there; they are not additional research work. Standalone and execution preflights add synthetic predictor calls but no real environment steps. Learned checkpoint restoration time is retained in each case. Equal model-example counts are not equal FLOPs between formula and neural predictors; compare blocks within a predictor. Timings overlapped a GPU experiment and are descriptive, with no energy or fixed-time claim.

No proposal parameters learn, no memory compresses and no recursive self-improvement occurs. This is a fixed search control. Prior POPLIN implementations already learn policies to guide action/parameter search, so a subsequent learned proposal memory must beat existing ideas and these fixed controls before a novelty or efficiency claim is justified.
