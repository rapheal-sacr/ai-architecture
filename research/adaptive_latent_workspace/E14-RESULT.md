# E14: neural dynamics and planning help selectively; protected memory does not engage

The learner acquires a neural transition/reward model from actual observations and uses it to search action sequences. Model weights, optimizer state and bounded replay persist across hidden gravity changes. The planner receives observations and model predictions only; it cannot access simulator equations, private state or the gravity parameter.

Twenty-step replay planning improves actual frozen-policy return over the one-step version in seven of eight paired seed/stage settings. It remains poor at high gravity, and one initialization is weak in the initial normal-gravity evaluation. Guarded sharing never creates a temporary model, grows a module or merges: its thresholds do not activate those mechanisms in this normalized target space. Its outcomes therefore cannot support a claim that protection or consolidation caused a gain.

| Seed | Arm | Stage / gravity | Mean training episode return | Frozen evaluation return | Prequential target MSE |
|---|---|---|---:|---:|---:|
| 14101 | random | 0 / 10 | -1317.09 | -1381.88 | — |
| 14101 | random | 1 / 30 | -1700.40 | -1682.32 | — |
| 14101 | random | 2 / 5 | -1110.65 | -1133.79 | — |
| 14101 | random | 3 / 10 | -1156.73 | -1329.93 | — |
| 14101 | replay_mpc1 | 0 / 10 | -1232.83 | -1365.32 | 0.004075 |
| 14101 | replay_mpc1 | 1 / 30 | -1657.34 | -1603.55 | 0.002449 |
| 14101 | replay_mpc1 | 2 / 5 | -952.81 | -1232.84 | 0.000653 |
| 14101 | replay_mpc1 | 3 / 10 | -1141.48 | -1511.18 | 0.000320 |
| 14101 | replay_mpc20 | 0 / 10 | -692.30 | -216.83 | 0.002950 |
| 14101 | replay_mpc20 | 1 / 30 | -1671.57 | -1662.40 | 0.002495 |
| 14101 | replay_mpc20 | 2 / 5 | -130.13 | -90.18 | 0.000312 |
| 14101 | replay_mpc20 | 3 / 10 | -316.84 | -183.84 | 0.000300 |
| 14101 | context_mpc20 | 0 / 10 | -419.24 | -283.06 | 0.001974 |
| 14101 | context_mpc20 | 1 / 30 | -1693.86 | -1678.60 | 0.005885 |
| 14101 | context_mpc20 | 2 / 5 | -544.26 | -569.18 | 0.002408 |
| 14101 | context_mpc20 | 3 / 10 | -335.81 | -178.06 | 0.000990 |
| 14101 | guarded_mpc20 | 0 / 10 | -591.71 | -896.15 | 0.003162 |
| 14101 | guarded_mpc20 | 1 / 30 | -1668.65 | -1663.85 | 0.003268 |
| 14101 | guarded_mpc20 | 2 / 5 | -460.67 | -266.70 | 0.000787 |
| 14101 | guarded_mpc20 | 3 / 10 | -289.86 | -452.31 | 0.000258 |
| 14102 | random | 0 / 10 | -1292.45 | -1385.77 | — |
| 14102 | random | 1 / 30 | -1688.95 | -1676.76 | — |
| 14102 | random | 2 / 5 | -1018.04 | -1374.02 | — |
| 14102 | random | 3 / 10 | -1239.06 | -1448.17 | — |
| 14102 | replay_mpc1 | 0 / 10 | -1299.68 | -1401.86 | 0.005881 |
| 14102 | replay_mpc1 | 1 / 30 | -1690.42 | -1638.72 | 0.004223 |
| 14102 | replay_mpc1 | 2 / 5 | -1090.01 | -1618.89 | 0.000720 |
| 14102 | replay_mpc1 | 3 / 10 | -1288.28 | -1823.60 | 0.000269 |
| 14102 | replay_mpc20 | 0 / 10 | -610.26 | -1323.46 | 0.002809 |
| 14102 | replay_mpc20 | 1 / 30 | -1666.58 | -1633.39 | 0.002127 |
| 14102 | replay_mpc20 | 2 / 5 | -182.45 | -545.29 | 0.000277 |
| 14102 | replay_mpc20 | 3 / 10 | -451.68 | -308.81 | 0.000349 |
| 14102 | context_mpc20 | 0 / 10 | -363.43 | -440.58 | 0.001637 |
| 14102 | context_mpc20 | 1 / 30 | -1682.27 | -1660.66 | 0.004951 |
| 14102 | context_mpc20 | 2 / 5 | -405.16 | -360.59 | 0.001251 |
| 14102 | context_mpc20 | 3 / 10 | -245.07 | -478.46 | 0.000668 |
| 14102 | guarded_mpc20 | 0 / 10 | -658.18 | -376.79 | 0.003355 |
| 14102 | guarded_mpc20 | 1 / 30 | -1679.39 | -1635.71 | 0.002898 |
| 14102 | guarded_mpc20 | 2 / 5 | -246.91 | -594.31 | 0.000353 |
| 14102 | guarded_mpc20 | 3 / 10 | -530.71 | -870.48 | 0.000384 |

Higher return (closer to zero) is better. Each frozen evaluation is only four 200-step episodes, with two independently trained seeds; stage summaries share trained state and are not independent replications. Prequential MSE mixes normalized angular/velocity changes and reward, so it is not a direct control-performance metric.

## Complete costs

| Seed | Arm | Total seconds | Forward examples | Parameters | Stored tensor bytes | Modules | Merges |
|---|---|---:|---:|---:|---:|---:|---:|
| 14101 | random | 1.19 | 0 | 0 | 0 | 0 | 0 |
| 14101 | replay_mpc1 | 8.86 | 2054908 | 4675 | 113528 | 1 | 0 |
| 14101 | replay_mpc20 | 67.27 | 40344316 | 4675 | 113528 | 1 | 0 |
| 14101 | context_mpc20 | 72.45 | 40344316 | 5635 | 247928 | 1 | 0 |
| 14101 | guarded_mpc20 | 68.16 | 40357116 | 4675 | 59708 | 1 | 0 |
| 14102 | random | 1.15 | 0 | 0 | 0 | 0 | 0 |
| 14102 | replay_mpc1 | 8.83 | 2054908 | 4675 | 113528 | 1 | 0 |
| 14102 | replay_mpc20 | 66.95 | 40344316 | 4675 | 113528 | 1 | 0 |
| 14102 | context_mpc20 | 72.34 | 40344316 | 5635 | 247928 | 1 | 0 |
| 14102 | guarded_mpc20 | 68.50 | 40357116 | 4675 | 59708 | 1 | 0 |

Twenty-step versus one-step replay uses 19.63× forward examples and 7.59× measured time. Both receive 12,800 training environment steps per seed and 3,200 frozen evaluation steps; learned arms also receive 640 diagnostic probe steps that never train the model.

The guarded arm uses fewer replay slots (128 local anchors versus the ordinary 2,048-example reservoir). Its smaller stored tensor count reflects that budget difference, not demonstrated semantic compression. The acting learner cannot read the archived per-stage research checkpoints. Costs count imagined, training, routing and evaluation model calls; counts are examples, not complete FLOPs. GPU research runs overlap this CPU pilot, so no isolated-hardware or fixed-time superiority claim follows.

## Multi-step probes

Four new initial states per stage receive the same fixed random action sequence in the real environment and the frozen learned model. One-step targets use each real current observation; the imagined trajectory accumulates its own predictions. These probe actions differ from the planner-selected control distribution, so probe error cannot by itself explain a control failure. Receding-horizon replanning also corrects state using real observations each step.

| Seed | Arm | Stage | Mean one-step target MSE | State MSE at 20 steps | State MSE at 40 steps | Predicted minus actual probe return |
|---|---|---:|---:|---:|---:|---:|
| 14101 | replay_mpc1 | 0 | 0.001076 | 0.020573 | 0.058284 | -34.29 |
| 14101 | replay_mpc1 | 1 | 0.001837 | 0.459056 | 0.408705 | 31.31 |
| 14101 | replay_mpc1 | 2 | 0.000418 | 0.088627 | 0.107634 | -25.51 |
| 14101 | replay_mpc1 | 3 | 0.000384 | 0.083922 | 0.547056 | -32.61 |
| 14101 | replay_mpc20 | 0 | 0.005742 | 0.006738 | 0.006977 | -82.76 |
| 14101 | replay_mpc20 | 1 | 0.001296 | 0.047885 | 0.076660 | -1.13 |
| 14101 | replay_mpc20 | 2 | 0.002647 | 0.559850 | 0.143223 | -40.67 |
| 14101 | replay_mpc20 | 3 | 0.001578 | 0.792465 | 1.057240 | -99.35 |
| 14101 | context_mpc20 | 0 | 0.002498 | 0.001129 | 0.007053 | -32.97 |
| 14101 | context_mpc20 | 1 | 0.007631 | 0.277950 | 0.839164 | -0.75 |
| 14101 | context_mpc20 | 2 | 0.039801 | 0.275794 | 0.106692 | -208.79 |
| 14101 | context_mpc20 | 3 | 0.002578 | 0.112818 | 0.407573 | -53.57 |
| 14101 | guarded_mpc20 | 0 | 0.007791 | 0.003770 | 0.004828 | -91.28 |
| 14101 | guarded_mpc20 | 1 | 0.001274 | 0.018794 | 0.033871 | 3.42 |
| 14101 | guarded_mpc20 | 2 | 0.002439 | 0.387685 | 0.244017 | -32.95 |
| 14101 | guarded_mpc20 | 3 | 0.001379 | 0.229755 | 0.871942 | -66.42 |
| 14102 | replay_mpc1 | 0 | 0.004042 | 0.050136 | 0.034375 | -44.78 |
| 14102 | replay_mpc1 | 1 | 0.001335 | 0.291705 | 0.547304 | 23.46 |
| 14102 | replay_mpc1 | 2 | 0.000485 | 0.236507 | 0.690870 | 27.46 |
| 14102 | replay_mpc1 | 3 | 0.000450 | 0.170760 | 0.780372 | -39.17 |
| 14102 | replay_mpc20 | 0 | 0.029842 | 0.059479 | 0.300208 | -139.12 |
| 14102 | replay_mpc20 | 1 | 0.001131 | 0.238281 | 0.227651 | -8.97 |
| 14102 | replay_mpc20 | 2 | 0.003042 | 0.675354 | 0.305116 | -76.83 |
| 14102 | replay_mpc20 | 3 | 0.002019 | 0.505334 | 0.714513 | -102.46 |
| 14102 | context_mpc20 | 0 | 0.011976 | 0.100818 | 0.029879 | -102.55 |
| 14102 | context_mpc20 | 1 | 0.019929 | 0.021101 | 0.057072 | -153.03 |
| 14102 | context_mpc20 | 2 | 0.003900 | 0.330004 | 0.593738 | 0.50 |
| 14102 | context_mpc20 | 3 | 0.003965 | 0.189842 | 0.663995 | 4.54 |
| 14102 | guarded_mpc20 | 0 | 0.007619 | 0.021387 | 0.049585 | -78.55 |
| 14102 | guarded_mpc20 | 1 | 0.003206 | 0.251480 | 0.407404 | -32.26 |
| 14102 | guarded_mpc20 | 2 | 0.003135 | 0.651722 | 0.271587 | -85.27 |
| 14102 | guarded_mpc20 | 3 | 0.000623 | 0.387295 | 0.449125 | -53.20 |

E15 separately tests the same search with an explicitly privileged accurate dynamics/reward model. This distinguishes failures of learned prediction from failures that persist in the planner/search budget. That diagnostic reference is not available to this agent.

The prototype now contains actual neural world-model learning and planning with persistent replay. It does not implement the full recurrent/latent architecture, general reasoning, a learned update procedure, recursive self-improvement or reliable memory compression. Episodes last 200 steps and search spans twenty; neither is a demonstration of very long-horizon competence. Strong model-free and uncertainty-aware model-based comparisons remain missing.
