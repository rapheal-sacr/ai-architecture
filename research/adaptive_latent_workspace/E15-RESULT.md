# E15: accurate dynamics do not repair the high-gravity search failure

The same cross-entropy planner receives an explicitly privileged predictor using the known transition and reward equations. A separate preflight matches real environment transition targets within 1.20e-7 maximum normalized error over 100 random-action steps at each gravity. This predictor is a diagnostic reference; E14 learners do not receive it.

| Seed | Stage / gravity | Learned replay, h=20 | Accurate model, h=1 | Accurate model, h=20 | Accurate model, h=60 |
|---|---|---:|---:|---:|---:|
| 14101 | 0 / 10 | -216.83 | -1484.67 | -217.66 | -245.25 |
| 14101 | 1 / 30 | -1662.40 | -1687.39 | -1603.34 | -1612.58 |
| 14101 | 2 / 5 | -90.18 | -1038.69 | -103.50 | -119.37 |
| 14101 | 3 / 10 | -183.84 | -1321.51 | -178.81 | -177.73 |
| 14102 | 0 / 10 | -1323.46 | -1415.78 | -206.08 | -265.62 |
| 14102 | 1 / 30 | -1633.39 | -1690.95 | -1585.81 | -1593.84 |
| 14102 | 2 / 5 | -545.29 | -1324.07 | -134.42 | -144.81 |
| 14102 | 3 / 10 | -308.81 | -1492.99 | -217.56 | -278.05 |

Higher return, closer to zero, is better. At gravity 30 the accurate-model twenty-step planner remains poor (about -1603/-1586); sixty steps remains poor (about -1613/-1594). Thus learned prediction error is not the sole explanation of the high-gravity failure. The search, action proposal family, horizon and objective still impose a limitation even when the tested dynamics/reward are accurate. This does not prove that the control task is impossible.

The accurate-model sixty-step planner exceeds the twenty-step return in only one of eight paired seed/stage settings. It uses exactly three times the model examples with the same population and two CEM iterations. Increasing the action-sequence dimension while keeping the search population fixed can make search harder; this is not a general rejection of longer planning.

At ordinary gravity, the first learned-model seed can approach the accurate-model return, whereas the second has a large initial gap. Learning/model error and search limitations can coexist; identifying one does not remove the other.

## Cost and scope

| Horizon | Mean seconds per four-episode evaluation | Model examples per evaluation | Actual environment steps per evaluation |
|---|---:|---:|---:|
| 1 | 0.236 | 102400 | 800 |
| 20 | 2.786 | 2048000 | 800 |
| 60 | 8.143 | 6144000 | 800 |

The scored diagnostic uses 19,200 real environment steps. Its standalone preflight and repeated execution preflight each use 300 further steps; their exact wall time was not retained. Formula evaluations are cheaper than a learned neural model and cannot be advertised as learned-model efficiency. GPU experiments overlapped the CPU work, so timings are local descriptive measurements.

This uses existing development seed/family choices and four evaluation episodes per setting. No parameters learn, no memory compresses, and no self-improvement occurs in the oracle reference. Its role is to falsify the explanation that more accurate memory or a larger dynamics model alone would solve E14. A stronger action-proposal/search method must be tested before that architectural addition is justified.
