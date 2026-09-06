# E13: does consolidation repay its cost over five million observations?

The original E6 first million observations and teacher are preserved exactly. The extension continues the same declared Markov slow-bit flips and independent fast bits with a separate seed. Recalling the bulk generator at a larger length would change the original prefix and is deliberately not used. Each learner reprocesses the prefix from the original initialization; that work is charged.

| Seed | Arm | Exact prefix error / counters | Full-stream MSE | Final ten blocks MSE | Seconds | Forward examples | Stored tensor bytes | Merges |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 9101 | context_replay | True / True | 0.078636 | 0.083684 | 200.9 | 14999968 | 427128 | 0 |
| 9101 | guarded_sharing | True / True | 0.058944 | 0.060620 | 330.1 | 35813952 | 620832 | 465 |
| 9102 | context_replay | True / True | 0.084173 | 0.059804 | 203.1 | 14999968 | 427128 | 0 |
| 9102 | guarded_sharing | True / True | 0.069762 | 0.030689 | 330.8 | 36726752 | 620832 | 470 |

## Error by observed block interval

Interior intervals are approximately one million observations because the recorded blocks do not necessarily end exactly on that boundary. The table gives actual endpoints, with no interpolation. The original million and final five million are exact boundaries.

| Seed | Arm | Start (exclusive) | End (inclusive) | Interval MSE | Cumulative MSE | Cumulative seconds |
|---|---|---:|---:|---:|---:|---:|
| 9101 | context_replay | 0 | 1000000 | 0.088298 | 0.088298 | 40.3 |
| 9101 | context_replay | 1000000 | 1991584 | 0.066639 | 0.077514 | 80.0 |
| 9101 | context_replay | 1991584 | 2993184 | 0.064583 | 0.073187 | 119.8 |
| 9101 | context_replay | 2993184 | 3994784 | 0.081311 | 0.075224 | 160.3 |
| 9101 | context_replay | 3994784 | 5000000 | 0.092199 | 0.078636 | 200.9 |
| 9101 | guarded_sharing | 0 | 1000000 | 0.097583 | 0.097583 | 68.8 |
| 9101 | guarded_sharing | 1000000 | 1991584 | 0.054102 | 0.075934 | 135.1 |
| 9101 | guarded_sharing | 1991584 | 2993184 | 0.049365 | 0.067044 | 201.1 |
| 9101 | guarded_sharing | 2993184 | 3994784 | 0.044204 | 0.061317 | 265.5 |
| 9101 | guarded_sharing | 3994784 | 5000000 | 0.049513 | 0.058944 | 330.1 |
| 9102 | context_replay | 0 | 1000000 | 0.100892 | 0.100892 | 40.4 |
| 9102 | context_replay | 1000000 | 1991584 | 0.073668 | 0.087338 | 80.4 |
| 9102 | context_replay | 1991584 | 2993184 | 0.087216 | 0.087297 | 121.2 |
| 9102 | context_replay | 2993184 | 3994784 | 0.083924 | 0.086451 | 161.5 |
| 9102 | context_replay | 3994784 | 5000000 | 0.075119 | 0.084173 | 203.1 |
| 9102 | guarded_sharing | 0 | 1000000 | 0.124842 | 0.124842 | 68.4 |
| 9102 | guarded_sharing | 1000000 | 1991584 | 0.072596 | 0.098829 | 136.0 |
| 9102 | guarded_sharing | 1991584 | 2993184 | 0.055303 | 0.084264 | 202.0 |
| 9102 | guarded_sharing | 2993184 | 3994784 | 0.051027 | 0.075931 | 266.7 |
| 9102 | guarded_sharing | 3994784 | 5000000 | 0.045248 | 0.069762 | 330.8 |

## Cost and accuracy comparison

Seed 9101: guarded sharing / context replay ratios are 0.7496 for cumulative MSE, 0.7244 for late MSE, 2.3876 for forward examples, and 1.6433 for elapsed time.
Seed 9102: guarded sharing / context replay ratios are 0.8288 for cumulative MSE, 0.5132 for late MSE, 2.4485 for forward examples, and 1.6283 for elapsed time.

A lower late error is not retrospective repayment of earlier error or work. Accuracy and work are reported separately rather than combined with an invented exchange rate. No unobserved break-even point is treated as measured.

The final ten blocks can be shorter than exactly 100,000 observations; weighted means use the actual recorded sizes. Complete blocks retain all cost counters, including model growth, updates, replay and merge-related work. CPU work overlaps independent GPU experiments, so operation counts are stronger evidence than local wall-time comparisons.

This continues two already studied streams. It is not a fresh external replication, a rare-fact preservation test, semantic compression proof, closed-loop reasoning agent or recursive improvement result. Passing anchor checks only establishes those sampled checks.
