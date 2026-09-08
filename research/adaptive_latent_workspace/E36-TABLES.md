# E36 audited action results

Source-frozen scored run; no checkpoint or hyperparameter selection on these worlds.

The planner is fixed. This tests whether predictive continual learning supplies
useful dynamics for actual actions. It does not train the goal generator or
reasoning procedure, nor prove memory compression or recursive improvement.

## Goal completion

Each world has a fixed paid-action budget. Neural rows cross initialization
seeds with the same worlds; control rows run once per world. Repeating a
control for paired neural comparisons does not create independent worlds.
Evaluation evidence is action-dependent, unlike equal-data training.

| Condition | Regime | Cases / worlds | Mean goals | Range | Mean actual accuracy | Mean NLL |
|---|---|---:|---:|---:|---:|---:|
| static_frozen | stationary | 6 / 2 | 128.00 | 116–136 | 22.59% | 1.6075 |
| static_frozen | recurring | 6 / 2 | 122.00 | 117–134 | 23.05% | 1.6246 |
| static_frozen | drifting | 6 / 2 | 115.33 | 98–130 | 22.10% | 1.6378 |
| static_frozen | noisy | 6 / 2 | 109.33 | 104–116 | 19.82% | 1.7168 |
| static_ttt | stationary | 6 / 2 | 137.00 | 130–153 | 78.87% | 1.1060 |
| static_ttt | recurring | 6 / 2 | 131.33 | 110–145 | 63.83% | 1.2530 |
| static_ttt | drifting | 6 / 2 | 122.50 | 101–136 | 52.93% | 1.3522 |
| static_ttt | noisy | 6 / 2 | 114.67 | 104–125 | 53.12% | 1.4088 |
| first_order_ttt | stationary | 6 / 2 | 185.33 | 179–194 | 98.11% | 0.2183 |
| first_order_ttt | recurring | 6 / 2 | 164.00 | 154–175 | 76.89% | 0.6930 |
| first_order_ttt | drifting | 6 / 2 | 166.83 | 157–176 | 83.53% | 0.6039 |
| first_order_ttt | noisy | 6 / 2 | 145.83 | 130–167 | 82.32% | 0.7518 |
| e2e_ttt | stationary | 6 / 2 | 182.17 | 172–188 | 97.66% | 0.2751 |
| e2e_ttt | recurring | 6 / 2 | 171.67 | 158–183 | 80.05% | 0.6695 |
| e2e_ttt | drifting | 6 / 2 | 164.67 | 159–176 | 82.39% | 0.6490 |
| e2e_ttt | noisy | 6 / 2 | 148.67 | 130–160 | 81.45% | 0.7831 |
| e2e_frozen | stationary | 6 / 2 | 119.67 | 103–138 | 19.99% | 1.7161 |
| e2e_frozen | recurring | 6 / 2 | 122.67 | 107–141 | 22.27% | 1.6690 |
| e2e_frozen | drifting | 6 / 2 | 115.00 | 95–137 | 22.92% | 1.6892 |
| e2e_frozen | noisy | 6 / 2 | 108.67 | 96–125 | 23.40% | 1.7337 |
| random | stationary | 2 / 2 | 128.50 | 111–146 | 17.58% | 1.7918 |
| random | recurring | 2 / 2 | 119.00 | 118–120 | 18.36% | 1.7918 |
| random | drifting | 2 / 2 | 109.50 | 105–114 | 15.62% | 1.7918 |
| random | noisy | 2 / 2 | 106.00 | 96–116 | 15.92% | 1.7918 |
| bfs | stationary | 2 / 2 | 170.50 | 166–175 | 99.02% | — |
| bfs | recurring | 2 / 2 | 203.00 | 182–224 | 90.72% | — |
| bfs | drifting | 2 / 2 | 236.00 | 230–242 | 92.58% | — |
| bfs | noisy | 2 / 2 | 170.00 | 164–176 | 71.58% | — |
| counts | stationary | 2 / 2 | 180.00 | 179–181 | 98.05% | 0.2099 |
| counts | recurring | 2 / 2 | 153.50 | 149–158 | 66.21% | 0.7472 |
| counts | drifting | 2 / 2 | 169.50 | 168–171 | 76.66% | 0.6382 |
| counts | noisy | 2 / 2 | 154.50 | 140–169 | 83.11% | 0.7956 |
| fixed0 | stationary | 2 / 2 | 170.50 | 166–175 | — | — |
| fixed1 | stationary | 2 / 2 | 169.00 | 169–169 | — | — |
| fixed0 | recurring | 2 / 2 | 175.50 | 175–176 | — | — |
| fixed1 | recurring | 2 / 2 | 172.00 | 170–174 | — | — |
| fixed0 | drifting | 2 / 2 | 168.00 | 167–169 | — | — |
| fixed1 | drifting | 2 / 2 | 168.00 | 168–168 | — | — |
| fixed0 | noisy | 2 / 2 | 134.50 | 130–139 | — | — |
| fixed1 | noisy | 2 / 2 | 136.50 | 128–145 | — | — |

## E2E paired differences

Positive goal differences favor E2E TTT. Counts are descriptive paired
cases, not independent population trials or significance tests.

| Baseline | Regime | Better / tied / worse goals | Mean goal difference | Mean acting-time ratio |
|---|---|---:|---:|---:|
| static_ttt | stationary | 6 / 0 / 0 | +45.17 | 1.00 |
| first_order_ttt | stationary | 2 / 0 / 4 | -3.17 | 1.00 |
| e2e_frozen | stationary | 6 / 0 / 0 | +62.50 | 1.11 |
| bfs | stationary | 6 / 0 / 0 | +11.67 | 508.93 |
| counts | stationary | 4 / 0 / 2 | +2.17 | 27.09 |
| random | stationary | 6 / 0 / 0 | +53.67 | 3347.23 |
| fixed0 | stationary | 6 / 0 / 0 | +11.67 | 3685.10 |
| fixed1 | stationary | 6 / 0 / 0 | +13.17 | 3510.99 |
| static_ttt | recurring | 6 / 0 / 0 | +40.33 | 0.99 |
| first_order_ttt | recurring | 5 / 0 / 1 | +7.67 | 1.01 |
| e2e_frozen | recurring | 6 / 0 / 0 | +49.00 | 1.10 |
| bfs | recurring | 0 / 0 / 6 | -31.33 | 513.35 |
| counts | recurring | 5 / 1 / 0 | +18.17 | 27.09 |
| random | recurring | 6 / 0 / 0 | +52.67 | 3339.37 |
| fixed0 | recurring | 3 / 0 / 3 | -3.83 | 4189.82 |
| fixed1 | recurring | 3 / 0 / 3 | -0.33 | 4092.79 |
| static_ttt | drifting | 6 / 0 / 0 | +42.17 | 1.01 |
| first_order_ttt | drifting | 2 / 0 / 4 | -2.17 | 1.02 |
| e2e_frozen | drifting | 6 / 0 / 0 | +49.67 | 1.14 |
| bfs | drifting | 0 / 0 / 6 | -71.33 | 516.41 |
| counts | drifting | 2 / 0 / 4 | -4.83 | 25.39 |
| random | drifting | 6 / 0 / 0 | +55.17 | 3335.16 |
| fixed0 | drifting | 2 / 0 / 4 | -3.33 | 3186.42 |
| fixed1 | drifting | 2 / 0 / 4 | -3.33 | 3121.46 |
| static_ttt | noisy | 6 / 0 / 0 | +34.00 | 0.98 |
| first_order_ttt | noisy | 3 / 1 / 2 | +2.83 | 1.01 |
| e2e_frozen | noisy | 6 / 0 / 0 | +40.00 | 1.09 |
| bfs | noisy | 0 / 0 / 6 | -21.33 | 499.96 |
| counts | noisy | 1 / 1 / 4 | -5.83 | 27.19 |
| random | noisy | 6 / 0 / 0 | +42.67 | 3178.84 |
| fixed0 | noisy | 5 / 0 / 1 | +14.17 | 3739.29 |
| fixed1 | noisy | 6 / 0 / 0 | +12.17 | 5039.61 |

## Acquisition and failure after first acquisition

Threshold: at least 11 of 12 transition argmaxes correct at a pre-action
probe. Probes reuse already-paid model queries; labels are evaluator-only.
This measures the map predictions expressed by this query interface, not
irreversibly retained information or actual noisy-outcome accuracy. Time starts
at each hidden segment boundary. Zero means threshold held before any new
segment action. A never-hit segment is censored at its full duration.
The first point hit is not stable acquisition; later failures are retained.
No post-final-action query or per-recovery wall-time measurement is available.

| Condition | Regime | Acquired / all segments | Mean restricted action fraction | Hits followed by failure |
|---|---|---:|---:|---:|
| static_frozen | stationary | 0 / 6 | 1.000 | 0 |
| static_frozen | recurring | 0 / 54 | 1.000 | 0 |
| static_frozen | drifting | 0 / 54 | 1.000 | 0 |
| static_frozen | noisy | 0 / 6 | 1.000 | 0 |
| static_ttt | stationary | 1 / 6 | 0.854 | 1 |
| static_ttt | recurring | 0 / 54 | 1.000 | 0 |
| static_ttt | drifting | 0 / 54 | 1.000 | 0 |
| static_ttt | noisy | 2 / 6 | 0.885 | 2 |
| first_order_ttt | stationary | 6 / 6 | 0.043 | 6 |
| first_order_ttt | recurring | 49 / 54 | 0.618 | 46 |
| first_order_ttt | drifting | 44 / 54 | 0.577 | 42 |
| first_order_ttt | noisy | 6 / 6 | 0.062 | 6 |
| e2e_ttt | stationary | 6 / 6 | 0.053 | 6 |
| e2e_ttt | recurring | 43 / 54 | 0.664 | 43 |
| e2e_ttt | drifting | 46 / 54 | 0.570 | 45 |
| e2e_ttt | noisy | 6 / 6 | 0.066 | 6 |
| e2e_frozen | stationary | 0 / 6 | 1.000 | 0 |
| e2e_frozen | recurring | 0 / 54 | 1.000 | 0 |
| e2e_frozen | drifting | 0 / 54 | 1.000 | 0 |
| e2e_frozen | noisy | 0 / 6 | 1.000 | 0 |
| bfs | stationary | 0 / 2 | 1.000 | 0 |
| bfs | recurring | 5 / 18 | 0.801 | 0 |
| bfs | drifting | 12 / 18 | 0.593 | 0 |
| bfs | noisy | 1 / 2 | 0.531 | 1 |
| counts | stationary | 2 / 2 | 0.044 | 0 |
| counts | recurring | 7 / 18 | 0.838 | 0 |
| counts | drifting | 10 / 18 | 0.787 | 0 |
| counts | noisy | 2 / 2 | 0.065 | 0 |

## Costs and scope

Scorer total time: 2276.959s. Audit: 1885.443s.
Audit verifies 1152 outer updates, 73728 evaluation actions and 612 noninitial checkpoints.
Training replay reuses the training routine. Evaluation gradient updates
bypass the action wrapper; physical transitions/noise/goals and NumPy
planner probabilities are independently reconstructed. Source hashes,
saved logits, optimizer/model states and declared work are checked.

Training and acting seconds, operation counts and storage are preserved per
case in summary.json. Acting timers include environment stepping and Python
control/update overhead, but exclude private map diagnostics and checkpoint
serialization. Total case timers include those. Neither is isolated kernel
latency. Reported state bytes omit Python/allocator overhead. Process peak
RSS includes multiple models, optimizer states, training graphs and data.

With the declared noise, perfect map knowledge yields expected actual accuracy 83.33% and NLL 0.718801.
These are distributional expectations, not bounds on each finite realized
sample. The controller is not supplied the true noise rate or map.

These randomly generated finite cycles contain strong structure, a fixed
alphabet, and only two test worlds per regime in the scored protocol.
Neither a local gain nor failure identifies a universal AI improvement-rate
bound. Use goal outcomes, retained knowledge and costs together before
deciding the next architecture change. The full goal remains active.

## Supplemental constant-action falsifier

Fixed0 and fixed1 were added before scored neural action results were
available. They do not change the frozen primary experiment, and both
are retained without selecting the better action per world. Because each
action independently tours every state, these policies complete goals
without learning. Raw goal count cannot establish acquired map knowledge.
Their 8192 receipts were separately physically audited.
Their timing includes world construction and receipt generation, excludes
audit/file writing, and is not isolated kernel latency. Accuracy/NLL are
omitted because these policies do not predict transitions. See
E36-CONSTANT-CONTROL.md for the structural derivation and scope.
