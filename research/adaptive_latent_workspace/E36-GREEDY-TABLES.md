# E36 greedy intervention: all audited pairs

Reused eight worlds, post hoc intervention. Different policies cause different experienced data. Not independent seed-world trials or fresh transfer.

| Model | Regime | Cases / worlds | Soft goals | Greedy goals | Better / tied / worse |
|---|---|---:|---:|---:|---:|
| static_ttt | stationary | 6 / 2 | 137.00 | 142.67 | 3 / 0 / 3 |
| static_ttt | recurring | 6 / 2 | 131.33 | 169.00 | 5 / 0 / 1 |
| static_ttt | drifting | 6 / 2 | 122.50 | 125.17 | 4 / 0 / 2 |
| static_ttt | noisy | 6 / 2 | 114.67 | 149.83 | 6 / 0 / 0 |
| first_order_ttt | stationary | 6 / 2 | 185.33 | 251.33 | 6 / 0 / 0 |
| first_order_ttt | recurring | 6 / 2 | 164.00 | 223.33 | 6 / 0 / 0 |
| first_order_ttt | drifting | 6 / 2 | 166.83 | 227.33 | 6 / 0 / 0 |
| first_order_ttt | noisy | 6 / 2 | 145.83 | 215.17 | 6 / 0 / 0 |
| e2e_ttt | stationary | 6 / 2 | 182.17 | 252.50 | 6 / 0 / 0 |
| e2e_ttt | recurring | 6 / 2 | 171.67 | 225.00 | 6 / 0 / 0 |
| e2e_ttt | drifting | 6 / 2 | 164.67 | 229.67 | 6 / 0 / 0 |
| e2e_ttt | noisy | 6 / 2 | 148.67 | 216.00 | 6 / 0 / 0 |
| counts | stationary | 2 / 2 | 180.00 | 246.00 | 2 / 0 / 0 |
| counts | recurring | 2 / 2 | 153.50 | 204.50 | 2 / 0 / 0 |
| counts | drifting | 2 / 2 | 169.50 | 219.00 | 2 / 0 / 0 |
| counts | noisy | 2 / 2 | 154.50 | 219.50 | 2 / 0 / 0 |

All80cases execute512actions. Neural model-query, gradient-update and Bellman
work totals equal their original cases exactly; persistent neural tensors also
match. Counts history occupancy may differ with experienced trajectories.
Float64 greedy Bellman arithmetic changes per-operation cost; original acting
timing sometimes overlapped a training audit. Timings are not an isolated
algorithmic speed comparison. Training was reused, not repeated or free.

New audit: 40960 actions, 36864 updates, 320 evaluation checkpoints. The36training checkpoints are inherited from the hashed prior audit, not newly replayed.

Full paired cases, observed costs, prediction metrics and artifact hashes are in summary.json.
