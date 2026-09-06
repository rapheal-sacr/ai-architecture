# E12: fresh acquisition, unseen action chains and return-task retention

Both entropy choices begin at random initialization with two fresh training seeds. The learner retains weights and optimizer state across two-room, four-room, key-door, and returning two-room stages. No stage identity is supplied to the policy. The third environment requires using a key, a behavior absent from MultiRoom.

| Seed | Arm | Training stage | Two-room / 50 | Four-room / 50 | Key-door / 50 | Training seconds | Evaluation seconds |
|---|---|---|---:|---:|---:|---:|---:|
| 13101 | standard_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 8 | 19 | 176.3 | 32.4 |
| 13101 | standard_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 0 | 0 | 1 | 159.8 | 42.4 |
| 13101 | standard_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 0 | 0 | 1 | 165.3 | 44.0 |
| 13101 | standard_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 21 | 17 | 175.3 | 31.9 |
| 13101 | zero_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 45 | 0 | 183.7 | 37.1 |
| 13101 | zero_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 0 | 0 | 0 | 181.5 | 42.9 |
| 13101 | zero_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 0 | 0 | 0 | 160.1 | 43.6 |
| 13101 | zero_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 0 | 0 | 0 | 170.3 | 42.7 |
| 13102 | standard_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 6 | 20 | 176.6 | 33.0 |
| 13102 | standard_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 3 | 0 | 2 | 172.2 | 41.9 |
| 13102 | standard_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 5 | 0 | 3 | 162.4 | 41.0 |
| 13102 | standard_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 10 | 10 | 180.7 | 35.0 |
| 13102 | zero_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 3 | 1 | 0 | 184.8 | 40.3 |
| 13102 | zero_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 0 | 0 | 0 | 172.2 | 41.3 |
| 13102 | zero_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 0 | 0 | 0 | 161.8 | 41.7 |
| 13102 | zero_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 6 | 1 | 171.0 | 39.3 |

Neither fixed entropy setting reliably acquires the harder sequence. Standard entropy ends four-room training with 0/50 four-room success in both seeds and ends key-door training with 1/50 and 3/50 key-door success. It reacquires the original task at 50/50 in both seeds. Zero entropy collapses to 0/50 on all three tasks after four-room training in both seeds, and remains at zero after key-door training. On return, one seed stays at zero while the other reacquires two-room success at 50/50, with 6/50 four-room and 1/50 key-door success. Thus E10's entropy-off correction does not generalize as a sufficient repair.

The zero-entropy failures are not limited to immediately following a task switch: seed 13101 has strong four-room training returns before a late collapse, and seed 13102 has strong initial two-room returns before a late collapse within that stage. Frozen endpoint evaluation is reported without selecting an earlier successful checkpoint. The precise cause of these late instabilities is not isolated by this two-arm experiment. Neither more entropy nor its deletion alone is a reliable learning safeguard.

Each arm/seed receives 1,048,576 training frames. The tasks have 40/80/640-step episode limits. This expands the action-chain and time horizon of the prior pilot but does not establish general reasoning or very long-horizon agency.

Evaluation uses frozen weights and separately seeded episodes. Its scores do not influence updates, stage order, entropy or checkpoint selection. Identical evaluation seeds recur across stages to measure retention; they are not additional independent training replications. Students do not reset between stages.

Complete counts include training, rollout and evaluation forwards, optimizer updates and model/optimizer tensor bytes. Per-stage checkpoints permit continuation without throwing away complete stages. Peak GPU allocation does not include all driver or process RAM. No energy measurement or fixed-time benchmark was run.

The architecture and objective choices are fixed by this experiment. This is a baseline/mechanism test, not a demonstration that the system learned its own objective. It contains no world model, planner, neural compression mechanism or persistent learned procedure. Its successes or failures constrain those subsequent proposals rather than completing the overall goal.
