# E12 interim: first fresh seed rejects a universal entropy-off repair

This records complete results for seed 13101 only. The second seed is still running; this is not the full E12 result. E10 remains valid for its own checkpoints and protocol.

| Arm | After stage | Two-room successes / 50 | Four-room successes / 50 | Key-door successes / 50 |
|---|---|---:|---:|---:|
| standard_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 8 | 19 |
| standard_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 0 | 0 | 1 |
| standard_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 0 | 0 | 1 |
| standard_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 21 | 17 |
| zero_entropy | 0: MiniGrid-MultiRoom-N2-S4-v0 | 50 | 45 | 0 |
| zero_entropy | 1: MiniGrid-MultiRoom-N4-S5-v1 | 0 | 0 | 0 |
| zero_entropy | 2: MiniGrid-DoorKey-8x8-v0 | 0 | 0 | 0 |
| zero_entropy | 3: MiniGrid-MultiRoom-N2-S4-v0 | 0 | 0 | 0 |

The zero-entropy learner acquires the initial task and transfers well to four rooms initially, but collapses late during four-room training. Subsequent key-door training and 262,144 returning two-room frames do not restore any measured success. The standard-entropy arm also loses competence during harder-task training but reacquires the original task on return.

This fresh counterexample prevents treating removal of entropy from the start as a sufficient continual-learning repair. It does not identify the exact cause of the late collapse, and it does not justify increasing entropy universally. The procedure must preserve usable competence while retaining the ability to gather new learning evidence. No changes to the registered E12 schedule or objectives were made in response to these results.
