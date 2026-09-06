# E11: self-application versus equal-budget conventional meta-updates

The fixed control receives the same eight proposal opportunities, task seeds, learning rate, moment coefficients and admission rule as E9. It sets the learned multiplier to one. This controls additional meta-training while preserving the proposed self-updater and its frozen predecessor unchanged. New final-task seeds were frozen before scored execution.

A replay preflight reconstructed replicate 12101 from its bootstrap checkpoint and exactly matched E9 final controller tensors and all eight admission decisions. This verifies the proposal-loop adapter; it is not a new scored replicate.

| Replicate | Self-updates accepted | Fixed updates accepted | Prior bootstrap seconds | Prior self-proposal seconds | Fixed proposal seconds |
|---|---:|---:|---:|---:|---:|
| 12101 | 1/8 | 6/8 | 65.79 | 9.57 | 9.34 |
| 12102 | 7/8 | 4/8 | 64.41 | 9.69 | 9.24 |

Query error follows; lower is better. The final tasks are paired within each setting. Settings share teacher seeds and are correlated, so the number of settings won is descriptive, not a significance test.

| Replicate | Distribution | Steps | Frozen MSE | Self-applied MSE | Fixed-meta MSE | Self lower than fixed: tasks / 16 |
|---|---|---:|---:|---:|---:|---:|
| 12101 | in_distribution | 16 | 0.048824 | 0.048280 | 0.049439 | 9/16 |
| 12101 | in_distribution | 128 | 0.015818 | 0.016235 | 0.015688 | 1/16 |
| 12101 | in_distribution | 512 | 0.007440 | 0.007251 | 0.007533 | 13/16 |
| 12101 | scaled_inputs | 16 | 0.234376 | 0.237901 | 0.232747 | 4/16 |
| 12101 | scaled_inputs | 128 | 0.107348 | 0.111184 | 0.106119 | 2/16 |
| 12101 | scaled_inputs | 512 | 0.049513 | 0.049418 | 0.052961 | 12/16 |
| 12101 | relu_teacher | 16 | 0.121303 | 0.121694 | 0.120380 | 6/16 |
| 12101 | relu_teacher | 128 | 0.045848 | 0.049487 | 0.042698 | 1/16 |
| 12101 | relu_teacher | 512 | 0.014023 | 0.013937 | 0.014091 | 10/16 |
| 12102 | in_distribution | 16 | 0.048168 | 0.045872 | 0.047480 | 16/16 |
| 12102 | in_distribution | 128 | 0.016505 | 0.016068 | 0.016009 | 7/16 |
| 12102 | in_distribution | 512 | 0.007297 | 0.007130 | 0.007344 | 13/16 |
| 12102 | scaled_inputs | 16 | 0.236201 | 0.227692 | 0.230995 | 15/16 |
| 12102 | scaled_inputs | 128 | 0.116137 | 0.113391 | 0.113062 | 5/16 |
| 12102 | scaled_inputs | 512 | 0.047903 | 0.050152 | 0.049983 | 9/16 |
| 12102 | relu_teacher | 16 | 0.123923 | 0.121157 | 0.123045 | 16/16 |
| 12102 | relu_teacher | 128 | 0.054139 | 0.050648 | 0.050369 | 4/16 |
| 12102 | relu_teacher | 512 | 0.014908 | 0.014807 | 0.015098 | 11/16 |

Replicate 12101: self-application has lower mean error than conventional meta-updates in 4/9 settings, and lower mean error than freezing in 4/9 settings.

Replicate 12102: self-application has lower mean error than conventional meta-updates in 5/9 settings, and lower mean error than freezing in 8/9 settings.

These are the two previously studied bootstrap initializations, not a new full-pipeline replication. The control only tests the declared fixed Adam-shaped rule; it does not search all meta-optimizers or learning rates. All rejected proposals and divergent evaluations remain in the complete records.

Bootstrap and prior self-trial work are inherited costs, not free checkpoints. The E9 records preserve their detailed training and evaluation counts. E11 records all extra conventional proposal and final-evaluation work. CPU evaluation overlapped E10 GPU work: wall times are descriptive, and no fixed-time efficiency claim follows.

Persistent controller parameters still coexist with students that reset between tasks. This control cannot establish persistent factual memory, general reasoning, open-ended recursive improvement or long-horizon agent efficiency.
