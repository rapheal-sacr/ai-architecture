# E7: feature renewal helps prediction with extra compute

The official generate-and-test implementation is applied to the same two-layer tanh learner with its custom AdamGnT optimizer. Nonrenewal controls use that same optimizer. This separates the renewal intervention from optimizer choice; it does not reproduce the published small-ReLU CBP experiment.

| Arm | MSE | Final ten blocks MSE | Seconds | Tensor bytes | Replacements |
|---|---:|---:|---:|---:|---:|
| context_replay | 0.084659 | 0.073925 | 40.88 | 427128 | 0.0 |
| adam_gnt | 0.147706 | 0.110775 | 41.84 | 89188 | 0.0 |
| cbp_gnt | 0.090919 | 0.067093 | 56.14 | 91236 | 408.5 |
| replay_gnt | 0.070605 | 0.046609 | 50.40 | 261220 | 0.0 |
| replay_cbp_gnt | 0.066330 | 0.048859 | 64.93 | 263268 | 391.0 |

cbp_gnt reduces whole-stream error 38.4% versus adam_gnt, with 34.2% more runtime. The error direction agrees on both seeds, but this is a two-seed screen rather than a powered replication.
replay_cbp_gnt reduces whole-stream error 6.1% versus replay_gnt, with 28.8% more runtime. The error direction agrees on both seeds, but this is a two-seed screen rather than a powered replication.

Replay plus renewal has the lowest whole-stream mean prediction error here. The incremental gain from renewal with replay is much smaller than without replay, and its extra runtime remains. In the final ten blocks, replay without renewal has lower mean error (0.04661 versus 0.04886), so the whole-stream gain does not show continuing late-time superiority. Fixed-budget comparisons are needed before calling this efficient. The final training-batch saturation diagnostic decreases with renewal; it is not a measure of global feature quality or proof of a causal saturation mechanism.

This result concerns shared prediction on the external fixed-function stream. It neither solves hidden conflicting contexts nor establishes memory compression or general reasoning. No conclusions about CBP across its published settings follow from this limited port.

The next registered screen moves to closed-loop partial-observation MiniGrid tasks using a pinned recurrent PPO baseline. It tests whether feature renewal remains useful with action-dependent data and sparse reward. That pilot is not the complete proposed architecture, and its 40/80-step tasks are not adequate evidence for very long-horizon competence.
