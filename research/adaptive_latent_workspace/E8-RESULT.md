# E8: closed-loop transfer and retention fail during harder-task learning

The registered pilot completed for two seeds and two arms, each with 786,432 training frames across two-room, four-room and returning two-room environments. Policies see partial 7×7 symbolic images. Weights, optimizer state and renewal utility persist across stages; only episodic recurrent state resets.

| Seed | Arm | After training on | Two-room successes / 50 | Four-room successes / 50 | Seconds | Cumulative replacements |
|---|---|---|---:|---:|---:|---:|
| 11101 | recurrent_ppo | two rooms | 50 | 7 | 185.2 | 0 |
| 11101 | recurrent_ppo | four rooms | 2 | 0 | 181.1 | 0 |
| 11101 | recurrent_ppo | returning two rooms | 50 | 9 | 187.9 | 0 |
| 11101 | recurrent_ppo_renewal | two rooms | 50 | 7 | 186.6 | 48 |
| 11101 | recurrent_ppo_renewal | four rooms | 2 | 0 | 177.0 | 101 |
| 11101 | recurrent_ppo_renewal | returning two rooms | 50 | 22 | 185.5 | 161 |
| 11102 | recurrent_ppo | two rooms | 50 | 13 | 177.6 | 0 |
| 11102 | recurrent_ppo | four rooms | 0 | 0 | 167.9 | 0 |
| 11102 | recurrent_ppo | returning two rooms | 50 | 13 | 181.6 | 0 |
| 11102 | recurrent_ppo_renewal | two rooms | 50 | 12 | 188.3 | 53 |
| 11102 | recurrent_ppo_renewal | four rooms | 3 | 0 | 181.7 | 96 |
| 11102 | recurrent_ppo_renewal | returning two rooms | 50 | 29 | 191.1 | 151 |

All arms acquire the two-room task in the first stage. Four-room training then loses almost all measured two-room competence while failing the four-room test. Both arms reacquire two-room success when training returns there. Reacquisition is not the same as retention. Feature renewal in the actor/critic heads does not prevent this failure.

There can be nonzero four-room transfer after returning to the easier task; the complete per-seed results above preserve that outcome. It does not reverse the failed acquisition/retention result during the registered harder-task stage, and two seeds cannot establish a broad transfer effect.

The final harder-task training entropy approaches log(7), the maximum for seven actions, while reward and value estimates approach zero. That is consistent with objective-driven movement toward uniform actions under weak reward feedback. It is not yet a causal proof. E10 tests entropy and critic-representation interference from the same starting checkpoints; the E8 outcome is not silently corrected.

The policy has no private-grid or transition-function access. Evaluation uses frozen weights and fresh environment seeds; evaluation RNG is restored afterward. No evaluation checkpoint selection or hyperparameter tuning occurs within this run.

Costs include rollout/update time and forward-example counts. Tensor-byte counts cover model, optimizer and renewal state; peak GPU allocation also includes runtime training tensors. They do not measure all process RAM or energy. Small reporting and CPU preflight tasks may have overlapped; timings are local indicative measurements, not isolated-device claims.

This is a fixed-procedure closed-loop pilot using pinned PPO and CNN/LSTM code. It does not demonstrate recursive self-improvement, a learned world model, memory compression or general reasoning. The environment limits are only 40/80 steps, so this cannot establish very long-horizon efficiency. The broader architecture remains unverified.
