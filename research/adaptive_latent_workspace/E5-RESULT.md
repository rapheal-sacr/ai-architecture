# E5: protection survives capacity controls; routing can be cheaper

Four fresh seeds each receive 262,144 observations. The protocol was frozen before running any of these outcomes.

| Arm | Stream MSE | Returning early MSE | Last quarter MSE | Seconds | Parameters | Tensor bytes | Forward examples |
|---|---:|---:|---:|---:|---:|---:|---:|
| online | 0.036261 | 0.164286 | 0.043259 | 7.56 | 4996 | 60120 | 524288 |
| context_replay | 0.024569 | 0.080681 | 0.017251 | 10.92 | 7300 | 480984 | 786400 |
| wide_context_replay | 0.022306 | 0.079525 | 0.015943 | 12.61 | 19828 | 631320 | 786400 |
| compact_wide_replay | 0.024412 | 0.088748 | 0.018469 | 12.33 | 19828 | 287256 | 786400 |
| isolated_adaptation | 0.009171 | 0.025676 | 0.004766 | 12.08 | 19984 | 239904 | 1515520 |
| overwrite_during_identification | 0.033051 | 0.163999 | 0.033712 | 10.21 | 4996 | 59976 | 808672 |
| active_first_isolation | 0.009171 | 0.025676 | 0.004766 | 9.53 | 19984 | 239904 | 796240 |

Increasing context replay to approximately the same parameter count as four isolated modules does not remove the candidate's prediction advantage on this family. The small-buffer variant also loses. These are parameter/near-storage controls, not exact FLOP matching. Replay performs nearly twice as many training examples, and the wide network has higher cost per forward example.

Copying temporary updates into the source persistent module eliminates most of the advantage and leaves one persistent module on every seed. The intervention changes only protection during unresolved adaptation; subsequent module counts and routing decisions can therefore differ. This supports the importance of protection within this implementation and tested family.

Active-first routing has identical stored prediction metrics and held-out current-context metrics on all four seeds: True. This check compares aggregate scored metrics, not saved per-example prediction tensors. It reduces measured time by 21.1% and counted forward examples by 47.5%.

This supports a cheap active-model acceptance test when regimes are well separated. It does not establish a learned router or calibrated confidence. A broadly acceptable but suboptimal active model may hide a better alternative; overlapping contexts remain an adversarial follow-up.

E4's independent transfer failure remains in force. Better routing does not solve unnecessary context partitioning, memory consolidation, or general reasoning. No closed-loop task was executed in E5.
