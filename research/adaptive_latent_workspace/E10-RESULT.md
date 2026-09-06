# E10: entropy and critic-gradient counterfactual

Two factors change from identical E8 two-room checkpoints: the entropy bonus and whether critic gradients reach the shared CNN/LSTM. Actor gradients remain enabled in every learning arm. A frozen-policy reference receives no updates. Each learning arm uses 262,144 four-room frames and the same starting optimizer/RNG state.

The first CUDA launch failed before its first optimizer update because restored custom optimizer step tensors stayed on CPU. The corrected adapter moves state to its parameter device and passed CUDA preflight. The failed launch and source hashes remain recorded in E10-RUNTIME-NOTE.md; its unmeasured startup time is not counted as zero.

| Seed | Arm | Two-room successes / 50 | Four-room successes / 50 | Last entropy | Training seconds |
|---|---|---:|---:|---:|---:|
| 11101 | entropy_shared | 2 | 0 | 1.945899 | 168.3 |
| 11101 | no_entropy_shared | 50 | 50 | 0.033867 | 169.4 |
| 11101 | entropy_detached_critic | 2 | 0 | 1.945889 | 172.5 |
| 11101 | no_entropy_detached_critic | 0 | 0 | 0.112615 | 170.6 |
| 11101 | frozen_no_update | 50 | 7 | — | 0.0 |
| 11102 | entropy_shared | 0 | 0 | 1.945898 | 169.8 |
| 11102 | no_entropy_shared | 50 | 50 | 0.012172 | 175.0 |
| 11102 | entropy_detached_critic | 0 | 0 | 1.945886 | 169.0 |
| 11102 | no_entropy_detached_critic | 50 | 50 | 0.012013 | 172.1 |
| 11102 | frozen_no_update | 50 | 13 | — | 0.0 |

Removing entropy while retaining shared critic gradients acquires the harder task and retains the old task at 50/50 successes each in both seeds. With entropy retained, detaching the critic still fails. With entropy removed, critic detachment fails in one seed and succeeds in the other. This supports the entropy intervention in this development setup and rejects critic detachment as a reliable general repair.

The maximum seven-action entropy is log(7) = 1.945910.

## Checkpoint control reproduction

Seed 11101: complete training rows equal to E8 = False; all recorded evaluation fields equal = True; maximum training metric differences = {'frames': 0, 'mean_return': 0.0, 'entropy': 0.00011471286416053772, 'value': 0.0007994032507667725, 'policy_loss': 0.0003164010646941051, 'value_loss': 6.036918911433986e-07, 'grad_norm': 0.0007727464116043105}.
Seed 11102: complete training rows equal to E8 = False; all recorded evaluation fields equal = True; maximum training metric differences = {'frames': 0, 'mean_return': 0.013437500596046448, 'entropy': 0.004850603640079498, 'value': 0.004211636958643794, 'policy_loss': 0.0023383968509733677, 'value_loss': 8.79362069099443e-05, 'grad_norm': 0.002333985573035685}.

The factors can change exploration and subsequent data as well as direct gradient interference. This is a closed-loop causal intervention, not an attribution of every downstream effect to one local gradient. Successful retention alone does not establish acquisition of the harder task.

These checkpoints and environment families were already studied in E8. The counterfactual is a development diagnostic, not fresh final validation. Only two training seeds are tested. Fifty evaluation episodes per checkpoint are not fifty independently trained models.

All learning arms have the same parameter count and collected frame budget. Counts and peak allocated GPU bytes are retained in the JSON. Peak allocation excludes driver/context memory and process RAM. E11 CPU work overlaps part of execution, so timings do not establish fixed-time superiority. The frozen arm has zero training work by definition.

This experiment does not implement memory compression, learned dynamics, planning or recursive objective learning. Any useful objective change here is researcher-selected and must not be described as autonomous procedural improvement.
