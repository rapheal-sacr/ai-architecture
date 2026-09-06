# E9: learned procedure and bounded self-application

This experiment distinguishes fixed-procedure student learning, bootstrap meta-learning of an update policy, and that learned policy proposing changes to its own parameters. It tests only a small fixed controller family.

A deterministic double-precision preflight matched the unrolled meta-gradient to a central finite difference and verified initial equivalence to the declared fixed optimizer. A separate dummy end-to-end run exercised bootstrap, self-proposal, admission, final evaluation and counters. Neither preflight is a scored result.

| Replicate | Bootstrap seconds | Accepted self-updates | Self-trial seconds | Controller parameters |
|---|---:|---:|---:|---:|
| 12101 | 65.79 | 1/8 | 9.57 | 65 |
| 12102 | 64.41 | 7/8 | 9.69 | 65 |

All rejected trials remain in the complete JSON and WD JSONL logs. Divergent tasks are recorded explicitly, never averaged away as missing observations. Admission uses fresh validation tasks plus a retained validation sample; final tasks are separate.

| Distribution | Inner steps | Arm | Replicates | Query MSE | Prequential MSE | Seconds per 16 tasks | Divergent tasks |
|---|---:|---|---:|---:|---:|---:|---:|
| in_distribution | 16 | adam_default | 2 | 0.248628 | 0.436008 | 0.223 | 0 |
| in_distribution | 16 | adam_selected | 2 | 0.072740 | 0.180782 | 0.223 | 0 |
| in_distribution | 16 | frozen_bootstrap | 2 | 0.068575 | 0.202799 | 0.682 | 0 |
| in_distribution | 16 | self_applied | 2 | 0.065569 | 0.192817 | 0.686 | 0 |
| in_distribution | 128 | adam_default | 2 | 0.036148 | 0.115733 | 1.771 | 0 |
| in_distribution | 128 | adam_selected | 2 | 0.018937 | 0.051440 | 1.747 | 0 |
| in_distribution | 128 | frozen_bootstrap | 2 | 0.020226 | 0.054852 | 5.432 | 0 |
| in_distribution | 128 | self_applied | 2 | 0.020117 | 0.053320 | 5.439 | 0 |
| in_distribution | 512 | adam_default | 2 | 0.014452 | 0.045864 | 6.905 | 0 |
| in_distribution | 512 | adam_selected | 2 | 0.010791 | 0.022690 | 6.883 | 0 |
| in_distribution | 512 | frozen_bootstrap | 2 | 0.009191 | 0.022991 | 21.627 | 0 |
| in_distribution | 512 | self_applied | 2 | 0.008997 | 0.022541 | 21.672 | 0 |
| scaled_inputs | 16 | adam_default | 2 | 0.677313 | 0.960273 | 0.224 | 0 |
| scaled_inputs | 16 | adam_selected | 2 | 0.306724 | 0.523584 | 0.221 | 0 |
| scaled_inputs | 16 | frozen_bootstrap | 2 | 0.325211 | 0.582089 | 0.693 | 0 |
| scaled_inputs | 16 | self_applied | 2 | 0.314485 | 0.562742 | 0.686 | 0 |
| scaled_inputs | 128 | adam_default | 2 | 0.234264 | 0.408860 | 1.746 | 0 |
| scaled_inputs | 128 | adam_selected | 2 | 0.143832 | 0.236585 | 1.742 | 0 |
| scaled_inputs | 128 | frozen_bootstrap | 2 | 0.152494 | 0.253445 | 5.435 | 0 |
| scaled_inputs | 128 | self_applied | 2 | 0.151534 | 0.249983 | 5.447 | 0 |
| scaled_inputs | 512 | adam_default | 2 | 0.116087 | 0.222454 | 6.953 | 0 |
| scaled_inputs | 512 | adam_selected | 2 | 0.072204 | 0.130568 | 6.890 | 0 |
| scaled_inputs | 512 | frozen_bootstrap | 2 | 0.066065 | 0.133379 | 21.667 | 0 |
| scaled_inputs | 512 | self_applied | 2 | 0.066144 | 0.131673 | 21.628 | 0 |
| relu_teacher | 16 | adam_default | 2 | 0.361998 | 0.537661 | 0.224 | 0 |
| relu_teacher | 16 | adam_selected | 2 | 0.171097 | 0.275992 | 0.222 | 0 |
| relu_teacher | 16 | frozen_bootstrap | 2 | 0.167194 | 0.299212 | 0.687 | 0 |
| relu_teacher | 16 | self_applied | 2 | 0.165760 | 0.290626 | 0.694 | 0 |
| relu_teacher | 128 | adam_default | 2 | 0.128153 | 0.211398 | 1.729 | 0 |
| relu_teacher | 128 | adam_selected | 2 | 0.051610 | 0.121425 | 1.726 | 0 |
| relu_teacher | 128 | frozen_bootstrap | 2 | 0.065034 | 0.132923 | 5.406 | 0 |
| relu_teacher | 128 | self_applied | 2 | 0.064913 | 0.131085 | 5.432 | 0 |
| relu_teacher | 512 | adam_default | 2 | 0.052698 | 0.119688 | 6.930 | 0 |
| relu_teacher | 512 | adam_selected | 2 | 0.020701 | 0.051385 | 6.861 | 0 |
| relu_teacher | 512 | frozen_bootstrap | 2 | 0.019759 | 0.055215 | 21.581 | 0 |
| relu_teacher | 512 | self_applied | 2 | 0.019232 | 0.054655 | 21.883 | 0 |

## Self-application versus the frozen predecessor

Replicate 12101: 2 settings improved, 7 worsened, 0 tied, 0 had divergence. These are nine paired distribution/horizon summaries, not nine independent replications.
Replicate 12102: 9 settings improved, 0 worsened, 0 tied, 0 had divergence. These are nine paired distribution/horizon summaries, not nine independent replications.

An admitted change is not automatically useful self-improvement. The final comparisons above determine whether it transfers; the acceptance count only records what the finite admission sample allowed. A small gain on some tasks cannot justify a claim of reliable recursive improvement across the tested family.

The selected fixed optimizer chooses its learning rate on development tasks. Its selection work is stored separately. Learned-controller inference costs, bootstrap meta-training, unsuccessful self-update trials and final evaluation all appear in the complete accounting. No fixed-time final comparison was run; lower query error at equal inner updates is not proof of efficiency.

An additional causal control is missing: continue meta-training the same bootstrap controller with a conventional meta-optimizer under the same proposal/admission budget. E9 compares self-application with freezing, so it cannot attribute any gain specifically to self-application rather than extra meta-training. The self-update code really uses its own learned rule; the benefit of that choice remains unproven.

The controller persists across tasks, but student weights reset between meta tasks. This experiment therefore does not establish persistent task-specific memory or long-horizon agent competence. The architecture and admission rule remain fixed; bounded self-application is not open-ended architecture invention. No full-goal completion is licensed.
