# E8 interim mechanism diagnosis

This note uses completed arm records while the full E8 run continues. Do not
treat it as the complete two-seed comparison.

For seed 11101, both the baseline and head-renewal arm achieve 50/50 two-room
held-out successes after the first stage. Four-room training ends with 2/50
two-room successes and 0/50 four-room successes in both arms. Both subsequently
relearn the two-room task. The second baseline seed likewise loses its old
two-room performance during four-room training.

The last four-room training minibatch reports action entropy:

| Completed arm | Entropy | log(7) minus entropy |
|---|---:|---:|
| 11101 baseline | 1.94589643 | 0.00001372 |
| 11101 renewal | 1.94589958 | 0.00001057 |
| 11102 baseline | 1.94589240 | 0.00001774 |

The maximum entropy for seven actions is log(7) = 1.94591015. At that point the
logged task return is zero and the value estimate approaches zero. This is
consistent with an entropy bonus pushing toward nearly uniform behavior after
task-learning gradients become weak. The shared critic can also change the
representation used by the actor. These are plausible objective/interference
mechanisms, not established causal conclusions from an entropy trace alone.

E10 freezes a counterfactual 2×2 test of entropy bonus and critic-to-representation
gradients from the same E8 starting checkpoints, plus a frozen-policy control.
It will distinguish retaining previous behavior from acquiring the harder task.
The experiment is **not run yet**. No hyperparameter correction has been applied
to E8, and its negative results remain intact.

This matters for the architecture: a self-improving learning procedure must
identify when an update objective is spending competence without obtaining
useful evidence. Adding memory modules or feature replacement alone does not
repair an objective that rewards the wrong behavior in that situation.
