# Frozen-checkpoint action-selection intervention

Post hoc motivation: the completed exact-current-model diagnostic improves all
eight E36 worlds with greedy action selection at the same exploration rate.
This tests R3 use of available predictions, not a memory or training repair.

Reuse all three final E36 training seeds, all eight worlds and 512 actions each.
Evaluate static_ttt, first_order_ttt, e2e_ttt and rolling counts: 80 cases.
Retain identical trained checkpoints, initial fast state, real update equations,
Bellman horizon6/discount.95, world/noise/goal seeds and exploration.1. Replace
softmax(Q/.1) by uniform choice among actions within absolute1e-6 of maximum Q.
Compute Bellman values in float64 for the intervention. All models retain the
same query count and learning work. No private map/noise rate enters decisions.
No retraining, hyperparameter search or favorable case selection is allowed.

Compare every case with its original soft-policy counterpart. Report actual
goals, prediction loss, map readout, logical storage, work and observed time.
Different policies change future evidence: this identifies the closed-loop
effect of the controller change, not equal-trajectory memory quality. Report
world-level reversals and shared-world dependence. These reused worlds are
diagnostic; an improvement requires fresh contingent-action families before
any general efficiency claim. Failure with an imperfect model remains useful.

e36_greedy_intervention.py hashes itself, the reused auditor and main inputs.
The source E36 manifest is preserved as an input descriptor; intervention.json
is authoritative for the supplemental policy and planned80cases. Original
training directories are read-only symlink inputs. A four-case preflight checks
mechanics on existing data without counting as new capability evidence.

Audit every real trajectory, gradient update, noninitial checkpoint and action
RNG using the existing audited state/physics checks. A separate explicit NumPy
Bellman implementation checks neural policy choices; any discrepancy above
3e-6 fails the audit, including near-tie ambiguity. Counts are replayed from
raw bounded histories using the substituted policy. Training is not rerun:
the completed training-audit hashes validate inherited checkpoints. No change
to the seven frozen main sources is permitted.
