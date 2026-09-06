# Adding a persistent learning procedure

This is a proposed extension, not an achieved recursive self-improvement result.
Read `FALSIFICATION.md` first. The updated user goal requires more than retaining
student weights or applying a fixed replay/renewal rule.

## State and update roles

- **Skill state:** reusable model parameters, temporary adaptations and a live
  inferred context. E2/E5 provide limited evidence for protecting these updates.
- **Experience state:** bounded observed trajectories and outcome samples. E6
  shows why passing checks on this sample cannot guarantee preserved behavior.
- **Procedure state:** learned parameters governing updates, memory admission,
  consolidation and allocation of computation. These persist across tasks and
  can themselves change from measured learning outcomes.
- **Procedure archive:** previous procedure versions and their measured costs,
  kept separately from the current candidate. A failed procedure update must not
  erase the functioning predecessor or its falsification record.

An outer episode measures how a proposed procedure affects subsequent learning
and retention. It supplies learning signals for the procedure state. A further
self-application step lets the learned update mechanism propose changes to its
own parameters. That is a concrete mechanism to test; it is not evidence of
indefinite improvement or the ability to invent arbitrary new architectures.

The measurement process remains outside the proposed self-modification boundary.
Changing the evaluator, counting training examples as unseen tests, ignoring
failed trials or replacing quality with a convenient proxy would invalidate the
claim rather than constitute improvement.

## E9's deliberately limited first implementation

E9 implements a persistent neural update policy that changes per-tensor learning
rates as a function of gradients, weight scale, momentum, step and loss. It is
bootstrapped by differentiating through short student learning trajectories.
Then the learned policy processes meta-gradients of its own parameters to
propose changed versions of itself. Independent fresh validation tasks and a
small retained validation sample decide admission. Every trial is recorded.

The final evaluation compares the self-applied procedure against its frozen
bootstrap copy, a default fixed optimizer and a fixed-rate optimizer selected on
development tasks. It tests longer unrolls and changed input/function families.
Those final tasks do not enter meta-training or admission. Bootstrap and
selection costs count toward any later efficiency claim.

This test does **not** reset the procedure state between student tasks, but it
does reset student weights. It therefore does not demonstrate persistent
task-specific knowledge. It also learns within a fixed update-policy family and
uses a fixed admission rule. Zero useful admitted self-updates means the
self-application hypothesis failed; successful bootstrap meta-learning cannot
be relabeled as recursive self-improvement.

## Prior art and paper connections

The cloned [learned_optimization](https://github.com/google/learned_optimization)
source at `9561bb68c880ddc4b4eeba7a6ec82c25fe1530d5` already implements learned
update policies, per-parameter MLPs and unrolled meta-gradients. Inspected source:
`learned_optimization/learned_optimizers/mlp_lopt.py` and
`learned_optimization/outer_trainers/full_grad.py`. E9 is a small independent
PyTorch experiment, not a port or reproduction of those published results, and
learning an optimizer is not claimed as novel.

The supplied TTT-E2E paper motivates aligning training with later adaptation;
Titans motivates persistent adaptive neural state. DGM motivates preserving an
archive of candidate procedures and testing descendants empirically. RecHarness
motivates charging unsuccessful outer trials and separating selection from final
evaluation. Macaron motivates protected skill modules. None of those papers,
individually or in combination, establishes that this proposed recursive loop
will improve itself safely, efficiently or indefinitely.

## Failure tests required beyond E9

1. Does the self-applied update improve fresh-task learning over the frozen
   procedure, or merely pass reused validation samples?
2. Does it preserve earlier competence under long unrolls, input scaling and
   changed functions? Short-unroll meta-learning can learn a brittle fast start.
3. Does the gain repay the procedure-training, selection and admission work?
   Lower inner-loop error alone is insufficient.
4. Can the procedure learn memory admission or consolidation decisions that
   explain both E2's need for separation and E4's need for sharing?
5. Does it improve closed-loop task success when its actions change the available
   evidence? E8 already shows a supervised-learning gain need not protect a
   policy during sparse-reward adaptation.
6. Can improvement persist across multiple generations without narrowing the
   task distribution or silently expanding memory/compute budgets?

Until these survive, the recursive extension remains a falsifiable research
proposal and the overall goal remains unachieved.
