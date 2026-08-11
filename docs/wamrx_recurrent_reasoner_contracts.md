# WAM-RX Milestone 3 recurrent-reasoner contracts

**Status:** frozen before task generation, E0.12, or neural model code

**Dependency:** Milestone 2 at `25b392a`

Milestone 3 isolates recurrence. It does not add neural memory, experts, adapter
consolidation, promotion automation, or self-improvement. The first executable
result is an assay-validation result, not a claim that recurrence works.

## Comparison

The registered arms are fixed depth, flat recurrence, and high/low hierarchical
recurrence. They share encoder, decoder, task examples, evidence interface,
optimizer budget, retrieval/tool budgets, and a 200 million FLOP maximum per
example. Registered parameter counts differ by less than one percent. Any
pre-training mismatch invalidates the run rather than becoming a covariate in
the analysis.

Unused adaptive-inference budget is recorded. It is not transferred into extra
retrieval calls, training examples, parameters, or updates. A recurrent arm
that wins only by consuming more FLOPs is test-time scaling, not an efficiency
result.

## Recurrent interface

Each execution begins with four immutable inputs:

- a problem representation and content hash;
- a stamped retrieval/analytic/graph evidence bundle;
- an explicit unresolved-constraint state;
- a hard macro, micro, FLOP, retrieval, and tool budget.

High state contains objective, decomposition, unresolved constraints, and a
progress estimate. Low state contains local computation, candidate answer,
retrieval/tool requests, and evidence references. Neither state is evidence.

Before every macro and micro recurrence, the reasoner must receive the original
problem hash and the complete revalidated evidence bundle. Every output records
the answer/action, evidence support, unresolved residuals, halt reason, exact
macro/micro counts, budget accounting, and trace hash. If the state disagrees
with evidence, evidence wins or the result remains unresolved.

## Depth protocol

Training macro and micro depths are independently sampled uniformly from
`{1, 2, 3, 4}`. Evaluation macro depths are `1, 2, 3, 4, 6, 8, 12`; depths 6,
8, and 12 are unseen during training. Initial recurrent state is a seeded 50/50
mixture of zeros and Gaussian noise with standard deviation 0.02.

The model receives intermediate-state supervision at weight 0.25. Training uses
full backpropagation through each sampled unroll and detaches only between
examples. Fixed training-loop and fixed inference-depth controls both use depth
4. Maximum execution is 12 macro steps, four micro steps per macro, and 48
micro steps total.

If accuracy peaks at training depth or extra loops cost more than one percentage
point, depth randomization plus early exit gets one registered repair. If that
repair fails, recurrence is rejected rather than scaled up.

## External residual and halting

A resolved halt is conjunctive: the learned head must choose stop and the
executable residual must be clear. The residual tracks unanswered constraints,
missing/conflicting evidence, unsatisfied tool postconditions, answer
instability, and remaining budget.

A learned stop with a non-empty residual forces another cycle while budget
remains. A learned continue with an empty residual also continues; this proves
the learned path is active. Exhausting a hard budget terminates with an
unresolved result. Resolved halt requires answer instability at or below 0.02.

Adaptive halting must save at least 20% median compute relative to maximum depth
while losing at most one percentage point of accuracy.

## Frozen assay

The three registered families are:

1. ordered modular checksums with held-out sequence lengths;
2. graph shortest-path problems with held-out graph structures;
3. multiview aggregation, contradiction, and constraint tasks with held-out
   operation compositions.

Frozen train, ID, and OOD splits are derived from versioned generators and
seeds. Content hashes, not only seeds, are registered. OOD axes must be disjoint
from training axes. Results are always reported per family, depth, and protected
region before pooling.

E0.12 must make all eight controls fire: no reinjection, fixed-loop training,
corrupt high state, no external halt, contradictory evidence, depth beyond
training, incompatible memory, and missing protected-region evidence. Until
E0.12 passes, model code is out of scope.

## Promotion thresholds

A recurrent arm needs a strictly positive 95% paired-confidence lower bound on
OOD improvement, no more than two points of ID or protected-region regression,
no more than one point of decline from extra loops, and complete evidence/tool
journals. Five paired seeds is the minimum.

Hierarchy must beat flat recurrence on at least one family and remain within
two points on every other family. Memory invalidation and manipulation failures
override accuracy. The decision is therefore one of: adopt hierarchy, adopt
flat recurrence, retain fixed depth, or retain a narrow specialist. None of
those decisions authorizes native neural memory or autonomous promotion.

## Post-run decision

E0.14 completed with `COMPLETE_RETAIN_FIXED`. The four-block
`fixed-depth-v1` arm is retained as the general reasoner core. Flat and
hierarchical recurrence failed the frozen promotion gate and remain only as
negative evidence. The post-run selection is hash-bound in
`contracts/wamrx_reasoner_selection_v1.json` and checked by
`tools/check_recurrent_selection.py`; it does not amend the frozen comparison or
authorize any separately gated memory, expert, consolidation, or autonomous
promotion mechanism.
