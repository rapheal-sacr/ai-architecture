# Next consolidation test: preserve future learning, not only current answers

Read FALSIFICATION.md, HYPOTHESES-P1-RESULT.md and E36-RESULT.md first.
This is a proposed bounded falsifier for attachment H5/H6, not an executed
result or a new compression method. It attacks R2 retained distinctions and
R3 the cost/correctness of using and updating them. It does not replace the
remaining learned-goal, general-reasoning or actual-RSI requirements.

The proposal inspired by the attachment is to train migration against future
observed updates as well as current readout. For memory M, proposed compressed
state C(M), update operator U and subsequent real evidence tau, compare

    f(U(C(M), tau), q)  with  f(U(M, tau), q).

Matching f(C(M),q) to f(M,q) at tau empty is only one slice of this objective.
E35's local-selection/whole-trajectory reversal motivates the distinction, but
does not prove that learning this larger objective succeeds. The reference
learner can itself be wrong; downstream real outcomes and retention remain the
external criterion. Replaying futures costs work and can overfit a finite set.

## An exact structural reason to test it

For f(x)=abx and half squared error, rescaling (a,b) to (ca,b/c) preserves every
current prediction. Let e=abx-y. Ordinary same-rate SGD changes the product to

    a'b' = ab - eta*e*x*(a*a+b*b) + eta*eta*e*e*x*x*ab.

The quantity a*a+b*b changes under rescaling. Identical current functions can
therefore respond differently to the same subsequent observation. This is an
elementary parameterization dependence of SGD, not a novel theorem. It refutes
current-output equality as a sufficient certificate of future-update equality.
It neither demonstrates compression savings nor proves all migrations fail.

The implemented fast SwiGLU has an exact real-arithmetic counterpart:

    (SiLU(x W1) * (x W3)) W2.

Replacing W3 by c W3 and W2 by W2/c preserves this expression for every input.
The existing same-rate clipped SGD is not generally invariant to that change.
No information need be lost for later behavior to change. Thus such a failure
should not automatically be described as destructive memory compression.

## Bounded diagnostic before training a consolidation policy

Use the audited E36 first-order/E2E checkpoints on all three training seeds,
all eight worlds, at steps128/256/384: 144 origins. Do not select good origins.
Replay the next16 actually recorded observation/action/outcome tuples for each
origin, keeping actions fixed across counterfactual learners. This is an
equal-evidence learning intervention, not new closed-loop task success.

For c in {.25,1,4}, compare frozen updates, ordinary clipped SGD, and a known
coordinate-transport control. Query all12 state/action pairs before and after
the16 observations. Require the untransformed adaptive trajectory to reproduce
saved forecasts and the final map readout. Verify initial functional equality
and frozen equivalence; retain finite-precision discrepancies rather than
describing approximate equality as bitwise identity.

For diagonal parameter scaling p'=S p, transport the update using

    g = S g';  alpha = min(1, clip / ||g||);
    p'_next = p' - eta*alpha*S*S*g'.

This converts the gradient back to the original clipping metric before mapping
the step into new coordinates. S is supplied by the diagnostic transformation;
it is not a learned consolidation mechanism. Its additional metadata and work
must be counted. The scale1 control must agree with ordinary updates. Compare
transported and untransformed full trajectories; reject a claimed repair if
they diverge outside declared numerical tolerances.

This tests the sufficiency of output-preserving migration and whether explicit
optimizer transport explains a failure. It does not test memory-size reduction,
learned migration, indefinite retention, encoder compatibility or broad transfer.
If run, freeze source/tolerances before scoring, preserve every case, audit real
receipt provenance, and keep all source checkpoints unchanged.

## Architecture implication, still conditional

A durable learned-memory interface should describe both readable content and
the update procedure that gives that content future meaning. Versioning or
transporting that procedure might preserve useful behavior; learning a compact
surrogate might fail. This adds a falsifiable obligation to H5 rather than an
extra module with assumed benefits. Prior art includes distillation, optimizer
geometry, meta-learning and continual learning; novelty is not established.
