# A procedure that changes while its student continues learning

Read `FALSIFICATION.md` first. E9's reset students left a central gap: a procedure
could improve a fresh start while damaging established knowledge. E17 tests that
transfer boundary with frozen procedures. The next implementation will actually
change the procedure during one persistent student's lifetime.

The state is `(student, optimizer moments, inferred context, bounded experience,
procedure, procedure-update moments, predecessor)`. A student never receives a
regime ID or resets at a regime boundary. The procedure still produces bounded
per-tensor update multipliers. This is a restricted implementation of the larger
architecture; it does not learn arbitrary source code, its own evaluator, a
world model or semantic compression.

After a block of ordinary learning, retain a short actual update trace and its
starting student state. Differentiate through eight past updates and score the
result on the next already-observed batch and an old replay sample. The proposed
procedure update therefore uses observations available before its trial. Both
learning the current function and retaining old responses enter the objective.
The query loss must not include labels that have not yet arrived.

Make a temporary copy of the current student and moments, with the candidate
procedure. For the next sixteen incoming batches, both branches predict before
seeing the new outputs and then learn from them. The incumbent branch supplies
the operational predictions throughout this trial. Only after it ends may the
candidate replace the incumbent, if its trial prediction error improves and its
old-sample error stays within tolerance. Never replace the operational record
with the better branch in hindsight. Retain every rejection and all duplicate
work. Admission is fallible selection on finite data, not a retention guarantee.

Two outer proposal rules isolate self-application: conventional Adam-shaped
meta-updates and the incumbent procedure applying its learned multiplier to its
own parameter gradients. Both receive the same proposal opportunities and
admission rules. Frozen procedure and fixed optimizer baselines expose the cost
of the entire online procedure-learning mechanism. Inherited bootstrap work is
charged separately, not silently treated as free knowledge.

This specifically attacks destructive adaptation and wasted learning work. It
cannot identify a newly drawn hidden regime before evidence arrives, and it
cannot retain arbitrarily many independent facts in bounded state. A successful
short retrospective gradient or accepted branch does not imply future gain.
Recovery under subsequent changes and total work must decide that.

The first test uses full-feedback supervised streams, where both branches can
legitimately consume the same observed labels. In closed-loop control, different
actions would change their data; replaying a single real trajectory cannot give
both branches their counterfactual outcomes. A future agent must pay for actual
trials or validate an appropriate causal estimator. This supervised mechanism
must not be advertised as solving that problem.

Learned optimizers, differentiable update traces, replay, copied models and
candidate admission all have prior art. The specific online combination is a
research candidate; scientific novelty is unestablished. The test is designed
to reject it if extra ordinary meta-training explains the gain, if admission
does not predict later improvement, or if duplicate learning consumes more than
it returns. Open-ended recursive improvement remains a stronger unproved claim.
