# WAM-RX registered recurrent-run protocol

**Status:** complete; terminal decision `COMPLETE_RETAIN_FIXED`

This protocol closes the gap between the structural E0.12 assay and the first
accuracy-bearing model run. It is an additive versioned contract; it does not
rewrite E0.12, its task hashes, or its promotion tolerances.

## What the primary comparison means

The primary experiment is equal-update and equal-data, with identical
seed-specific schedules, optimizer settings, evidence interfaces, and a common
200M maximum inference-FLOP cap. It is not described as equal realized compute.
The final checkpoint after exactly 2,000 updates is evaluated; validation-based
checkpoint selection is prohibited.

Final-depth inference is the primary policy. Fixed depth executes its four
blocks. The recurrent arms use the deepest registered macro depth that fits the
common cap with one final readout; hierarchical micro depth is one. Every other
registered depth is reported separately and is not averaged into the primary
claim.

Adaptive halting is a distinct analysis. Its maximum-depth reference and its
adaptive arm both pay for a decoder and executable-residual readout at every
position. Their largest feasible registered depth can therefore be shallower
than primary final-depth inference. This prevents halting overhead from being
silently charged to one policy but not the other.

## Realized-compute accounting

The registered run record supplements the original per-task trace record with
four explicit totals:

- estimated training FLOPs: the frozen schedule before execution;
- realized training FLOPs: successfully completed optimizer updates only;
- estimated inference FLOPs: every requested evaluation before execution;
- realized inference FLOPs: recurrent positions and readouts actually executed.

The registered training estimator is
`batch × 3 × differentiable-forward FLOPs + 17 × parameters` per update. The
factor three is one forward plus two backward equivalents; the parameter term
covers clipping, AdamW, and the deterministic state-layout canonicalization
needed for bitwise resume. These are deterministic analytical architecture
counts, not device-profiler measurements. Wall time is recorded separately.

The primary comparison preserves the frozen 2,000 updates and equal examples.
A compute-normalized secondary uses the latest 50-update learning-curve
checkpoint not exceeding the smallest final training-compute total. At
inference, it chooses the deepest observed position not exceeding the fixed
arm's realized final-depth FLOPs. It never interpolates accuracy. A primary gain
that disappears under this control is classified as training-compute scaling,
test-time scaling, or both—not architectural efficiency.

## Frozen statistics

The paired unit is one seed-level candidate-minus-reference accuracy delta.
There are exactly five registered pairs. Each family is scored first; pooled
accuracy is the unweighted mean of the algorithmic, structured, and multiview
family accuracies within a seed.

All bounds are one-sided paired Student-t lower bounds at 95%. Holm step-down
controls familywise error separately for:

- six OOD family claims: two recurrent arms versus fixed across three families;
- three hierarchy-versus-flat family claims;
- two pooled ID non-inferiority claims;
- eight protected-region non-inferiority claims.

ID and protected-region margins are -0.02. Hierarchy's nonwinning-family margin
against flat is also -0.02. Protected-region accuracy micro-aggregates ID and
OOD correct/example counts within each seed and region before forming the paired
delta. No depth averaging enters the headline claim;
depths 1/2/3/4/6/8/12 form the registered robustness curve.

## Checkpoint, resume, and failure rules

The runner checkpoints every 50 updates. A checkpoint contains model weights,
AdamW moments, optimizer step and learning rate, completed update, schedule and
seed identity, config/split/model hashes, examples seen, and cumulative realized
training FLOPs.

Resume is valid only when every identity and hash matches. The same failed seed
may be retried from its latest valid checkpoint; a replacement seed is
forbidden. Model initialization, batch order, sampled unrolls, and randomized
recurrent states are all seed-derived and recorded.

NaN or non-finite training produces `INVALID`. A contract/code/hash mismatch
also produces `INVALID`. Interruption, memory exhaustion, or incomplete
evaluation produces `INCOMPLETE`. Missing seeds are never dropped and are never
imputed as zero. Status precedence is `INVALID`, then `INCOMPLETE`, then one
explicit final decision, with `NOT_RUN` used only before execution.

## Train-only operational preflight

E0.13 runs the flat arm at seed 41001 for 20 updates with batch size 16. It
compares an uninterrupted run against interruption after update 10 followed by
checkpoint restoration. It never loads ID/OOD evaluation metrics.

The preflight correctly failed during repair. First, lexical reconstruction of
the nested AdamW state changed gradient-reduction order by a few float32 ulps.
Restoring the model's original parameter-tree order removed that path. A stricter
repeat then localized checkpoint-sensitive atomic accumulation in the repeated
token-embedding backward. The final repair pools deterministic token histograms
through the same embedding weights and canonicalizes model/optimizer storage
after each update. Two consecutive reruns pass with exact schedules, per-update
losses, model and optimizer state hashes, final checkpoint bytes, examples,
updates, and compute totals.

The preflight realizes 73,961,927,680 analytical training FLOPs. One checkpoint
is 100,479,866 bytes. Fifteen final checkpoints therefore need about 1.51 GB;
retaining every 50-update checkpoint for the compute-normalized learning curve
needs about 60.29 GB before evaluation journals. The measured train-only wall
time is recorded in the E0.13 result as an operational estimate, not a stable
performance claim.

## Runner boundary

`rig_a/experiments/e0_14_recurrent_model_comparison.py` is inert unless passed
`--execute-registered`. It trains every arm on the identical seed schedule,
hashes final checkpoints, evaluates every family/split/depth/region in
final-depth plus maximum/adaptive modes, validates trace and evidence journals,
runs all eight manipulation paths, applies the frozen statistics and secondary
compute control, and writes an explicit terminal status.

## Terminal result and selection

E0.14 completed every registered arm/seed, stratum, journal, manipulation, and
compute control without invalid or incomplete reasons. Neither recurrent arm is
promotion-eligible. The terminal decision is `COMPLETE_RETAIN_FIXED`, which
selects the four-block `fixed-depth-v1` core and stops recurrent development.

`contracts/wamrx_reasoner_selection_v1.json` binds that choice to the exact
result SHA-256 and frozen manifest. `tools/check_recurrent_selection.py` verifies
the binding without importing MLX or re-reading performance metrics. Flat and
hierarchical recurrence remain in the repository as reproducible negative
evidence. Neural memory, MoE, adapters, consolidation, and self-improvement
remain blocked and require separate contracts.
