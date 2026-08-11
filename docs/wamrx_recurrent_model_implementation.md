# WAM-RX recurrent-reasoner model implementation

**Status:** implemented and gradient-smoke-tested; registered comparison NOT_RUN

The model slice follows the pre-model freeze at commit `ad09b2d`. It does not
change the E0.12 task hashes, promotion thresholds, arm budgets, or interface
schemas.

## Runtime and configuration

The optional model runtime is MLX 0.32.0 on Apple Silicon, pinned in
`requirements-m3.txt`. Milestones 1 and 2 remain standard-library-only because
the MLX modules are isolated from the base `wamrx` package imports.

`contracts/wamrx_recurrent_training_v1.json` fixes the remaining training
details before a comparison run: 2,000 AdamW updates, batch size 16, five paired
seeds, learning-rate schedule, clipping, initial-state mixture, depth sampling,
intermediate loss, and halt targets. The model comparison has not consumed
those updates yet.

## Shared architecture

All arms use the same canonical-JSON byte codec, 260-token vocabulary, 1,536
input positions, 64 output positions, 472-dimensional prelude, and
non-autoregressive byte decoder. Every registered task fits without silent
truncation.

Each arm owns four residual MLP blocks with 1,888 hidden dimensions:

- fixed depth applies four distinct blocks once;
- flat recurrence applies the same four-block core at every macro recurrence;
- hierarchy applies three high-level blocks per macro step and one weight-tied
  low-level block per micro step.

Small used low-rank residuals and partial-width biases close the sub-percent
capacity gaps exactly. The resulting trainable counts are 8,388,608,
8,372,224, and 8,404,992—exactly the pre-registered arm targets.

## Evidence, recurrence, and halting

The immutable problem/evidence encoding is computed once and reinjected through
the capacity-matched input path at every recurrent step. It is never overwritten
by recurrent state. Hierarchical execution exposes separate high and low neural
states.

The executor revalidates every evidence view and protected region before each
step. It combines the learned halt logit with the executable task residual,
answer instability, and hard FLOP/step budget. Outputs include a support
manifest, unresolved residual, halt reason, exact macro/micro counts, trace
hash, and compute record. Untrained or incorrect output therefore terminates as
unresolved rather than acquiring authority from a model state.

Training halt targets are tied to each deterministic task's registered required
reasoning steps, not a single global loop count. Short sampled unrolls contain
no fabricated stop target.

## Smoke result and scope

The unregistered smoke runner performs two shared-schedule optimizer updates per
arm. It verifies gradients, exact parameter counts, decoding, shared example and
depth schedules, compute accounting, and fail-closed trace emission. The smoke
sample scores 0/2 for every arm, as expected after only two updates; that number
is deliberately not a result artifact and has no bearing on promotion.

The registered run protocol now labels this primary comparison precisely as
equal-update/equal-data under a common inference cap, not equal realized
compute. Estimated and realized training/inference FLOPs are separate fields,
and a compute-normalized secondary is frozen. E0.13 also verifies bitwise
checkpoint/resume equivalence over a train-only 20-update prefix.

E0.14 has now completed the five paired seeds, 2,000 updates per arm, full
ID/OOD/depth/region reporting, manipulation ablations, and promotion audit. Its
terminal decision is `COMPLETE_RETAIN_FIXED`. The four-block fixed-depth arm is
the selected general core; flat and hierarchical recurrence are retained only
as negative experimental evidence. The selection does not authorize native
memory, MoE, adapters, consolidation, or self-improvement.
