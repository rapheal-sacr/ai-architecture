# E33: persistent E2E after actual acquisition

Read FALSIFICATION.md and E2E-REVISION.md first. Freeze this protocol and JSON
before scored generation. Three seeds/three training arms execute128 outer
updates each; final checkpoints only, no score-based stopping or selection.
This is a small mechanism assay, not the full architecture or paper replication.

Each context is a fresh uniformly permuted directed cycle over16 symbols.
Observed consecutive symbols provide targets. No hidden map or context ID
enters the learner. One supplied initial token starts each session, followed by
64 scored tokens. Training is A→B→A, with fresh maps each update and independently
sampled starting offsets each session. Boundaries clear local attention in every
arm: boundary information is provided, so change-point discovery is not tested.
Saved latent labels/maps are solely for auditing.

All arms share exact initialization and ordered tokens:

- `static`: ordinary next-token training, no inner updates.
- `e2e_reset`: differentiate through within-session updates; reset fast weights
  at each session boundary, as in the source's per-sequence evaluation.
- `e2e_carry`: differentiate through updates across A→B→A, clearing only KV.

Four32-wide layers, one adaptive suffix MLP,64-wide SwiGLUs, four heads,
window4, chunk4, clipped innerSGD.3 and outer AdamW.001. Exact fields/seeds are
in protocol_e33.json. Frozen four-layer SWA has maximum local displacement12,
while a transition repeats after16 symbols. This restricts token-only recall;
it does not prevent parameters from learning structure across examples.

Evaluation has four fresh streams per seed/regime: A→B→A with four cycles per
session; A→B→A with16 cycles per session; and A→B→C→D→E→A with four cycles.
The static checkpoint is tested frozen, with resetting ordinary SGD and with
carried SGD. Both E2E checkpoints receive reset and carry evaluation. Inference
uses the same update equation throughout. All fresh maps are checked disjoint
from training and each other; within-stream repeats are intentional. Policies
share evaluation streams, not independent replications.

Before describing a return failure as forgetting, firstA must reach75% final-
cycle accuracy. Also report each intervening context's acquisition. Separate
first-return-cycle accuracy/NLL, last-cycle reacquisition and all-token NLL.
Early uncertainty after a switch may be identifying-evidence cost. Mean loss
reduction without competence does not establish successful continual learning.
Preserve every seed/regime reversal; no aggregate threshold hides those cells.

A specialized control stores the last actually observed successor of each symbol
without context labels, predicting before writing. Report accuracy and query/
update counts; deterministic wrong predictions do not receive fabricated finite
NLL. Logical16-entry storage excludes Python overhead. This is a hand-designed
task control, not a general agent. Preserve any efficiency domination it shows.

Comparisons match examples, not FLOPs/time. Charge inner gradients, outer steps,
forward tokens, actually computed attention scores, training and evaluation
seconds. Counts do not measure all second-order FLOPs. Arms rotate each outer
step. E32 may run concurrently on GPU: wall times are observed shared-machine
measurements. Saved outputs and outer-training graphs add memory beyond bounded
fast/KV inference tensors; no process-peak RAM claim follows.

Artifacts preserve complete streams, every training loss/gradient norm, final
model/Adam/RNG, all evaluation logits/losses and session fast/KV state.
`record_e33.py` checks independent transition references, replays every outer
update to exact model/Adam, and every session to exact logits/state. It checks CE
with separate float64 log-sum-exp and table predictions with a history scan.
Replay shares the numerically checked core; it is not an independent JAX port.

Development seed1933 uses two steps and smaller dimensions. Six updates and21
sessions replay exactly. Separate core checks validate causality, meta-gradients
and persistence. These are functional checks, not task competence. No scored
result exists at this freeze.

Keep these limits after any pass: synthetic grammar, explicit boundaries,
deterministic contexts, known training return pattern, tiny scale and a fixed
human-written optimizer. No autonomous actions, UUID/passkey benchmark, learned
episodic compression, broader reasoning or recursive procedure revision is
implemented here. The full goal remains active and unachieved.
