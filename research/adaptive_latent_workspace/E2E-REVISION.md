# E2E revision: train the future update, then challenge persistence

Read FALSIFICATION.md first. The user's renewed E2E emphasis changes the next
design priority. E30–E32's conventional LoRA updates do not implement the E2E
meta-objective. No result here establishes novelty, reliable RSI, general
reasoning or efficient continual learning. Qwen1.5B source was cloned, but no
weights were downloaded or tested; that direction is paused.

## Source and scope

PDF: `/home/rapheal/Documents/AI Architecture/Papers/e2e.pdf`, SHA256
`b7fd0d8b1d58ad9ebcb57973729cf077e0e301c36ed766bd8e3f8df8e253ecfa`.
Clone: WD `2026-09-05/repos/e2e`, commit
`a4fc4788ace38e29b5067916d4f4be33da894085`, from
https://github.com/test-time-training/e2e. The prior falsification/corpus record
was read first. This refresh checks methods, figures, evaluation and recipe
against source; it is not a numerical replication of the large runs.

For each chunk, score next tokens using current weights, then update only late
adaptive MLPs using newly observed labels. Outer training differentiates through
those updates to improve initialization and fixed weights for future prediction.
Stopping the update derivative changes the objective. Static training followed
by test-time SGD is another control.

Source uses causal sliding-window attention with window at least chunk size.
A fast SwiGLU residual precedes a static SwiGLU residual. Attention, norms and
static MLP are fixed in the inner loop; all participate in outer training.
`spec_inner` excludes even the adaptive MLP's norms. Defaults include Q/K
normalization and pre/post-sublayer RMS normalization. Inner SGD has global
clipping and no momentum; outer training uses AdamW.

The paper reports predictive compression, not lossless memory. At128K, Table2
reports full attention / E2E correctness of.99/.06 for passkeys,.86/.05 for
numbers and.64/.03 for UUIDs. Figure6's NLL advantage is greatest early,
including before the first test-time update. Its entire advantage therefore
cannot be attributed to distant-memory retrieval. Reported128K prefill speedup
is2.7x onH100; outer training costs3.4x full attention at8K in the H200
comparison. These are reported numbers, not local measurements. Separate
context-length finetunes, data filtering and smaller-model long-run instability
also qualify the result. The source evaluator copies initialization and discards
the final model carry per sequence. Persistent cross-session retention, arbitrary
exact facts and recursive improvement are not established by that evaluator.

## Proposed integration and roots

Replace ordinary adaptation with an update-trained execution core. Its proposed
complete state is `(slow initialization, persistent fast weights, local KV,
bounded evidence records, learned selection/procedure state)`. Clearing local
context must not silently delete persistent knowledge or procedure state.

| Proposed part | Root attacked | Required falsification |
|---|---|---|
| Meta-train fast updates through changing contexts and returns |2: interference;3: acquisition cost|Compare reset and carry training on exact paired observations; establish acquisition before retention; charge second-order work|
| Preserve unpredictable, consequential residual facts in bounded records |2: lost distinctions|Rare/independent facts and changed queries at matched complete storage/access cost|
| Learn evidence selection, retrieval and update decisions |1: identifying evidence;3: access|Hidden shifts, misleading generated text, no free context labels or uncharged routing|
| Learn planning/procedure choices through observed consequences |3: useful computation;1: evidence gathering|Longer closed-loop compositions; preserve executed versus advisory actions; compare fixed updates and extra training|
| Admit persistent procedure changes using subsequent outcomes |All, through the learning procedure|Actual multi-generation self-application, persistent learner state, held-out consequences and all failed trial costs|

This integration has not established novelty: fast gradient memory, episodic
exceptions, meta-learning and trial admission have prior art. Only the first
row's small core/test is newly implemented. Writing the remaining rows does not
implement learned compression, planning or autonomous procedure revision.

The operational binding candidate remains **recovery cost relative to useful
knowledge lifetime**. E2E attacks the cost of converting observations into
predictive state; residual records attack distinctions compression discards.
Neither identifies independent unobserved facts or avoids growing storage for
arbitrarily many independent facts. A universal numeric rate bound remains
unmeasured. E33 retains thresholded acquisition/return results alongside NLL.

## Implemented core and source boundaries

`e2e_core.py` preserves causal SWA, tied embeddings, RMS/QK normalization,
sequential fast/static SwiGLUs, late-MLP clipped SGD, pre-update CE and full
outer differentiation. It explicitly returns final fast state. All layers run
chunk by chunk instead of upstream's separate fixed-prefix pass. Absolute RoPE
positions replace relative cache rotation. Dimensions, data and initialization
scale are smaller/different. No upstream JAX numerical comparison, released
checkpoint, fused kernel, mixed precision or rematerialization is used.

`preflight_e2e_core.py` checks:

- Independent dense attention agrees within2.45e-15;16 future-token interventions
  leave earlier predictions exactly unchanged; scoring precedes the update.
- Saved weights/caches resume exactly, including two adaptive layers whose fast
  history affects later-layer caches.
- Central-difference outer gradients agree under active/inactive clipping
  (errors below6.1e-12 in these directions), and across a session boundary with
  two adaptive layers (3.89e-10). First-order surrogate gradients differ.
- A real outer update changes prefix and fast initialization. Inner updates
  preserve initialization/static parameters while same-query logits change
  by.7584. Frozen downstream weights do not mean unchanged outputs. The random
  model had no acquired task: this is not measured forgetting of learned skills.
- Persisted inference tensors occupy13,312 bytes after either16 or64 tokens;
  model parameters add83,328 bytes in this float64 preflight. Outputs, Python
  objects, allocator overhead and training graphs are excluded. This is not a
  constant total-process RAM or training-memory claim.

Initial omitted Q/K and post-sublayer norms were caught and corrected before
scoring. Both clipping branches and fast-dependent caches were then checked.
Finite-direction checks are not a proof for all parameters/inputs. Results and
hashes: `results/e2e_core_preflight.json`. E33-PROTOCOL.md challenges persistence;
a grammar pass cannot replace long-horizon autonomous tasks or the full goal.

## Representation compatibility of persistent evidence

MSA-SOURCE-AUDIT.md adds a concrete integration obligation: fast/slow weight
updates must not silently invalidate stored evidence encodings. A fixed-prefix
write/read boundary before fast layers is a candidate to test; higher-layer
features can depend on fast history. Re-encoding on slow updates must be charged,
with the conditional throughput bound in CONSTRAINTS.md. Compatibility and
metadata checks do not establish useful retention or efficient learned compression.

## Implemented extension and next integration boundary

E33 now establishes acquisition on its grammar family, and E34 isolates retained
function and interference with frozen-state interventions. E35 implements a
bounded protected fast-state bank, future-observation probation and causal
selection. Its admission/selection controller is fixed, not learned. The shared
prefix is numerically exact only for the implemented single final adaptive MLP;
both that MLP and the subsequent static residual must be evaluated per candidate.
E35's scored results require the independent audit before interpretation.

FRONTIER-ARCHITECTURE-AUDIT.md supplies a stronger account of the proposed
execution core. A recurrent sequence state can reduce the growing context cost,
periodic selective evidence access can retain consequential distinctions, and
shared reasoning layers can spend more work on difficult decisions. These are
existing mechanisms with costs and failure modes, not architectural novelty.
Replacing the fixed prefix with KDA/GDN or adding loops is not yet implemented
in the E2E archive, and the current results cannot be credited to those additions.

The full candidate should distinguish seven states: slow encoder/init version,
short recurrent/context state, active fast parameters, protected parameter
snapshots, bounded evidence records, temporary reasoning workspace, and persistent
learning-procedure state. The current E35 core implements only context state,
active parameters and a protected bank behind a fixed slow version. Its probe
is an admission sample, not a general episodic evidence store.

For an action-capable integration, encode the latest real observation and the
previous executed action/outcome; retrieve evidence under an explicit budget;
refine candidate consequences in the temporary workspace; execute one selected
action; then learn from the actual new outcome. Imagined outcomes can guide
planning but must not be silently relabeled as observed targets. This preserves
the causal distinction the imitation/source audits required. E2E meta-training
must optimize later actual decisions through the learning updates, while
evaluation measures acquisition, recurring-goal success, failed exploration,
retention and complete work. Better next-token loss is only an intermediate test.

A tractable next integration target is an action-conditioned transition learner
on previously unseen worlds, using its learned predictions for bounded planning.
It must face unannounced world changes and returning goals, without hidden maps
or context IDs. An observed-transition planner is a strong specialized control;
a private-world oracle is diagnostic only. This is not yet a frozen experiment
or a replacement of the general-reasoning requirement with grammar prediction.
E30–E32's action failures remain part of the completion boundary.

Slow representation improvement introduces an unresolved obligation: current
archives are meaningful only with their original slow base. Hash rejection
prevents silent reuse but does not make learning a new base possible cheaply.
Migration must pay for replay/re-encoding, compatible transport or temporary
coexistence, and must test old queries after its compression budget is exhausted.
Procedure self-revision similarly needs multiple actual generations and fresh
outcomes against equally funded conventional learning. Neither ability follows
from a successful fixed archive selector. The full goal remains unachieved.
