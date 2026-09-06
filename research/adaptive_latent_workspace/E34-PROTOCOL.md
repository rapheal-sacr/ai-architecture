# E34: stored competence or learning during the return?

Read FALSIFICATION.md and E33-RESULT.md first. E33 acquires its synthetic contexts
and benefits from carried weights, but return-cycle scoring includes updates
between chunks. E34 is a post hoc diagnostic on the same audited streams and
checkpoints, not a fresh generalization experiment. Freeze protocol_e34.json and
the scorer before executing counterfactuals. No new fitting or model selection.

For every three-seed/three-training-arm/twelve-stream source, compare five return
conditions: current weights with/without updates; archived first-A weights
with/without updates; and trained initialization without updates. Current means
the state at the end of the last interfering context. All conditions clear KV
at the already supplied session boundary and process the same returningA tokens.

Archived first-A restoration uses the experimenter's known context identity.
It is an oracle capability/selection comparison, with extra snapshot storage,
not an implemented autonomous memory or free repair. Interpret forgetting only
where initialA acquired the declared source gate. Keep static failures censored.

Report accuracy/NLL before the first update (first4 tokens), through the first
cycle(16) and across the full return. Frozen weights still receive causal token
context through attention; disabling parameter updates does not remove observed
context or freeze activations. It distinguishes read-time capability from further
gradient adaptation, not all possible definitions of retrieval/relearning.

All540 conditions run. Current-adaptive must reproduce the exact E33 return,
including final state. Frozen fast weights and all initialization parameters
must remain unchanged. Paired adaptive/frozen first chunks must match. No source
checkpoint is written; hashes verify source immutability. Charge forward work,
updates, time and archive tensor bytes, with inherited training still explicit.

`record_e34.py` validates persisted records and source-state selection. Adaptive
conditions replay exactly. Frozen logits are also compared against an independent
full-sequence, float64, explicitly indexed attention reference with the selected
fast weights installed as fixed parameters. This reference shares MLP/norm
primitives but not streaming caches or masks. Allow2e-4 absolute/1e-5 relative
logit tolerance for float32 versus float64 arithmetic; report the actual maximum.
Metrics and all expected cells must reconstruct. Tensor bytes exclude process
overhead; attention-score entries are not full FLOPs.

Development uses only E33's distinct small preflight source. All15 conditions
execute, retaining source replay, frozen-weight and pre-update equality checks.
The separate auditor checks six adaptive replays and nine dense references.
These establish the intervention's implementation, not acquired competence.

This experiment does not add a learned archive router, memory admission policy,
bounded compression or recursive procedure changes. It diagnoses what the next
integrated design would need to preserve and select. Even success remains far
short of the full autonomous long-horizon architecture goal.
