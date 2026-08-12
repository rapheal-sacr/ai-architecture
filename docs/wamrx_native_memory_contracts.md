# WAM-RX Milestone 4 · authority-limited native working memory

Milestone 4 begins from the immutable Milestone 3 selection, not from a new
reasoner search. Local tag `wamrx-m3-fixed-depth-v1` identifies commit
`0bb80a8`, whose selection record binds `fixed-depth-v1`, macro depth 4, to
E0.14 result SHA-256
`274bee04023d44a1d1cc361715204dd62e18f3b3c7944825f58f31c9d16d85a6`.
The native-memory contract cannot change that core, its depth, or its selected
status.

## Boundary

Native memory is ephemeral working state. It has fixed slot and serialized-byte
limits and is bound to owner, session, task, epoch, expiry, core version,
component version, and ontology version. Reset clears all slots and rotates the
epoch; expiry clears all slots and rejects access. Checkpoints are canonical,
hash-bound, and exact-identity only.

A future learned gate may choose exactly four operations: `remember`, `update`,
`merge`, and `forget`. Those decisions remain advisory to the non-neural
boundary in `wamrx/native_memory.py`. The boundary validates evidence, enforces
capacity, prevents silent key collisions, protects rare-region state from
automatic decay, and records exact analytical memory FLOPs and canonical slot
bytes.

Every model call must present a freshly rebuilt current-frontier evidence
bundle containing all support for every active slot. If a supporting event is
tombstoned, retracted, refuted, unverified, missing, or ontology-incompatible,
the affected slot is disabled before the read. An earlier read also fails once
the ledger frontier changes.

Native memory has no ledger append method. It may return only a
`candidate-only` object whose support manifest contains ledger event IDs and
reaches an observed root. A slot or memory identifier is never independent
evidence. An external witnessed admission path would have to convert a
candidate into a durable event.

## Frozen comparison

The later neural comparison has three arms:

1. the fixed-depth core with explicit retrieval, analytic, and graph views;
2. the same core with a deterministic exact-key session cache;
3. the same core with one minimal learned gate over sixteen fixed slots.

The frozen registry contains nine task families: delayed recall, correction,
temporal update, distractor-heavy sessions, similar-fact interference, context
overflow, reset/task switching, rare-region retention, and poisoned or
contradictory input. It deterministically generates 72 train, 36 ID, and 36 OOD
tasks. Every OOD family axis is disjoint from training. E0.16 later consumed
this frozen population without changing it.

The five E0.14 fixed-depth checkpoints are separately bound by seed, byte count,
file SHA-256, metadata hash, and state hash in
`contracts/wamrx_native_memory_core_checkpoints_v1.json`. They are now present
at the registered ignored Linux relocation paths; the artifact checker reports
`READY` with no identity error. E0.16 still cannot substitute newly initialized
or retrained core weights.

## E0.15 structural result

E0.15 passes all eleven kill checks and all eleven registered manipulations.
The compliant fixture exercises all four operations, emits a witnessed
candidate without changing the ledger, round-trips a bound checkpoint, and
stays within both caps. Controls fire for latent durable writes, omitted
reinjection, tombstoned old state, reset and cross-user leakage, key collision,
capacity overflow, stale model/ontology checkpoints, unverified poison,
protected-state decay, interrupted checkpoint creation, and memory cited as
evidence.

This result authorizes only the smallest separately registered neural-memory
comparison. It is not evidence that learned memory improves accuracy, beats
explicit multiview memory, or merits persistent use. MoE, LoRA, continual
weight updates, consolidation, and self-improvement remain outside scope.

## E0.16 registered-run boundary

The additive E0.16 contract freezes the three-arm paired comparison before any
evaluation metric is read. The only learned component is a 472-by-4 affine gate
(1,892 parameters) trained for 400 balanced updates per core seed. The selected
four-block core is load-only and hash-checked before and after gate training.

The runner reported ID/OOD exactness, family and protected-region strata,
realized compute and storage, explicit-retrieval comparisons, correction
latency, reset leakage, capacity saturation, unsupported durable writes, and
post-invalidation behavior. Gate checkpoints and the append-only task journal
are bound to the full run manifest. The two-update resume preflight passed
bitwise without reading evaluation metrics.

E0.16 completed all five seeds, 2,000 gate updates, and 1,080 evaluation rows.
Every arm scored 4/36 per seed on both ID and OOD. Learned-minus-simple accuracy
deltas were zero, while the learned arm's compute-normalized OOD deltas were
negative and its correction gate failed. All eleven manipulations passed;
reset leakage, unsupported durable writes, and stale post-invalidation
emissions were zero. The terminal decision is
`COMPLETE_RETAIN_EXPLICIT_MULTIVIEW`, hash-bound by
`contracts/wamrx_native_memory_selection_v1.json`. The next memory-policy scope
is explicit ledger-derived views only; details are in
`docs/wamrx_native_memory_run_protocol.md`.
