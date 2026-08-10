# WAM-RX milestone 1 contracts

**Status:** frozen version 1 implementation contract
**Machine-readable registry:** `contracts/wamrx_milestone1.json`

This milestone is intentionally mostly non-neural. It establishes the authority
and audit boundary on which recurrence, native memory, experts, consolidation,
and self-improvement must later depend.

## Authority boundary

The event ledger is authoritative. Resolved snapshots, retrieval indexes,
selection journals, analytics, graphs, summaries, skill cards, adapters, and
router state are derived and replaceable. A derived artifact can affect a read
only while its content hash, frontier, version tuple, and support manifest pass
read-time validation.

Events are immutable. Corrections are an atomic pair: a retraction/refutation
targeting an earlier event plus a new observed/asserted event. Deletion is a
tombstone targeting an earlier event. Neither operation erases the audit trail.
Version-1 events explicitly carry actor, source and verifier identity, verifier
class, modality, payload hash, parent/target links, provenance witnesses,
confidence, policies, ontology/resolver versions, and an optional signature.
Any event claiming a non-`unverified` verifier class without a verifier identity
is rejected.

Validity time and transaction time are explicit, timezone-qualified inputs.
The resolver never reads the wall clock. Tombstone and retraction dominate
refutation, which dominates verification. Contradictions remain in the resolved
snapshot as typed status and control-event links.

## Artifact identity

Every artifact records:

- ledger frontier sequence and chain hash;
- base-weight version (`none` for the non-neural prototype);
- component/compiler versions;
- ontology version;
- verifier version;
- build-configuration hash and content hash;
- supporting, contradicting, and considered candidate event IDs.

Missing lineage is a hard read failure. Indexes use item-level support manifests
so one tombstone can immediately disable one document while an asynchronous
rebuild proceeds. Other artifacts default to conservative all-support
invalidation.

## Registered gates

The exact thresholds and K1–K13 kill criteria are frozen in the JSON registry.
The milestone passes only if:

1. replay hashes are identical and batch interruption exposes no partial write;
2. unstamped, hash-mismatched, frontier-invalid, and unsupported artifacts fail
   closed;
3. every protected region has 100% coverage and zero registered query
   distortion in the synthetic world;
4. contradictions are preserved, tombstoned evidence is unavailable
   immediately, surviving evidence remains available, and repaired output equals
   a clean rebuild;
5. every retrieval journals the complete candidate set, all component scores,
   filters, versions, selections, and top-k boundary margin.

Pooled values are always reported but never override a worst-region failure.
The biased compiler manipulation must demonstrate that distinction: it is
expected to pass pooled coverage while failing the rare region.

## Recovery and rollback

A failed event batch rolls back to the prior chain frontier. A failed integrity
check stops compilation. Invalid evidence immediately disables affected reads.
Repair creates a new artifact from the surviving ledger image and compares it
with a separate clean-from-scratch build. The system never edits an existing
artifact or chooses a threshold after seeing its result.

## Deferred scope

There is no neural native memory, recurrent reasoner, LoRA/MoE expert, continual
weight update, autonomous mutation, network service, or distributed consensus
in milestone 1. Their contracts may be implemented only after these executable
gates pass.
