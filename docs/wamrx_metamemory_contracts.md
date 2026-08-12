# WAM-RX explicit-view metamemory and compression boundary

**Status:** E0.17 `PASS`; learned comparison E0.18 `NOT_RUN`

E0.17 begins from the hash-bound E0.16 selection of
`explicit-multiview-v1`. It does not reopen native memory, the selected
four-block reasoner, or any E0.16 metric. Metamemory sees only ledger-derived
retrieval, analytic, temporal, and belief/constraint views.

## Advisory actions

The policy interface has exactly seven actions: ignore, stage, link, retrieve,
summarize, structure, and request evidence. Every output is candidate-only.
The policy object has no ledger append method, cannot promote a staged skill,
and cannot cite policy, summary, memory, or slot identifiers as evidence.

Ignore applies only to unverified noise. Evidence requests name a gap rather
than manufacturing support. Link, retrieve, summarize, structure, and stage
require usable observed ledger roots. Staging additionally requires at least
three verified repetitions of one procedure identity.

## Protected compression

A compression artifact is a rebuildable advisory explicit view, not a new
evidence root. It preserves exact manifests for every source item and region,
contradiction, refutation, temporal qualifier, supporting event, and unresolved
item. The source digest, ontology, and ledger frontier are bound. Rebuilding
from the same explicit frontier is byte-for-byte equivalent; any frontier or
ontology drift, including tombstoned support, disables the artifact before use.

Compression is capped at 64 source items and 32,768 canonical bytes. Policy
FLOPs, scanned items, decision bytes, artifact bytes, and skill-candidate bytes
are accounted separately.

## Frozen population and E0.17 result

Seven deterministic families map one-to-one to the seven actions. The registry
contains 56 train, 28 ID, and 28 OOD tasks; every OOD axis is disjoint from
training. E0.17 uses only one train fixture per family for structural
falsification and reads no performance metric.

All twelve kill checks and thirteen manipulations pass. Controls reject
fabricated or metamemory-derived support; dropped regions, contradictions,
refutations, temporal qualifiers, unresolved items, and source evidence; stale
frontiers and ontologies; self-promoting or under-repeated skill candidates;
and both capacity escapes. The compliant path exercises all seven actions,
preserves the ledger row count, rebuilds compression exactly, and records every
operation and byte.

## Next comparison boundary

E0.18 is registered only at the mechanism level and remains `NOT_RUN`. Its
three arms are full-resolution explicit views, the deterministic heuristic,
and one minimal learned policy. A learned policy can be selected only if it
improves long-horizon task/query quality or reduces retrieval/storage cost by
at least 10% without any ID, OOD, protected-region, grounding, contradiction,
temporal, unresolved, rebuild, or deletion regression.

Because E0.16 rejected native memory, every E0.18 arm must operate solely over
explicit ledger-derived views. MoE is not blocked by the negative native-memory
result, but it remains a separate later contract. No E0.17 result authorizes
learned policy training, MoE, LoRA, continual consolidation, promotion, or
self-improvement.
