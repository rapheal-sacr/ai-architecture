# WAM-RX Milestone 2 foundation contracts

**Status:** frozen before E0.3/E0.4 and the multiview-memory run

These gates sit between the Milestone 1 authority kernel and every Milestone 2
compiler.

## Grounding and no-compounding

Every witness is typed. `event:<event_id>` is a ledger witness and must resolve
by the end of its atomic batch. `external:<authority>:<id>` identifies an
external witness, but does not change an inferred or synthetic event into an
observation.

A durable promotion is an event whose payload contains `promotion`. Its
transitive closure follows parents and ledger witnesses. Admission requires at
least one earlier `observed` event in that closure. Synthetic/inferred-only
chains are rolled back as one batch. This is narrower than general event
admission: ungrounded working hypotheses may be recorded as unverified, but may
not be promoted as durable capability.

## Temporal order

SQLite ledger sequence is authoritative transaction order. A caller-supplied
`transaction_time` is normalized to UTC before hashing and retained as metadata;
it cannot backdate an append ahead of an earlier transaction. `valid_from` and
`valid_to` represent real-world applicability and are also normalized to UTC.

The resolver consumes ledger order. It never compares timestamp strings and
never reads the wall clock. A replay of the same sequence and valid-time point
must produce one state hash.

## Artifact compatibility

Completeness is not compatibility. Every artifact read supplies an active
runtime policy for base weights, component versions, ontology, and verifier.
Exact matches pass. Mismatches fail closed unless a directional predecessor or
ontology-migration rule explicitly admits the stored version into the active
runtime.

Compatibility rules are part of runtime configuration and are not inferred from
version names. In the absence of a rule, rebuild.
