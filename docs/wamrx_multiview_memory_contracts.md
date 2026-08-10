# WAM-RX multiview-memory contracts

**Status:** frozen before E0.11

This slice follows the authoritative-memory kernel at commit `b2cd1b0`. It adds
two derived representations and no neural components.

## Analytic memory

An event may carry an `analytic` payload with a record type, entity, effective
time, dimensions, numeric measures, scalar fields, and explicitly uncertain
candidate fields. Every compiled row has one authoritative event as support.
Candidate fields retain confidence and their witness IDs; they are not silently
promoted to ordinary fields. Analytic effective times normalize to UTC before
the compiler orders rows or performs time-window operations.

The view implements temporal filters, count/sum/mean/min/max, grouped
aggregation, window comparison, trend, and ranking. A query result carries all
supporting event IDs and a conservative compiler candidate set. Before returning,
every operation appends an immutable query record containing the complete
candidate rows, filters, operation, versions, selected support, and result hash.
Retraction or
tombstone disables a stale row immediately, and repaired output must equal an
independent clean rebuild.

## Belief and constraint graph

An event may carry a `claim` payload containing subject, relation, object, and
validity conditions. The compiler retains active, verified, refuted, retracted,
and tombstoned edges. Competing objects under the same subject/relation/condition
key form a contradiction set; refuting one resolves the competition without
erasing it.

Constraints are query-time objects, not evidence. Each requirement is satisfied,
violated, missing, or conflicting. Missing/conflicting requirements produce a
next-step plan, but that generated plan is never added to the support manifest.
Only ledger event IDs can support a constraint conclusion.

## Structural comparison

E0.11 uses four exact synthetic tasks: aggregation, temporal comparison,
contradiction resolution, and multi-constraint selection. The baseline is
deliberately narrow and named precisely: it succeeds only when an exact answer
already appears in a retrieved hit. It is a test of the retrieval API alone—not
of retrieval plus an LLM or custom arithmetic.

The multiview candidate must solve all four tasks, improve at least two over that
baseline, retain 100% evidence lineage in each protected region, reject the
tail-blind and contradiction-dropping controls, disable deleted support
immediately, reject unjournaled analytic reads, and match clean rebuilds exactly.

## Registered negative controls

E0.11 must deliberately exercise malformed schema extraction, rare-region
lineage omission, derived-evidence laundering, an unregistered stale ontology,
live contradictory claims, tombstoned support, and a registered analytic query
that never commits a journal record. A passing happy path without all seven
controls is not a passing Milestone 2 assay.
