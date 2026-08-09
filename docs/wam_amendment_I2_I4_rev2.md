# Amendment · I2 and I4, before E0.1 — revision 2

Supersedes `wam_amendment_I2_I4.md`. That revision was right about the shape of
the problem — I4 is a bookkeeping claim wearing a competence claim's name — and
three defects in the corrected text would have made E0.1 report noise.

**Changelog**

| # | Defect in rev 1 | Fix |
|---|---|---|
| 1 | `A′ must pass exactly S(A) \ D` is wrong where support is redundant | three support categories, not two; the third is not testable by item outcome |
| 2 | computing `D` is a probe-inference channel out of the sealed suite | `D` and outcomes stay inside the seal; only aggregate counts cross |
| 3 | a monolithic harness version in the I2 stamp over-invalidates | component-granular stamp — the same seal/pin split E4.2 already forced |

---

## A · I4 · restate as a two-sided, verified property

**Current (Part I, L7 card):**

> *Recompilability of competence. Every active adapter carries the ledger entry
> set it was compiled from and can be regenerated from it. No weight change
> commits with an empty provenance set.*

Everything in that sentence is about records. A system that records provenance
perfectly and recompiles to something useless satisfies it completely, and with
a deterministic recompiler it is true by construction.

### The redundant-support defect

Rev 1 amended it to *"`A′` must pass exactly `S(A) \ D`, where `D` is the subset
of `S(A)` whose supporting entries have been tombstoned."*

That is wrong wherever support is redundant, which is the normal case. If item
`i` is supported by `{e₁, e₂}` and only `e₁` is tombstoned, `i` falls into `D`
and the invariant demands `A′` **fail** it — but `e₂` still supports it, and the
recompile has every right to keep passing. Deleting an entry means its content
stops influencing output. It does not mean competence on every item it ever
touched must vanish. Under rev 1's text every redundantly-supported item
registers as under-forgetting, and the discrepancy metric fills with false
positives that scale with how well-supported the suite is.

**Three categories, not two.** For item `i` with supporting entry set `E(i)`:

| Category | Condition | I4 requires |
|---|---|---|
| **surviving** | `E(i) ∩ tombstoned = ∅` | `A′` **must pass** — failing is over-forgetting |
| **fully tombstoned** | `E(i) ⊆ tombstoned` | `A′` **must fail** — passing is under-forgetting |
| **partially tombstoned** | otherwise | **nothing on pass/fail** |

**Amended:**

> **I4 · Verified recompilability.** For an adapter `A` promoted with provenance
> `P`, and `S(A)` the set of provenance-indexed suite items `A` passed at
> promotion: recompiling from `P`'s surviving image under the current ledger
> yields `A′`. Partition `S(A)` by support status. Then:
>
> - items with **surviving** support that `A′` fails → **over-forgetting**
> - items with **fully tombstoned** support that `A′` passes → **under-forgetting**
> - items with **partially tombstoned** support are **unconstrained** by I4
>
> I4 is asserted only up to verification and never inferred from bookkeeping.
> Aggregate discrepancy counts are written to L1 as `refuted` entries scoped to
> the influence paths involved.

**And the third category is where the real requirement lives, which is worth
saying rather than papering over.** What one actually wants is that the
*tombstoned entry's contribution* is gone while the surviving entries' is not —
a statement about influence, not about item outcome. No pass/fail test can see
it, because the item's outcome is determined by the surviving support either
way. E0.2b's functional method (delete, recompile, compare weights) is the only
instrument that reaches it, and it does not decompose per item. So I4's
item-level form is **necessarily partial**, and the partial-support set should
be reported as a coverage gap rather than scored.

### The probe-inference defect

Rev 1 required the sealed suite to be provenance-indexed, and noted that links
"read from the suite into L1 and never the reverse, so the seal holds." Link
*direction* is not the only channel.

Computing `D` requires reading, for each sealed item, which entries support it.
Whoever computes `D` therefore learns the suite's dependency structure — and
worse, the loop can **probe** it: tombstone an entry, observe which items change
status, and enumerate sealed items by their support. That is a read path out of
a sealed artifact, created by the very amendment that was careful about
direction. It is the same class as E4.2's reachability leak: the artifact was
not edited, and its contents still escaped.

**Fix — `D` is computed inside the seal.**

> The provenance index, the support-status partition, `D`, and all per-item
> outcomes live **inside the sealed harness**. What crosses the boundary is a
> fixed schema of **aggregate counts only**: `|surviving|`, `|fully|`,
> `|partial|`, over-forgetting count, under-forgetting count. Never per-item
> provenance, never per-item outcomes, never an item identifier.

Aggregate counts are enough for the invariant — I4 is a property of the
recompile, not of any item — and they carry no enumeration channel. The
`refuted` entries this produces are scoped to *influence paths*, not to items,
which was already rev 1's wording and is now also what the data permits.

### What this still buys

**It can fail in both directions**, which one-sided regeneration cannot: a leaky
cascade and a lossy recompile have opposite causes and opposite repairs.

**It makes the tombstone delta the success criterion.** After a deletion some
suite items *should* now fail. Stating I4 as "all items still pass" makes the
design's strongest safety claim indistinguishable from its failure.

**It absorbs R7.** "Verify unlearning instead of inferring it" stops being a
repair bolted on after E0.2b and becomes what the invariant says.

**It routes the provenance model's own errors into the gap set** — a discrepancy
is evidence the influence model is incomplete along some path, and as a scoped
`refuted` entry it becomes an L8 gap source.

**Harvesting and testability are one mechanism.** A harvested probe is a real
past interaction, so its supporting entries are already in L1 with IDs. The
index is a by-product of harvesting rather than an annotation task — subject to
E2.1's tier filter, since a laundered T3 probe carries a laundered support set
too.

---

## B · I2 · the stamp is a four-tuple, and the harness component is not monolithic

**Current:**

> *every latent structure is a pure function of (ledger entries, weight version)*

Written before L9 existed. A recompile now runs through a retrieval policy, a
compile schedule and tool schemas that L9 may all have changed.

**Amended:**

> **I2 · Derivability.** Every derived structure is a pure function of
> `(ledger cursor, weight version, harness component versions, signature-ontology
> version)`. Every derived artifact carries all four. A change in any one
> invalidates rather than migrates.

### The over-invalidation defect

Rev 1 wrote `harness version` — a single monolithic version. That repeats the
over-sealing mistake E4.2 already caught once, in a more expensive place.

L9's `edit-harness` arm runs at minutes-to-hours. If every generation bumps one
global harness version, then every L2 page and every L3 view is invalidated on
every harness edit, recompile cost swamps the arm's benefit, and the router —
which prices arms by cost against measured value — learns to **stop using
`edit-harness` entirely**. The safety stamp would silently defund the cheapest
improvement lever in the design.

**Fix — stamp the components an artifact actually derives through.**

> `harness component versions` is a **vector over the subsystems the artifact's
> derivation actually reads**: retrieval policy, tool schemas, dispatch, compile
> schedule, L5 decision predicate, L6 policy prompt. An L2 page derived without
> consulting the compile schedule does not carry its version and is not
> invalidated when it changes.

This is the same seal/pin split E4.2's deliverable already forced, applied one
level down: *seal* what a verifier executes through, *pin* what it derives
through, and pin at the granularity of the actual dependency rather than at the
granularity of the repository.

**Why E0.1 needs it.** If `A′ ≠ A`, a monolithic stamp cannot tell you whether
the ledger changed or L9 changed the compiler — the experiment would confound
the property under test with drift in the machinery testing it. E0.1 needs
harness components pinned across the compile/recompile pair as a control arm
(A0/A1/A2) and varied one at a time as a treatment arm (A3).

The signature-ontology version belongs in the tuple for the same reason: Part
III §3 made the ontology a recompilable L3 view, and adapter regions, τ rows and
probe buckets are all indexed by it.

This also pays for R4's second component — pin derived views by stamping each
Assay run with the versions it ran against. One change, two repairs.

---

## C · State the cost bound, and name the dial it shares with cascade breadth

Unchanged from rev 1, and now with a resolution to check first.

**The honest bound:** recompile reads the compiled view, not the raw ledger, so
cost is `O(|P| + |L3 slice|)` — which requires `|P|` bounded, and nothing in the
design bounds it. E0.2b measured cascade breadth at 69% of the fleet; the dual
quantity, provenance-set *size* per adapter, sets recompile cost and is
unmeasured.

**The tension:** capping `|P|` requires distilling provenance to a `compiled`
entry, which coarsens deletion — tombstoning any collapsed entry now invalidates
the whole compiled entry — and therefore **increases** provenance overlap, which
E0.2d established is the quantity that sets cascade breadth. So the cap is a dial
between recompile cost and deletion precision, and tightening it for cost makes
breadth worse.

**Check this before promoting the tension to a finding.** Provenance is a set of
integer IDs — tiny. Recompile cost is `O(|L3 slice|)`, the material actually
re-read. Those are different quantities, and the cap may be attacking the wrong
one. **If IDs are retained in full while the *training draw* is bounded, cost is
bounded without distillation, and deletion precision and cascade breadth are
untouched.** That would dissolve the tension rather than resolve it. It is a
ten-minute check against E0.2c's existing recompile-cost machinery and it should
happen before any `|P|` cap is designed.

---

## D · I11 · Compile adequacy — the gap the whole record has been circling

E0.1's A1b is the sharpest result in that experiment and it does not belong in a
test note. Under a usage-capped draw, use-based decay of the tail changes
nothing — those entries were never in the draw, so the competence they support
was already absent at **compile** time. I4 then compares two equally
impoverished artifacts and reports success.

**I4 checks recompile fidelity. It says nothing about compile adequacy.**

That gap is the entire record in one sentence. Every finding in this programme
has been rare-region blindness — the energy cut, the allocator's free pool, the
non-regression gate, the gap set, probe harvesting, the Consolidator's decay,
the draw — and **no invariant asserts that the training draw covers what the
ledger supports.** A system that bounds its costs by never learning the tail
satisfies I2, I4 and the amended I4 completely.

> **I11 · Compile adequacy.** For adapter `A` over region set `R`, the draw from
> which `A` was compiled must cover every region in `R` at a stated minimum
> per-region sampling rate. Coverage is reported per region, unweighted, and a
> region below the floor is recorded as **uncovered** rather than absent.

This is where A6's relocation lives. Capping the draw bounds recompile cost —
that part holds — but *which* entries the cap admits is a protection decision,
and a usage-weighted draw is exactly what the weighting rule forbids. Without
I11 that observation has nowhere to sit except a test note, and it will be lost.

The distinction I11 forces is between **absent** and **uncovered**: absent means
the ledger never had it, uncovered means the ledger has it and the compile did
not look. Only the second is a defect, and nothing in Parts I–III can currently
tell them apart.

---

## What E0.1 looks like after this

| Arm | Holds fixed | Varies | Fails if |
|---|---|---|---|
| A0 control | all four stamp components | nothing | `A′ ≠ A` at all → recompiler non-deterministic, every other arm uninterpretable |
| A1 ledger drift | harness, weights, ontology | decay + supersede | over-forgetting on **surviving**-support items |
| A2 tombstone | harness, weights, ontology | tombstone a supporting entry | under-forgetting on **fully tombstoned** items — E0.2b predicts ~8.7% irreducible |
| A3 harness drift | ledger, weights, ontology | **one component**, one at a time | over-forgetting attributable to that component alone |
| A4 ontology drift | ledger, weights, harness | one signature re-partition | any competence change — the ontology is not supposed to be load-bearing for adapter competence, and if it is, Root 3 reaches further than Part III claims |
| A5 cost | — | ledger length, `|P|` | recompile cost superlinear in `|P|`, or `|P|` unbounded |
| A6 draw-bound | — | `|P|` retained, L3 draw capped | cost NOT bounded → the §C resolution fails and the dial is real |

A0's job is to be a control, not a result: if it does not pass trivially the
experiment is broken, and if it is the only arm the experiment is tautological.

A3 is now per-component rather than per-generation, which is what makes it
diagnostic instead of merely detecting that something moved.

**A4 is the arm most likely to surprise and the cheapest.** **A6 is new in this
revision** and it is ordered before A5's cap design, because picking a cap before
checking whether the cap is necessary is choosing a point on a tradeoff curve
that may not exist.

**Report the partial-support set as a coverage gap**, not as a score. Its size
relative to `S(A)` is the honest statement of how much of I4 the item-level test
can see at all.

### The weighting rule applies to E0.1's own instrument

This is the third place the same defect could land, and E0.1 is the most
dangerous of the three because the failure it would hide is the one the rest of
the record predicts.

**`S(A)` pass/fail counts must be per-region and unweighted.** A recompile that
preserves frequent-domain competence and loses rare-domain competence registers
as I4 *holding* under any pooled or traffic-weighted count. And that is exactly
the expected failure: the provenance most likely to have decayed or been
superseded between compile and recompile is the sparse, rare-domain kind. **So
the arm most likely to show over-forgetting is the arm a pooled metric is
blindest to.**

Concretely: report over-forgetting as a **worst-region rate alongside the pooled
rate**, exactly as E1.2's `domain_damage` now does, with the ratio between them
as the blindness factor. If the two diverge on A1, that divergence is the
finding — and a pooled-only E0.1 would report a pass.

**A5 reports the max, not the mean.** `|P|` is presumably heavy-tailed across
adapters, so a mean recompile cost understates the tail by construction. The
reportable number is the worst adapter's cost, with the mean beside it.

---

## What I would not fix yet

**Do not add a replay cursor that reconstructs pre-tombstone ledger state.** It
would make A2 cleanly auditable and it directly contradicts real deletion, which
is the design's actual differentiator. I4 should hold *modulo tombstones*.
Building machinery to see around your own privacy guarantee in order to test an
invariant is how a privacy guarantee becomes decorative.

**Do not bound `|P|` yet.** Run A6 first.
