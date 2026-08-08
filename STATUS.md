# WAM · status by design element

**The inverse view.** `PLAN.md` is ordered by experiment, which is right for
building the record and wrong for reading it. This page is ordered by the
*design*: every load-bearing claim, its status, the constant it rests on, and
what would change it.

It is also the honest answer to *"is this architecture sound?"* — the question
the harness was built to be able to answer at all.

| Status | Meaning |
|---|---|
| **verified** | measured, holds |
| **assumed** | not measured; rests on a stated constant |
| **broken** | measured, fails, **repair named** |
| **open** | cannot currently be evaluated |

---

## The short answer

**The ledger-first thesis is untouched.** Nothing in the record attacks the claim
that one authoritative append-only log with derived views is the right shape.
Every failure found is a *stated mechanism being the wrong shape for its job*,
and all but one has a named repair.

**One gap has no repair yet**, and it is not that the repair failed — it is that
R9 cannot be evaluated until three things about fleet composition are stated
*and* I7 is expressed as the rate it already is.

**The meta-repair needed a repair on first contact with its own example.**
Auditing R9's draft declaration found that `trades against` named the right
quantity in the right shape but omitted *the threshold* — and the threshold is a
decide quantity that inverts the finding. The column now carries the threshold's
status (measured / decided / **unset**), and `unset` blocks as hard as a blank.
Found by inspection, so it refines the amendment rather than failing it — but it
is the first evidence that four columns may not be four.

**The meta-repair is registered prospectively, not claimed retrospectively.**
Its eight retrospective catches are a fit to the data it was designed from. The
real test: R9's declaration exists in draft, and the amendment fails if any later
experiment finds a coupling, resource or tier requirement it omitted.

**The meta-repair matters more than any remaining experiment.** Repairs are
gap-productive: R9 arrived carrying two unstated couplings, which is the defect
the repairs exist to fix. `docs/wam_amendment_mechanism_declaration.md` makes
every mechanism declare, before it is written, what tier it requires, what it
consumes, what it trades against, and what it assumes about parts it does not
own. Retrospectively that would have surfaced eight of this record's findings as
questions before they cost a build.

**Five numbers decide the rest, and they are two different kinds** — a
distinction worth making explicit, because one kind is cheap and the other is
not.

---

## Invariants

| Invariant | Status | Rests on | What would change it |
|---|---|---|---|
| **I1** grounding | assumed | never tested directly | E0.4, unrun |
| **I2** derivability | **broken → repair** | stamp was a pair, not a four-tuple | amendment rev 2 §B: component-granular stamp. Vindicated by E0.1 A3 — one harness component moves competence 21.6% |
| **I3** no-compounding | assumed | untested under synthetic experience | E0.3, unrun |
| **I4** recompilability | **broken → repair** | was a bookkeeping claim wearing a competence claim's name | E0.1: 6.7% pooled over-forgetting hiding **79.8% worst-region**, 12.4× blind. Amended to a two-sided verified property with three support categories |
| **I5** anchored improvement | **broken → repair** | blast-radius rule bounds *which* thresholds, not their *values* | E4.2. R4: bound the values; pin derived views; seal only harness code a verifier executes through |
| **I6** refutation permanence | assumed | untested | E4.3/E4.4, unrun |
| **I7** compile adequacy | **new — proposed** | did not exist | E0.1 A1b: I4 checks recompile fidelity and says nothing about compile adequacy. A system that bounds cost by never learning the tail satisfies I2 and I4 completely |

---

## L1–L3 · ledger, index, compiled views

| Claim | Status | Rests on | What would change it |
|---|---|---|---|
| tombstone cascade reaches the weights | **broken → repair** | provenance recorded at compile time | E0.2b: `transitive` recalls **0.913, not 1.0**; no set-based closure reaches 1.0, because the residual dependency is a retrieval never run. R7: verify, don't infer |
| L3 admission (cos ≤ 0.93) controls cascade breadth | **broken → no repair yet** | conflated content with provenance | E0.2d: content cosine moves 0.498, breadth moves 0.009. The design has **no lever on breadth**. R9 proposed — see *open* below |
| signature ontology is not load-bearing for competence | **broken → repair** | never tested | E0.1 A4: a pure re-partition moves competence **21.7% pooled, 41.6% worst**. Root 3 reaches into Root 1 |
| "maximise off-diagonal mass" is a partition objective | **broken → repair** | monotone in fineness | E3.3: argmax is total atomisation. R5: score by held-out predictive error — *validated only on nested partitions* |
| Consolidator decay is safe | **broken → repair** | "decay unused entries" is frequency-keyed | E5.1: a global use-based cut removes **60–70% of the rarest regions**. Weighting rule, site 9 |

---

## L4–L7 · cache, deliberation, policy, weights

| Claim | Status | Rests on | What would change it |
|---|---|---|---|
| `occupied rank = #{σ_k > ε}` is a computed budget | **broken → repair** | ε as an eigenvalue threshold | E1.1 fails; **E1.1b passes** under GPM's energy criterion. R1 demonstrated. §A states it in the one form that does not work |
| the budget holds per domain | **broken → repair** | traffic-weighted aggregation | E1.1c: 12/16 domains exceed the bar while the mean passes; Spearman −0.965, monotone in rarity. ρ becomes a *measured* parameter set from worst-domain interference |
| free rank is ample | **assumed** | subscription ratio and per-region overlap of real traffic | E1.1d: the boundary sits at **subscription 1.0×**, exactly where the whole record was measured. Overlap 0.7 moves it past 3.0×. **Rig B measurement** |
| three-λ unanimity makes release safe | **broken → no clean repair** | — | E1.2 fails **both ways**: unanimity collapses the budget 69 → 6.3 and starves 3.29×; allocating on the short estimator writes into committed directions a traffic-weighted check cannot see. Tightening one breaks the other |
| posterior variance is an epistemic gap detector | **broken → repair** | conflates epistemic with aleatoric | E1.4: a coin-flip region takes **51%** of the practice budget. R2: score by *reducible* variance. But R2 needs outcomes, so it relocates to Root 2 |
| L7 adapters compile in "hours–days" | **broken → measured** | draw size, model size, hardware — none stated | **EB.1**: at the established draw cap, **18 minutes at 1.5B, not 8 hours**. `H = draw × tokens × epochs / throughput` |
| net-transfer ranking accumulates abstractions | **conditional** | patches must not spill *more* than skills by ~0.03 | E3.1 + E3.1b: the margin *grows* under noise; T4's inversion was an **asymmetry artifact**. The condition is differential, not absolute — much weaker |
| the transfer **matrix** earns its cost | **open** | never tested in either direction | E3.1c: the *statistic* wins; no arm consults accumulated history. τ storage, the partition objective and the ontology view all hang off the untested matrix |

---

## L8–L9 · practice field, harness field

| Claim | Status | Rests on | What would change it |
|---|---|---|---|
| blast-radius fixed point seals the Assay | **broken → repair** | enumerates artifacts, not reachability or values | E4.2: all four Assay tiers execute through L9-editable code. R4 |
| typed arm tuples expand the search space | **assumed** | untested | E4.5, unrun |

---

## The Assay

| Claim | Status | Rests on | What would change it |
|---|---|---|---|
| probe harvesting widens the verifiable surface | **broken → repair** | "resolves with a verified outcome" names no tier | E2.1: **29–63%** of an unfiltered suite is laundered T3 judge opinion. R6: one tier predicate. Yield is high (33–68%), contradicting this plan's own prediction |
| harvested suites are representative | **conditional** | correlation between checkability and difficulty — *swept, not measured* | E2.1: gap-set-weighted yield falls to 0.165 at high correlation. **Rig B measurement** |
| the staged ladder makes an archive affordable | **broken → repair** | the filter ranks by a **mean** | E2.3: recall of the full rung's top decile falls to **0.349**; ~100% of good rare specialists dropped vs ~36% of generalists — *identical in the unbiased control*, so not a weighting problem. R11 |
| verifier synthesis moves domains T3 → T1 | **open** | declared a research bet by the design itself | E2.2, unrun. Part III §0 calls this the binding constraint on everything |

---

## The five numbers — measure these, decide those

**MEASURE** — these need data, and no amount of writing settles them:

| Quantity | Gates | Status |
|---|---|---|
| **`H`** recompile wall-clock | the C3∧C4 window | **resolved.** EB.1: 18 min at the draw cap, not 8h. The window goes EMPTY → OPEN, so E5.1's infeasibility at the design's own profile was carried entirely by the assumption |
| **support redundancy** `m of k` | C6 | **procedure + decision rule written.** ≥5 supporting entries and C6 stops binding. Awaiting an interaction corpus |
| **per-region overlap + subscription** | whether any Root 1 number transfers | **open — Rig B.** E1.1d puts the boundary at subscription 1.0×, exactly where the whole record was measured. Nothing else can settle which side real traffic sits on |
| **cascade breadth `β`** — or the fleet↔bank coupling that would let it be computed | whether an operating envelope exists | **open.** R9 is the only proposed repair and it is not yet evaluable |

**DECIDE** — these need a sentence, not data, and they are the cheapest thing
remaining:

| Quantity | Gates | Status |
|---|---|---|
| **restoration latency tolerance `L`** | C4 | a product requirement. Never written down |
| **I7 per-region density floor** | R9's coverage side | a policy choice. E0.2e sweeps it: at a floor of **1** there is no tension at any τ; at **5** the tension is total even at the loosest. "100% of regions below floor" is a statement about the 3.0 I chose |

**Two sentences and one coupling make R9 evaluable.** That is the whole distance
between the last open gap and a drawable curve, and none of it needs an
experiment.

## The one open gap

**R9 · provenance-aware admission.** E0.2e: tightening the cap reduces
cards-per-entry (9.18 → 1.00 — the mechanism works) *and* shrinks the bank, which
raises how much of the adapter fleet each surviving card reaches. The directions
are **opposed at every pool structure swept**, so the net depends on a coupling
Parts I–III do not specify.

Three things must be stated before R9 can be designed:

1. how many distinct cards an adapter's training draw uses
2. whether that is an absolute count or a fraction of the bank
3. whether fleet size tracks bank size

**And the coverage side is unevaluable too.** I earlier reported that R9's
expected tension with I7 "does not bind, coverage stays at 1.000." That was an
artifact of the metric: coverage was measured as *at least one card per region*,
a binary, while I7 is written as a **rate**. Re-measured as per-region density,
the tension is severe — tightening `tau` takes density from 25 → 1 and pushes
**100% of regions below the floor**, worst region falling fastest.

The tension was invisible, not absent. And the quantity that got binarised was
per-region density on rare regions, which is what this entire record has been
about.

So R9 is **doubly unevaluable**: the breadth side needs three specification
decisions, the coverage side needs I7 stated as the rate it already is.

---

## How much to trust this

**Seventeen rig bugs across twenty-two experiments.** The rate is flat, and
that is the wrong statistic to worry about — each experiment is new code doing
something not done before, so a *converging* bug rate would mean the experiments
were becoming more similar to each other, which would mean they were exploring
less. Flat rate against rising subtlety is the healthy signature.

The statistic that matters is **bugs that reached a conclusion**, and those are
four withdrawals clustered early:

| | |
|---|---|
| **severity** | falling. B4 survived a full write-up and produced a published headline. B14/B15 were caught mid-sweep by the manipulation check. B16 was caught because the scale was two orders off a number already in the record |
| **detection latency** | falling. Inspection → contradiction → *"is this a parameter I chose?"* → *"why is this so extreme / going the wrong way?"* → the record itself functioning as an error detector, which only works once the record exists |

The single most productive check has been **"why is this number so extreme, or
going the wrong way?"** — B7, B9, B11, B12, B13, B16. Cross-checks between two
measurements have caught none; the manipulation check has caught three.

Read the retraction record (`claims/claims.yaml`, `bugs:`) alongside the results
— not because the rate is alarming, but because the *reversals* are the most
informative entries in the file.

The strongest single regularity: **every mechanism specified without stating its
tension has been found out by an experiment.** That now includes R9, which was
proposed as a repair and arrives with the same defect.
