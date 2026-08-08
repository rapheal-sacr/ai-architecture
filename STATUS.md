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
R9 cannot be evaluated until three things about fleet composition are stated.

**Four numbers decide the rest.** One is measured. One has a procedure and a
decision rule, waiting on data. One is a product requirement. One is the gap.

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

## The four numbers

| Quantity | Gates | Status |
|---|---|---|
| **`H`** recompile wall-clock | the C3∧C4 window | **resolved.** 18 min at the draw cap, not 8h. The window goes EMPTY → OPEN, so E5.1's infeasibility at the design's own profile was carried entirely by the assumption |
| **support redundancy** `m of k` | C6 | **procedure + decision rule written.** If real probes rest on ≥5 entries, C6 stops binding. Awaiting an interaction corpus |
| **latency tolerance `L`** | C4 | **not a measurement** — a product requirement to write down |
| **cascade breadth `β`** | whether any window exists | **open.** R9 is the only proposed repair and it cannot yet be evaluated |

---

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

The tension R9 was *expected* to trade against — I7 coverage — does not bind:
coverage stays at 1.000 throughout. The blocking unknown is the fleet coupling.

---

## How much to trust this

**Sixteen rig bugs across twenty-one experiments**, four withdrawn conclusions,
two reversals. That is not a converging error rate, and it is the main reason to
read the retraction record (`claims/claims.yaml`, `bugs:`) alongside the results.

Both serious errors that *survived a write-up* were mine, not the architecture's,
and one manufactured a headline. The mechanism that has caught the most is a
single question — **"why is this number so extreme / why does it go the wrong
way?"** — which found B7, B9, B11, B12, B13 and B16. Cross-checks between two
measurements have never caught one.

The strongest single regularity: **every mechanism specified without stating its
tension has been found out by an experiment.** That now includes R9, which was
proposed as a repair and arrives with the same defect.
