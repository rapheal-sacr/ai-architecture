# Amendment · what else to fix before continuing

Ten fixes, ordered by leverage. Four are one-line changes. One is structural and
would have caught three of the programme's findings by inspection, before any of
them needed an experiment. A closing section lists two things to *cut*, because the
design has been growing faster than its evidence.

Companion to the I2/I4 amendment. That one was required for E0.1 specifically; these
are not blocking any single experiment, which is why they are easy to postpone
indefinitely.

---

## 1 · Structural · every mechanism declares the verifier tier it needs to function

**This is the fix that pays for the others.**

Right now, verifier tiers appear in exactly one place: the promotion gate's floor
per blast radius. T0 harness, T1 memory, T2 weights. That table governs what a
mechanism is *allowed to promote*.

It says nothing about what a mechanism needs in order to **work at all**, and
several mechanisms silently need a great deal:

| Mechanism | Declared tier | Tier it actually requires | Where this surfaced |
|---|---|---|---|
| gap set from L4 variance | none — "free" | **T1** once the reducible form is used, since a variance *decline* needs per-region outcomes at two times | E1.4 → R2 |
| transfer matrix `T` | none | **T2** — off-target Δ on probe regions | Part II §B, never stated |
| counterfactual replay | is T1 | T1 | fine |
| compositional ranking | — | **T2**, and interactions cannot be read off first-order Δ at all | E3.1c |
| probe harvesting | "verified outcome" | **T0/T1 filter**, absent in the spec | E2.1, 29–63% laundered |
| credit via typed transitions | none | **T1** to know an outcome changed | Part II §C |

Three of the programme's findings — R2's relocation, the compositional ceiling,
tier laundering — are the same error in three costumes: *a mechanism was described
as free or cheap while silently requiring verification the domain may not have.*
Each cost an experiment to find. A column in a table would have found all three.

**The fix:** add `requires_tier` to every mechanism table in Parts I–III, distinct
from the promotion floor. Then the claim in Part I §3 that L4's covariance gives a
gap detector "for free — no new estimator" becomes visibly false in T3 domains,
which is where it matters, and it becomes false on the page rather than in month
nine.

**What this reveals immediately.** In a T3-only domain, the surviving gap sources
are only the **event-driven** ones — `refuted` without successor, KG conflict,
active request, L5 escalation. Every *statistical* source needs outcomes. That is a
real narrowing of L8 and it is currently nowhere in the design.

---

## 2 · Structural · inertness must be an observable state

Follows from (1) and is the thing I would most want on a dashboard.

If a domain is T3-only, then: reducible variance cannot be computed, `T` cannot be
populated, the counterfactual gate cannot run, and nothing can promote past a skill
card. So the improvement loop is **inert** there.

The design does not say what happens in that case. Which means it does nothing,
silently, and reports success — because every metric in Part II §10 is computed over
promotions that occurred, and in an inert domain there are none to be wrong about.
The system stops improving in exactly the domains that need it most and the
dashboard stays green.

**The fix:** a domain's available verifier tier is a first-class, measured property,
and a new headline metric sits beside improvement-rate-per-tier:

> **inert traffic share** — fraction of traffic in domains where the highest
> available verifier tier is below the floor the improvement machinery requires.

Then Part III §0's claim that verification coverage is the binding constraint stops
being an argument and becomes a number you watch. And §9's verifier-synthesis bet
gets an objective function: reduce that number.

---

## 3 · One line · the blast-radius rule must bound threshold *values*

E4.2's second leak. Part I currently reads:

> *L9 may never edit: L1's schema, the Assay, the promotion gate's tier
> requirements, or this rule.*

That enumerates **artifacts**. E4.2 showed the reachable set is unbounded anyway,
because `noise_leeway`, `eps_reg` and the shrinkage on `w` are values, not artifacts,
and relaxing all three together moved compliance 0.005 → 0.322.

**The fix:** every threshold L9 may edit carries a permitted interval, and *the
intervals are in the non-editable set.*

Costs nothing — it removes no threshold from L9's search space, only the ability to
leave the space. It is R4's first component, and it belongs in the rule rather than
in a repair list.

---

## 4 · One line · the promotion gate must specify its tie-break

E3.1c's Panel C is a rig bug, and it is also a warning about the design. Ties broken
toward the pool's first emission gave comp-only 0.550; toward patches, 0.000, and
the margin inverted. `if score > best_score` was a policy nobody wrote.

Part II §E's gate ranks by net transfer and says nothing about ties. In a real
system ties will be common — early, and whenever the transfer signal saturates,
which E3.1b showed happens at excess spillover ≥ 0.03. **An unspecified tie-break in
a promotion gate is a policy, and it will silently be whatever the implementation's
iteration order is.**

**The fix:** on ties, prefer in order — higher verifier tier, broader provenance,
lower rank cost, older candidate. All four are already recorded, and all four point
away from the failure mode the gate exists to prevent.

---

## 5 · One line · split L7's timescale into disable and recompile

Part I §7's substrate table gives L7 adapters a single timescale, "hours–days."
E0.2c established that this hides the entire deletion story: disabling an affected
adapter is ~28,800× cheaper than recompiling it, **and disabling is what makes
deletion sound** — recompilation restores competence. E0.2b reported a
service-quality limit as a correctness limit precisely because the design offers one
number where there are two operations.

**The fix:** L7 has two operations. `disable` — milliseconds, restores *soundness*.
`recompile` — hours–days, restores *competence*. State both, and state that I4's
correctness half depends only on the cheap one.

---

## 6 · Statement · §B must state its enabling condition

Part II §B asserts that ranking by net transfer accumulates abstractions. It states
no condition under which this holds. E3.1b and E3.1c bracket one:

> Net-transfer ranking discriminates only while narrow candidates do not produce
> systematically *more* off-target movement than general ones. In E3.1's units the
> margin degrades to zero at roughly `0.25 × GAIN_SKILL` of excess. Carry it as a
> ratio, not an absolute.

Note this is weaker and more plausible than the absolute condition ("patches have
near-zero real spillover") — a skill adapter is fit to a broader distribution and
should if anything spill more. But an unstated condition is not a satisfied one, and
this is the design's central generalization claim.

---

## 7 · Statement · distinguish the transfer *statistic* from the transfer *matrix*

E3.1c: no arm in E3.1 ever consults accumulated history. `transfer` scores
`d.sum() − d.max()` on the current candidate's measured deltas. The **statistic** now
has evidence — it beats both target gain and a breadth penalty. The **matrix** has
none, in either direction.

Which matters because they have entirely different justifications, and everything
expensive hangs off the matrix: τ storage at observation grain, the partition
objective, the censoring bias, and the L3 compiled signature-ontology view.

**The fix:** separate them in the text, and mark the matrix's three remaining uses
— L8 curriculum prior, L7 merge prior, diagonal-`T` as a memorisation diagnostic —
as **unevidenced**, since none of them was ever the stated reason for building it.

---

## 8 · Statement · name the counterfactual-retrieval influence path

E0.2b's irreducible 8.7%: rollouts retrieve their conditioning card by query
similarity, so deleting an entry can hand a rollout to a *different* card. Provenance
records the card that was selected; the card that would have been selected was never
run and is recorded nowhere.

Part I §7's promotion path shows `Event → L1 → L3 card → L7 candidate`. It does not
show that **retrieval selects the card, and that selection is itself
entry-dependent.** An influence path with no arrow in the diagram is one nobody
reasons about, and this one defeats every set-based closure.

**The fix:** draw the edge, and annotate it as the path provenance cannot cover.

---

## 9 · Statement · L3 admission carries two thresholds serving opposed objectives

E0.2d: content cosine ≤ 0.93 is a redundancy filter on card *content* and is not a
lever on cascade breadth; breadth is set by provenance overlap. So admission needs a
second threshold — a cap on source-entry overlap between admitted cards.

And from the I4 amendment: capping `|P|` for recompile cost requires distilling
provenance, which **increases** provenance overlap and therefore breadth.

**The fix:** state that L3 admission has two thresholds pulling in opposite
directions, that no jointly acceptable setting is known, and that finding one is
open. This is the most likely place for a second genuine architectural finding, and
E0.2c already has the recompile-cost machinery to look.

---

## 10 · Scoping · the three readings need different weightings of the estimator

E1.1c: a traffic-weighted energy cut allocates protection in proportion to
frequency, Spearman(rate, exposure) ≈ −0.95. The gate *should* be traffic-weighted —
trust weights where traffic is dense. The budget *must not* be, or rare domains are
classified free.

So Part II §A's headline — "epistemic uncertainty and plasticity headroom are the
same number" — is too strong. It is one estimator, three readings, and **two
weightings**: traffic-weighted for the gate, frequency-balanced or worst-domain for
the budget.

**The fix:** state the weighting per reading. The unification survives and is still
worth having; the sentence claiming they are one number does not.

Minor, same section: `w`'s shrinkage in the gate must be derived from τ counts alone.
Part I §6's firewall governs the router; the gate now has a transfer term, and if
`w` can be set by anything textual the firewall is bypassed at the gate rather than
at the router.

---

## What to cut

The design has grown faster than its evidence. Two items should come out until
something supports them.

**Cut Part III §4(b) — transfer measurement as a router arm.** It spends promotion
budget to fill `T` cells so the cold start resolves faster. But E3.1c shows the
matrix is unevidenced. Spending real budget to populate a structure whose value has
never been measured is backwards. Keep §4(a) — writing a τ row for every off-target
Δ in every evaluation — because that is nearly free and useful regardless. Defer (b)
until the weighted-vs-unweighted arm runs.

**Defer Part III §7 — typed arm tuples.** E4.5 predicts under 10% of the
`artifact × edit_type` product is semantically valid, in which case the mechanism is
a rejection engine rather than a search-space expansion. That prediction is cheap to
check by enumeration and needs no simulator. Check before building.

---

## What I would leave alone

**The ledger-first thesis.** Nothing in the record threatens it. Every finding so far
constrains a *rate* or corrects a *formula*; none has attacked the claim that one
authoritative append-only log with derived views is the right shape.

**I3's no-compounding rule.** It is carrying more weight than it was designed for —
it contains synthetic experience for free, which is why L8 needs no new containment
— and it has not been tested. Leave it as is and test it (E0.3) rather than
strengthening it pre-emptively.

**The sealed Assay.** E4.2 found the *boundary* is drawn wrong, not that sealing is
wrong. Fix the boundary (item 3, plus R4's pin and seal), keep the artifact.
