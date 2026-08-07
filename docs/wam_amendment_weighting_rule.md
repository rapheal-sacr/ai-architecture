# Amendment · the weighting rule

Generalises item 10 of `wam_amendment_before_continuing.md`, which is drawn too
narrowly. That item says Part II §A's "epistemic uncertainty and plasticity
headroom are the same number" is too strong, and that the estimator needs **two
weightings** — traffic-weighted for the gate, frequency-balanced for the budget.

That is right and it is a special case. The general statement:

> **Any statistic used for PROTECTION or ALLOCATION must be computed per region
> on equal-sized samples. Only statistics used for ROUTING may be
> traffic-weighted.**

The distinction is what the number is *for*. Routing asks "where should this
query go", and being wrong in proportion to frequency is exactly right — a rare
region routed slightly worse costs little. Protection asks "what will this
damage", and being wrong in proportion to frequency is exactly backwards, because
the damage to a rare region is not smaller for being rare. Allocation is
protection wearing a different verb.

Four measured symptoms of the same anisotropy, all in L7:

| # | Symptom | Where |
|---|---|---|
| 1 | the energy cut classifies rare-domain subspace as **free** | E1.1c |
| 2 | the allocator's free pool is rare-domain-dominated, so allocation concentrates there under every ordering | E1.2 |
| 3 | the non-regression gate cannot see regressions there — worst-domain damage 0.858 against a traffic-weighted 0.166 | E1.1c, E1.2 U2 |
| 4 | the gate's `qᵀR⁻¹q` reads practised-but-unprobed directions as known | analytic, untested |

Every protective mechanism in L7 fails on the same axis, and that axis is the
long-tail personal knowledge the ledger exists to preserve. That is not four
findings. It is one, four times.

---

## The enumeration

Every place a per-region quantity is averaged over traffic. Marked **P** where
the statistic protects or allocates and must therefore be per-region, **R** where
it routes and may stay traffic-weighted.

| Site | Where | Weighted by | Class | Status |
|---|---|---|---|---|
| subspace budget — `rank_for_energy` sums energy over held-out queries | Part II §A / L7 | traffic | **P** | measured, E1.1c |
| free-pool definition for allocation | Part III §2 / L7 | traffic (via the estimator) | **P** | measured, E1.2 |
| non-regression probe set — "held-out probes drawn from other regions" | L7 gate rule | traffic, if probes are sampled from it | **P** | measured, E1.1c |
| gate `g` from posterior variance along the query direction | Part II §A / L4 | traffic | **R** | correct as-is |
| gap set — "integrate posterior variance per domain signature **over turns**" | Part I §3 / L8 | traffic, structurally: a rare region accumulates fewer turns | **P** | inferred |
| transfer matrix `T` — off-target Δ on probe regions | Part II §B | traffic, if probe regions are traffic-sampled | **P** | inferred |
| `w` shrinkage on `T`'s columns, from τ counts | Part II §B | traffic, since τ rows accrue with traffic | **P** | inferred |
| probe harvesting — sample resolved interactions | Part III §5 / Assay | traffic, by construction | **P** | measured, E2.1 |
| Consolidator "decay unused entries" | L0–L6 card | frequency of reference | **P** | inferred |
| L2 LRU — "hot and recent history stays encoded, cold history exists only as text" | L2 card | recency + frequency | **P** | inferred |
| sealed-suite drift | Part I §10 | traffic, via the harvested suite | **P** | inferred |
| I4's `S(A)` pass/fail counts | E0.1's own instrument | pooled over items, hence traffic | **P** | inferred, and the most dangerous |
| A5's recompile cost over `|P|` | E0.1 | mean over adapters, `|P|` heavy-tailed | **P** | inferred |
| router arm value / cost-per-promotion | Part III §10 | traffic | **R** | correct as-is |
| improvement-rate-per-tier | Part I §10 | traffic | **R** | correct as-is |

**Thirteen protection sites, three of them measured and ten inferred from the
design text.** More than the four item 10 anticipated, and the pattern is that
*every* statistic in the stack defaults to traffic-weighted because traffic is
what the system sees. Frequency weighting is not a choice anyone made; it is what
you get by not making one.

---

## What the amendment requires

**1 · Classify every statistic P or R at the point it is defined.** Not in a
review, not in a repair list — in the sentence that introduces it. A statistic
with no class marking should be treated as unspecified rather than as R.

**2 · P-class statistics are computed on equal-sized per-region samples.** The
cost is domain labels at calibration time and equal-sized probe sets per region.
E1.1c establishes this is affordable: labels are needed only where the statistic
is computed, not on every query.

**3 · P-class statistics report the worst region, not the mean.** A mean over
regions is better than a mean over traffic and still hides the tail. The
reportable number is `max_r`, with the traffic-weighted value beside it and the
ratio between them on the dashboard as a **blindness factor**. E1.2 measured
5.2–7.2×; anything above ~2 means the instrument cannot see what it is for.

**4 · ρ, ε and every other protection threshold become measured, not chosen.**
Set from worst-region interference rather than from a retention target. This is
R1 and R7's move — measure what happened rather than trust the bookkeeping rule —
and it is now the third place it applies.

---

## What this does not fix

**The tail is not free.** Per-region equal-sized sampling costs probe budget
linear in region count, and Root 2 already says evaluation is the binding
constraint. So this amendment converts an invisible correctness problem into a
visible cost problem, which is progress and not a solution. The joint-feasibility
question — whether tail-safe protection, bounded recompile, acceptable cascade
breadth and deletion latency have a non-empty intersection — gets *harder* under
this amendment, because per-region protection is what makes tail-safe ρ expensive
in the first place.

**Region count is now a variable in the feasibility problem, not a constant.**
Per-region sampling costs probe budget linear in region count; region count is
set by the partition; the partition objective is E3.3. So Root 3 appears inside
Root 1's feasibility problem — a fifth edge in the coupled system, and it means
joint feasibility must take region count as an axis alongside subscription ratio
and overlap (E1.1d). Region count enters twice and in opposite directions: more
regions means finer protection, and more regions means more probe budget.

**Region granularity is now load-bearing twice.** Signature ontology already
determined `T`'s index set (Part III §3, E3.3). It now also determines what
"per region" means for every P-class statistic. A partition too coarse hides the
tail inside a region; too fine and per-region samples are too small to estimate.
Root 3 reaches further than Part III claims.

**`qᵀR⁻¹q` is not addressed by re-weighting.** Symptom 4 is anisotropy *within* a
region — practised in some directions, unprobed in others — and no per-region
weighting sees it, because the region is being sampled as a unit. That needs
coverage-based rather than count-based uncertainty, and it is E1.3's territory.
