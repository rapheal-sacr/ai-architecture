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
*and* I11 is expressed as the rate it already is.

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

**Four numbers decide the rest** (a fifth, `L`, was dissolved by measuring
`H`), **and they are two different kinds** — a
distinction worth making explicit, because one kind is cheap and the other is
not.

## WAM-RX implementation status

**Milestones 1 and 2 pass their registered synthetic gates.** E0.10 adds the actual
single-process authority layer: immutable version-1 events in SQLite,
deterministic replay, atomic batches, corrections and tombstones, enforced
artifact lineage, item-level deletion disabling, a hybrid retrieval baseline,
complete selection journals, and regional compile-adequacy measurement.

The negative control matters: pooled coverage is **0.923** while rare-region
coverage is **0.0**, and the regional gate rejects the compiler. The compliant
arm reaches 1.0 regional coverage, 0.0 distortion, immediate tombstone disable,
and exact clean-rebuild equivalence.

The Milestone 2 foundation then fixes typed witness closure, observed-root-only
promotion, ledger-sequence transaction order with canonical UTC timestamps, and
fail-closed runtime compatibility. E0.3 and E0.4 pass at their registered
structural scope. E0.11 adds provenance-linked temporal analytics, immutable
analytic query journals, and an explicit belief/constraint graph. It answers all
four registered structural tasks, preserves contradiction history, blocks
missing/conflicting constraints, disables tombstoned support, and matches clean
rebuilds. All seven requested negative-control classes fire as expected.

These are small single-process representation results, not evidence for
recurrence, expert routing, continual weight updates, self-improvement, or AGI.

---

## Invariants

| Invariant | Status | Rests on | What would change it |
|---|---|---|---|
| **I1** grounding | **verified at registered structural scope** | E0.4: both promoted capabilities resolve to observed roots; coverage 1.0 | broader and adversarial grounding populations |
| **I2** derivability | **broken → repair** | stamp was a pair, not a four-tuple | amendment rev 2 §B: component-granular stamp. E0.1 A3 **rebuilt** (B20): a draw policy that differs between compile and recompile moves competence **8.1 points above the resampling floor**, ~4.5σ. The old 21.6% varied nothing and is withdrawn |
| **I3** no-compounding | **verified at registered structural scope** | E0.3: synthetic/inferred-only promotion rejected atomically; grounded multi-hop accepted | broader gap-derived and externally mediated chains |
| **I4** recompilability | **broken → repair** | was a bookkeeping claim wearing a competence claim's name | E0.1: 6.7% pooled over-forgetting hiding **79.8% worst-region**, 12.4× blind. Amended to a two-sided verified property with three support categories |
| **I5** anchored improvement | **broken → repair** | blast-radius rule bounds *which* thresholds, not their *values* | E4.2. R4: bound the values; pin derived views; seal only harness code a verifier executes through |
| **I6** refutation permanence | assumed | untested | E4.3/E4.4, unrun |
| **I11** compile adequacy | **new — proposed** | did not exist | E0.1 A1b: I4 checks recompile fidelity and says nothing about compile adequacy. A system that bounds cost by never learning the tail satisfies I2 and I4 completely |

**Renumbered, and the collision is the point.** This invariant was **I7** in every
prior version of this record. Rev 2 of the design independently adds a block
I7–I10, so the number carried two different invariants — compile adequacy here,
recorded commitment there. That is the *fourth* instance of one name covering two
quantities, after L7's "hours–days", I11's own scalar-vs-rate shape, and `fleet`.
The other three were found by a measurement going the wrong way; this one was
found by two documents being read side by side, which is a weaker detector.
Renumbering this one rather than the design's block, because a falsification
record is downstream of the thing it falsifies.

**Attribution, since it belongs with the finding.** The collision was minted on
the design side — I7–I10 were added without checking whether the record already
had an I7 — which is the same failure to check the artifact that produced B19 on
this side. Two instances, opposite directions, one week apart. And the
renumbering turned out to be substantive rather than clerical: **I8 and I11 are
complements.** I8 is a within-owner property and I11 is compile adequacy, and
E0.6's unowned-provenance fraction is the instrument for I11 that E0.1 said did
not exist — you cannot protect what was never compiled, but the count is free.

### Proposed in rev 2, not yet in this record

None of these has been tested here. They are listed so that a reader can tell
which invariants the record has an opinion about and which it does not.

| Invariant | Status | Would retire | What would test it |
|---|---|---|---|
| **I7** recorded commitment | proposed — untested | ε thresholds, ρ retention, three-λ unanimity, release-as-inference | worklist 3.1, the hinge: register vs spectrum on E1.1c's per-domain bar and E1.2's blindness ratio |
| **I8** equal audit | proposed — untested | the weighting-rule amendment's 13-site enumeration, and pooled protection numbers | subsumes the amendment as a design property rather than a rule applied at enumerated sites. E4.2 is the standing evidence that enumerations are the shape that fails |
| **I9** instrument separation | proposed — untested | the artifact-list form of the blast-radius rule | worklist 5.3: rank-correlate reference reader against production reader before adopting |
| **I10** oracle conservation | proposed — untested | the assumption that the Assay is hand-maintained at the rate the system improves | worklist 5.4 (E2.2 by hand) decides whether L10 is real at all |

---

## Rev 2 mechanisms · what the recording layer now measures

Phases 0–2 of the rev 2 worklist change no behaviour. These are the two Phase 1
recordings, and both returned something the design document does not say.

| Mechanism | Status | Rests on | What would change it |
|---|---|---|---|
| **the card compiler** | **settled by I1, not measured** | L1 is the single authoritative store | Two capabilities on the same fact must both cite it — no second copy exists. **Sharing is architectural.** Degree ≡ 1 is the signature of a relabelling, so `doc = card` was never a compilation |
| **§3's fan-out kill** | **evaluable, and about structure** | needed a demand-driven bank | E0.9: demand gives max degree 5 / 49, and real exceeds the rewired null by **2.12× / 3.34×** in both ledgers. Schematic sits *below* its null (1.00 vs 4.68) — anti-concentrated by construction |
| **§2's certified fraction** | **bank-dependent, sign unstable** | assumed a property of the design | E0.9: SciFact **0.691 → 0.442** (falls), NFCorpus **0.619 → 0.783** (rises) under the same compiler change. Cannot be quoted without naming compiler *and* ledger |
| **asset B1** | **built — unblocks M2, not M3** | worklist claimed M2 + M3 | R16: BEIR SciFact+NFCorpus, 800 selections, **12/15 predicates pass**. But **entry degree ≡ 1** — doc=card gives disjoint provenance, so §3's fan-out kill is unevaluable. E0.2e's shape. M3 needs a card-compilation step where cards share entries |
| **asset B2** | **resolved — 2WikiMultihopQA** | was "the genuinely scarce half" | [Candidates checked against the predicates](docs/asset_b2_candidates.md). **HotpotQA rejected**: its `level` is derived from baseline model answerability — P1b by construction. 2Wiki: structural hop count from `evidences`, template-generated `type`, Wikidata `entity_ids`, Apache-2.0. Gold answers make P0/P0a hold by the shape of the task |
| **P1d as written** | **wrong — corrected** | "capture-time" | Applying it to Wikidata types showed capture-time was a **proxy**. The requirement is **independence from the system being measured**: a partition is inferred when the system derives it from its own traffic, recorded when it comes from an independent authority |
| **the acquisitions** | **B specified, C spent, A specified by EB.3** | was "one corpus" | R13: asset B's spec is **executable predicates over the asset** ([validator](tools/validate_asset_b.py), self-tested — all six fire on an asset built to violate them). P1b and P3a are the two that cannot be retrofitted, because both need a decision at *capture* time |
| **`r` and whether `A` decays** | **measurable now; decay is partial** | was listed corpus-blocked | EB.6: needs a generator + predicates, not a corpus. `r_generic` **0.78 → 0.00** across 0.5B→1.5B; `r_specific\|reached` **1.00 → 0.72**. `fan_out` fires on **68%** of 1.5B candidates — a cap stated in the prompt. `r_specific` is also a **design variable** |
| **`A` as the binding quantity** | **withdrawn — it never was** | assumed screens amplify the oracle budget | The accounting: promotions = `v × sound` either way, so labels/promotion is `1/v` with screens and `1/(v(1−r))` without. Screens recover exactly what unsoundness cost, **nothing more**. The constraint was always **`v = P(valuable\|sound)`**, and the one number that decides §4 is its correlation with soundness |
| **L10 amplification `A`** (§4.2) | **ceiling computable — "categorical" withdrawn** | assumed synthesis reaches the value judgment | E2.2 by hand: checks decompose **soundness only**, so `A = 1/(1−r)` with `r` the sound-screen rejection rate, itself bounded by the *unsound* fraction. **`A` degrades as the generator improves.** Measure `r` — needs no labels and no L10 |
| **§4 vs §6 on what an instrument is** | **unadjudicated** | never connected | E2.2: screens give bounded self-limiting `A`; estimates give E2.3's recall problem (0.349) one level up. A synthesised instrument *is* a cheap rung under another name |
| **R-a…R-d under churn** | **hold; bill is mild** | §1.2b priced nothing | E1.7: 0 monotonicity violations in 600 cycles; growth **linear not accelerating** (4.17→3.58 evals/cycle); residual **52.5%** of defended, so R-c is the larger half as E0.6 said |
| **sub-threshold owners** (R12) | **closed — third answer** | R12 refused the request and didn't say what replaces it | It does **not become an owner** — stays an L3 card, retrieval-served. No similarity judgment (neighbourhood clause stays dead), no pooling (keeps its independent statistic), and `\|provenance\| ≥ threshold` is a **sound screen**. Cost lands on the tail; measurable as compiled-vs-retrieval on the same probes |
| **within-owner coverage** (1.4) | **now reported** | I8 checks count, not coverage | `ProbeRegistry.coverage()` reports worst/best/spread per owner. 40× spread on Zipfian provenance, matching E0.8's 38× |
| **§8's abstraction test** | **confounded — format control needed** | assumed a third region tests abstraction | EB.3 generalised: if all three regions share a form, the merge wins by learning **form**, not by abstracting. The third region must differ in **form as well as topic** |
| **§0's thesis as a prediction** | **confirmed, weakly** | never tested against new records | R14: 2 clean deletions (merge prior; deletion risk, which was **off-list**), 2 partial. Yield has changed shape — the originals were all immediate and total. And it produced a criterion §0 lacked: **a record substitutes only if it is available when the decision is made** |
| **EB.5's repairs** | **built; one of two is the repair** | centering assumed to matter | EB.7: discrimination of the top-5 goes 0.0380 → **0.1579** under between-owner ranking (1.5B), 4.2×. **Centering moves it barely.** Costs energy: 67.9% vs 89.8% captured at r=5 |
| **§1.3 vs the rank ceiling** | **contradiction — R15 proposed** | granularity and discrimination assembled separately | 64 owners at rank 8 needs **512** directions; `rank(B) ≤ n_owners − 1` supplies **63**. R15: rank by between-owner variance to a **measured** crossover, then energy. Measured at ~2(G−1), not G−1 — switching at G−1 gives up directions still worth having |
| **what the subspace budget is scarce in** | **fourth reading** | assumed `dim` | R15: under between-owner ranking the denominator is the number of **distinguishable capability regions the ledger supports** — a ledger property, not a model one. After ε, ρ, and ownership set-arithmetic |
| **rank ceiling from owner count** | **new bound — unstated** | `rank(B) ≤ n_owners − 1` | EB.7: the gain inverts to **0.5× at r=32** with 8 groups, because past rank 7 the between-group eigenvectors are in B's null space. A 64-owner fleet commits at most 63 directions this way. Third data-shaped bound, and the only one that **loosens** as the fleet grows |
| **`R_t` is uncentered** | **broken → two repairs** | `R = λR + outer(a,a)` | EB.5: centering buys 22.2%→6.0% at 0.5B and **78.3%→77.4% at 1.5B** — right about the small model, wrong about the large one. Centering is one line and is **not sufficient** |
| **energy criterion ranks total energy** | **broken → rank between-owner** | assumed energy tracks capability | EB.5: the 1.5B's top-5 dims hold **77.4% of variance and 0.2% between-domain share** vs 2.0% for all others. Ranking by between-owner variance is **free under the register** and impossible for the spectrum — an argument for §1 that §1 doesn't make |
| **rank allocation has no data bound** | **R12 proposed** | §1.2 never says how large a request may be | EB.4 + R12: `rank_o ≤ f(\|provenance_o\|)`. At 40–250 entries **no rank is stable** at either dimension. Converts a silent noise-fitted basis into a refusal |
| **per-layer `R_t`** (Part II §A) | **broken — first-order cost** | "cheap but not free" | EB.4: **55–61% of model weight memory**; a full budget read is **1.7 s / 8.4 s** because the eigendecomposition is per layer. float32 halves the memory and not the conclusion |
| **I8's minimum equal-N** | **new constraint — unstated** | I8 never says how large | EB.4: the spectrum's split-half stability floor is **≈0.65 × dim** (~586 at 896, ~1026 at 1536). Below it a per-owner spectrum is sampling noise. **I8's N is estimator-limited and grows with the model** — ~66k vectors for a 64-owner fleet at dim 1536 |
| **§B's condition** (R10) | **blocked — instrument, not adapters** | seven sentences per domain, one shared form | EB.3: manipulation check fired (argmax on target 0/3, then 1/3). **78% of raw spillover was format transfer** — adjusted off-target −0.03 against on-target +0.12, directionally consistent with §B but not readable. Needs the corpus |
| **the hinge on REAL geometry** | **survives — advantage smaller** | Qwen2.5 activations, not DomainMixture | EB.2: KB3 holds at all 12 points. But register advantage falls **2.17× synthetic → 1.37× (0.5B) → 1.09× (1.5B)** — real domains overlap, so the spectrum is far less blind than the generator implied, and the advantage **shrinks with model size** |
| **DomainMixture's geometry** | **broken — both assumptions** | never measured | EB.2: decay 0.237 (0.5B, flatter) vs 1.070 (1.5B, steeper) against an assumed 0.50; cross-domain overlap **~0.30** against an assumed ~0. E1.1c's panel C concern confirmed, not swept |
| **GPM energy criterion** (E1.1b's repair) | **new defect — unaddressed** | assumed energy tracks capability | EB.2: on the 1.5B **71% of variance sits in 5 input-independent dimensions** (massive activations, max \|a\| 227). The criterion commits its budget to directions that distinguish nothing. No synthetic generator would have shown this |
| **the hinge, ratio half** (§1) | **PASS — §1 survives** | traffic weighting, not subscription | E1.6: blindness **2.88 → 1.33**, Spearman(rate, leakage) **−0.971 → −0.056**, exposure ordering **+0.045** so the change is not cosmetic. At ρ=0.99 the spectrum leaves 12.8 domains failing and the register 0.0. Free rank printed and **not scored** — that half is unreadable at subscription 1.0 (E1.1d) |
| **register vs freq-balanced R_t** | **same protection, no estimate** | R-b needs the rate vector | E1.6: R-b 1.45 against register 1.33. R-b must estimate the traffic distribution the weighting rule says protection must not depend on; the register needs only provenance I1/I4 already require |
| **within-owner coverage** (I8) | **open — 37.6×** | equal *count* ≠ equal *coverage* | E0.8: coverage falls 55× with depth; between-owner spread 1.7× under uniform rollouts but **37.6× under Zipfian experience**. I8 is satisfied and the tail inside an owner is unprotected. Third level after between-region and between-owner |
| **selection journal** (§2) | **sound, narrow** | scores the system already computes | E0.5: no certified selection ever flipped, anywhere in the sweep — but certification collapses to ~0.00 at every batch ≥ 4 while the world stays 0.60–0.95 stable. Worklist 2.2 on real traffic decides whether it pays |
| **"uncertified ⊂ closure edges"** (§2) | **broken — better than claimed** | assumed influence runs through the chosen card | E0.5: 24,598 uncertified selections outside the closure, **2,178 of which really flipped** — a rival rose while the chosen card was untouched. The journal enlarges the cascade and part of the enlargement is influence no closure can see |
| **R-c total residual set** (§1.2b) | **load-bearing, and understated** | a cover is not a partition | E0.6: never-owned exceeds retirement-orphaned **3.3–5.5×** at every traffic level. R-c is not a refinement of R-b, it is most of the job |
| **the cover's reach** | **traffic-shaped** | I8 is a within-owner property | E0.6: unowned provenance 0.483 at 16 rollouts → 0.027 at 512. Equal draws inside an owner say nothing about how far the owners collectively reach |
| **§1.3 granularity pricing** | **holds only below the crossover** | a saving statistic that was measuring its denominator | E0.7: the 77% is withdrawn — pool backs out at ~480 against 2048 draws, so 98.5% was consumed and the count was pinned. On `distinct/\|pool\|`: at pool 4102, **97.1% of draws are distinct** — every owner pays. A real ledger sits in that regime. What rescues the verdict is E2.1's harvest yield (0.68→0.33, ~2×), not overlap |
| **R-d probe dedup** | **works, on a stated and falsifiable precondition** | a probe is a stimulus + an expectation | E0.7: the stimulus is entry-keyed and shares (oracle line); the expectation is bookkeeping and free (compute line). Condition: **the expectation must be derivable from the stimulus** — false for owner-specific target behaviour, where distinct probes equal draws at every pool size. The cost of restricting to ground-truth-keyed stimuli is **coverage**, unmeasured |
| **inert-arm detection** | **mechanical, was inspection** | B7, B8, B20, A4-usage are one defect | [rig_a/core/trace.py](rig_a/core/trace.py): an arm is inert when nothing it mutated is read on the path to the measurement **and** both calls took the same arguments. E0.1 asserts it over every arm before any metric is read |
| **`unowned fraction` as reported** | **wrong grain** | pooled over the ledger | E0.6: worst card 1.000 against pooled 0.483, blindness peaking at 4.2. The same defect as E0.1 KB, E1.1c and E2.3 — and the first time it appears inside a statistic proposed to fix it |

**That tension was mine, and E0.5's window analysis dissolves it.** §2 and R8 act
on different operations — disabling is per tombstone (load **0.138**, and the only
place a certificate is needed); recompiling is what gets batched. Running
certification is sound and at window close is *algebraically the same set* as the
batch reading, so it changes when the answer is needed, not the number. The
narrower true claim: 92.9% of selections are uncertified over a window while
11.4% flipped, so the certificate is nearly useless for pruning the recompile
queue and earns its place on the disable path only.

**And the two currencies are not one curve.** E0.8 holds ledger and fleet fixed so
depth alone moves: unowned provenance collapses 0.692 → 0.000 while distinct
probes stay **flat at ~225**. Depth governs coverage; the oracle line is governed
by `fleet × probes` against the pool and never moved. Pool consumption falls with
depth only because its denominator grows — the same defect as the withdrawn 77%,
caught because the absolute count was carried beside the ratio.

---

## L1–L3 · ledger, index, compiled views

| Claim | Status | Rests on | What would change it |
|---|---|---|---|
| tombstone cascade reaches the weights | **broken → repair** | provenance recorded at compile time | E0.2b: `transitive` recalls **0.913, not 1.0**; no set-based closure reaches 1.0, because the residual dependency is a retrieval never run. R7: verify, don't infer |
| L3 admission (cos ≤ 0.93) controls cascade breadth | **broken → no repair yet** | conflated content with provenance | E0.2d: content cosine moves 0.498, breadth moves 0.009. The design has **no lever on breadth**. R9 proposed — see *open* below |
| signature ontology is not load-bearing for competence | **restored — K4 withdrawn** | K4 was scored on `identical`, False for any stochastic arm | E0.1 A4 differenced against a same-policy null (worklist 2.4): effect **−0.0118 pooled** against a null seed spread of 0.0218 — indistinguishable from resampling. The 21.7%/41.6% was the stratified draw's floor. Root 3 does **not** reach into Root 1 by this path, and rev 2 §1.2 may keep the ontology for routing |
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
| **the real bank : fleet : rollouts-per-adapter ratio** | whether there is a breadth problem *at all* | **open, and it replaced `β`.** E0.2f: over a 400-card / 64-adapter bank the functional instrument reads **0.087** against E5.1's 0.31 target. E0.2d's 0.63–0.98 came from 8 adapters, 8 cards, 60 entries — a world where each draw covers most of the bank by construction. Breadth is a function of that ratio, and neither world is calibrated to reality |

**DECIDE** — these need a sentence, not data, and they are the cheapest thing
remaining:

| Quantity | Gates | Status |
|---|---|---|
| ~~**restoration latency tolerance `L`**~~ | C4 | **largely dissolved by EB.1.** At the measured `H` the drain term is 0.21 days, not 5.33, so the window is open even at a **one-day** tolerance — `L` has ~13 days of slack before it binds. It was a gating quantity only because `H` was assumed at 8h. Still worth writing down, but it no longer decides anything |
| **I11 per-region density floor** | R9's coverage side | a policy choice. E0.2e sweeps it: at a floor of **1** there is no tension at any τ; at **5** the tension is total even at the loosest. "100% of regions below floor" is a statement about the 3.0 I chose |

**Two sentences and one coupling make R9 evaluable.** That is the whole distance
between the last open gap and a drawable curve, and none of it needs an
experiment.

## The one open gap — and it may not be a gap

**`R9` is suspended, not blocked.** E0.2f reconciled the two breadth numbers by
running both instruments over the same admitted bank, and it dissolved the
premise rather than the repair:

```
   tau   bank   synthetic draw   influence graph   ratio
  0.00     16            0.375             0.082   0.22x
  0.35    110            0.118             0.080   0.68x
  1.00    400            0.088             0.087   0.99x
```

**The two agree at the uncapped bank** (0.088 vs 0.087) and diverge only as the
bank shrinks — exactly where the synthetic draw's fixed-count-from-a-shrinking-
bank assumption bites. The influence graph spans **0.007** across τ where the
synthetic draw spans 0.287. So E0.2e's opposition — the thing that made R9
unevaluable and appeared to point it backwards — **was the instrument.**

**And neither reproduces E0.2d.** Both read ~0.087 against E5.1's 0.31 target.
E0.2d's 0.63–0.98 came from a world with 8 adapters, 8 cards and 60 entries,
where each adapter's 6-rollout draw covers most of the bank by construction.
E5.1 fitted `cascade_breadth = 0.60 + 0.45·po` to that point and made the result
decide whether an operating envelope exists.

So **breadth may not be a problem**, R9 may have no job, and the quantity that
decides it is the real **bank : fleet : rollouts-per-adapter** ratio — a measure
quantity for deployment, not something either simulation settles.

### Three corrections still owed

Raised in review, verified, not yet applied:

1. **Applied.** Two fleets, one number — E5.1's C1 asks `free >= RANK_REQUEST` — whether
   the budget fits *one* rank-8 adapter — while C3/C4 use `fleet = 64`. Same
   symbol, 64× apart, and 64 basis-disjoint rank-8 adapters need 512 dimensions
   in a 128-dimensional space. **Active fleet** (bounded by `free_rank /
   RANK_REQUEST`, so ≈2 at E1.1c's tail-safe rank) and **stored fleet** (every
   promoted adapter, grown by promotions) are different quantities. C1 governs
   the first; C3, C4 and every breadth measurement govern the second.
2. **Applied — and the ranking inverted.** E5.1 re-run with the fleet split,
   `β` swept rather than fitted, and the concurrent:promoted ratio on an axis:

   ```
     before                        after
     C4 latency   58.3% / 13716    C1 free rank  66.0% / 50820   <- MOST
     C6 forget    57.3% /  8262    C4 latency    51.7% / 48390
     C1 free rank 11.1% /   108    C6 forget     43.3% / 27252
                  ^ LEAST          C3 throughput 11.7% /  8388   <- least
   ```

   **C1 goes from 108 configurations eliminated alone to 50,820** — 470×, purely
   from stating the constraint against the right population. The subspace work
   was aimed at the *most* binding constraint, not the least. And every sampled
   feasible configuration sits at **β = 0.087**, E0.2f's reconciled value; none
   at E0.2d's 0.65.
3. **Still owed.** I11's floor is in the wrong units — It counts *cards* per region; I11 claims
   *ledger* coverage. Five cards drawing the same eight hot entries are worse
   covered than two drawing sixteen distinct ones — and `concentration` is the
   knob in that very experiment controlling exactly that. Correct form:
   *distinct supporting entries reachable per region, as a fraction of the
   region's entries, worst region* — which makes "derive from measured `k`" a
   real derivation rather than a deferral, and inherits a Root 2 verifier
   requirement, since `k` needs a corpus.

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
going the wrong way?"** — **seven** found that way (B7, B9, B11, B12, B13, B16,
B18) against **zero** from cross-checks between two measurements. The
manipulation check has caught three. That ratio is the most transferable thing
this programme produced.

**And a claim can be wrong twice for unrelated reasons.** The C1 attribution
failed first to B11's grid confound, was corrected, re-registered as curves and
returned to the front page — then failed again to the fleet-of-one. *Surviving
one correction does not confer credibility, because a correction only tests the
support it touched.* A claim resting on two supports needs both audited before
it goes back up.

Read the retraction record (`claims/claims.yaml`, `bugs:`) alongside the results
— not because the rate is alarming, but because the *reversals* are the most
informative entries in the file.

The strongest single regularity: **every mechanism specified without stating its
tension has been found out by an experiment.** That now includes R9, which was
proposed as a repair and arrives with the same defect.
