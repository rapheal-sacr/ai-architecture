# WAM Falsification Plan

**Goal: find where the Write-Ahead Memory architecture breaks, cheaply, before building it.**

The design (L0–L9 + sealed Assay + bandit router, six invariants) is unusually
self-critical already — Part III is a response to its own review, and §9 of
Part I lists seven places it expects to hurt. This plan does not re-litigate
that list. It converts the architecture's *asserted* mechanisms into
experiments with pre-registered kill criteria, runs the cheap decisive ones
first, and treats "this mechanism is not well posed" as a finding worth having
in week one rather than month nine.

---

## 0 · The thesis that organises this plan

**Most of WAM's load-bearing claims are not claims about neural networks.**
They are claims about the dynamics of a bookkeeping system: whether a threshold
is well posed, whether a gate can be steered, whether a curriculum converges,
whether a seal leaks, whether a provenance cascade is transitive. Those are
testable in a discrete-event simulator with a synthetic solver, on a laptop, in
days — with no model in the way.

This matters because the alternative ordering is expensive and uninformative.
If you build L7 on a GPU and it underperforms, you cannot tell whether the
mechanism is wrong or the model is small. Test the mechanism where it is a
hundred lines of numpy, and only spend GPU hours on what survived.

Three rigs, in strict order:

| Rig | What it is | Runs on | Covers |
|---|---|---|---|
| **A** | Ledger simulator, no model. Synthetic world with known ground truth: which regions are learnable, which are noise, which skills genuinely transfer. | This M2, seconds–minutes | ~18 of 24 experiments. **Where most breakage will be found.** |
| **B** | Small real model, MLX, ≤3B at 4-bit | This M2, 8 GB — tight but feasible | The 6 claims that genuinely need language |
| **C** | Rented GPU, episodic | Modal / RunPod, hours | L7 adapter compilation and Stage 5–6 co-evolution, **only for survivors** |

The 8 GB ceiling on this machine is a real constraint and it drove the rig
split. It is not a limitation to work around — it is the reason to do the
structural work first, which is the right order anyway.

---

## 1 · Status — two experiments run, both found breakage

Rig A is built and two of the cheap decisive tests are complete.

### E1.1 · `occupied rank = #{σ_k > ε}` is not a well-posed budget

Part II §A turns the subspace budget from a guessed integer into "a constraint
you can check." That requires ε to be a property of the system rather than a
knob. Two conditions were pre-registered: **K1** a decade-wide plateau where
rank is insensitive to ε; **K2** held-out queries carry <5% energy in the
sub-ε "free" subspace.

```
stream                                   plateau  swing   r95  free    leak  interf   K1   K2  verdict
bimodal (shape the design assumes)          3.14   24.0    23   105   0.087   0.111   ok   no  FAIL
exponential (soft knee)                     1.30    1.6    26   102   0.000   0.003   ok   ok  PASS
power law alpha=1.0                         0.07   29.0    99    29   0.750   0.190   no   no  FAIL
domain mixture (realistic WAM traffic)      2.33    1.1   122     6   0.059   0.105   ok   no  FAIL
```

Three findings, in increasing order of how much they cost the design:

**Power-law features destroy the budget outright.** No plateau at all (0.07
decades), rank swings 29× across two decades of ε, and **75% of held-out query
energy sits in the subspace declared free**. If real transformer feature
covariances are heavy-tailed — and they generally are — "free rank" is where
most of the traffic lives.

**Realistic mixed traffic leaves no budget to spend.** Twelve domains at rank
10 in a 128-dim space commit 122 directions; **6 are free**. The threshold is
well posed here (2.33-decade plateau) and the budget it computes is empty. Every
allocation immediately hits `basis = ∅ → enqueue(reclaim_arm)`. The design's
own gate algorithm (Part II §E) then returns `DEFERRED` forever: the system
deadlocks into permanent reclamation.

**Even the design's own assumed shape fails, and the failure is quantised.**
Leakage as the cut moves through a true boundary at rank 24:

```
keep top-20   leakage 0.170        each committed direction ≈ 0.042 of energy
keep top-22   leakage 0.087   ← where the automatic knee-finder landed
keep top-23   leakage 0.041
keep top-24   leakage 0.002   ← the true boundary
keep top-26   leakage 0.002
```

Leakage moves in **units of one whole committed direction**. ε is a hard cut,
and the eigenvalues nearest the cut are by construction the ones whose ordering
is least certain — so the mechanism is least reliable exactly at the boundary
it depends on. There is no safety margin: off by one direction means an adapter
is allocated straight through a committed skill. Placing ε correctly requires
already knowing the rank you are using ε to compute.

**What this costs:** the computed interference cap and the computed reclamation
set (Part II §A) both rest on counting free rank. Neither survives as stated.
See §5 for what to try instead.

### E1.4 · The gate reading and the gap reading cannot be the same number

Part II §A's headline is "epistemic uncertainty and plasticity headroom are the
same number." The gap-set reading (Part I §3) treats L4's posterior variance as
"a free map of what the system does not know." The word carrying the weight is
*epistemic* — and posterior predictive variance is epistemic **+ aleatoric**.
The design never separates them, and the two readings need opposite things:

- the **gate** must use *predictive* variance — a region where outcomes are
  random is one where weights should not be trusted;
- the **gap set** must use *epistemic* variance — a region where outcomes are
  random is one where practice is worthless.

Worse, frontier shaping actively selects for the confusion. SESA's reward
`4(ℓ+p̂)(h−p̂)` peaks at `p̂ = 0.5`, and a coin flip sits at 0.5 permanently. A
pure-noise region earns **the maximum possible proposer reward, forever.**

11 learnable regions + 1 pure-noise region, 400 cycles:

```
arm                   noise share  x fair    comp   short  gate rho   A1   A2  verdict
predictive                  0.513     6.2   0.925   0.002      1.00   no   ok  FAIL
predictive+cap              0.341     4.1   0.925   0.001      1.00   no   ok  FAIL
epistemic                   0.160     1.9   0.927   0.000     -0.99   ok   no  FAIL
reducible                   0.083     1.0   0.927   0.000      1.00   ok   ok  PASS
```

**The magnet is real and severe.** One coin-flip region in twelve captures
**51% of the practice budget** — 6.2× fair share. And it is an absorbing state,
not a transient: noise share climbs 0.086 → 0.49 as the horizon lengthens,
because as learnable regions master out (`p̂ → 1`, reward → 0) the coin flip
becomes the only region still sitting at the reward peak. The steady state of
this curriculum is *practise the coin flip forever*.

**Part III §6.3's per-source cap does not fix it** — 6.2× only falls to 4.1×.
The cap binds on *sources*; the noise region sits inside a legitimate source.

**Switching to epistemic variance fixes the magnet and inverts the gate.**
ρ = **−0.99**: not uninformative, *anti*-correlated. Practice accumulates
visits in the noise region, driving its parameter variance to the lowest in the
world while its error rate stays the highest. The gate becomes maximally
confident exactly where the system is most wrong.

**Honest scoping of the damage.** Competence lost on learnable regions is only
~0.005 even at 51% waste, because those regions are on a diminishing-returns
approach to their ceilings and get there anyway. So this is a **compute
misallocation, not a capability loss** — which still matters, since Part III
§10 makes cost a first-class in-loop signal and this is precisely the
misallocation the router exists to prevent.

**The fix that passes both:** score gaps by *reducible* variance — the decline
in variance attributable to prior practice, a derivative rather than a level.
Fair share exactly (1.0×), gate ρ = +1.00, no competence cost. A region that
does not improve when practised stops asking for budget. This is a small change
to the gap-set definition and it is the first concrete repair this plan
proposes.

---

## 2 · The claim ledger

Every load-bearing claim, its rig, and what would falsify it. ★ marks a
pre-registered prediction that it breaks. Full kill criteria live in each
experiment's docstring; `claims/claims.yaml` is the machine-readable version.

### Root 0 · The invariants (I1–I6 are asserted and have never been tested)

| ID | Claim | Rig | Kill criterion |
|---|---|---|---|
| E0.1 | **I4 recompilability.** Competence regenerates from provenance alone. The design calls this "a design intention rather than a tested property." | A→B | Any non-tombstoned competence not returning within δ. Also: recompile cost vs ledger length — I4 is worthless if it is O(ledger) and ledger is unbounded. |
| E0.2 ★ | **Tombstone cascade is transitive.** "A tombstone cascades to every adapter whose provenance contains the entry." | A→B | Any residual influence after cascade. **Predicted break:** the real path is entry → skill card → card-conditioned rollouts → adapter, and §7 says explicitly that the card *generates the adapter's training set*. If provenance is recorded entry→adapter, the cascade never reaches the parametric carryover. Unlearning is then approximate, which is the exact property L7 claims to have eliminated. |
| E0.3 | **I3 no-compounding under synthetic experience.** L8 needs no new containment because synthetic entries reuse the untrusted-content quarantine. | A | Any promotion whose provenance closure contains only `synthetic` + `inferred`. Tests whether Part III §6.1's "chain to an `observed` entry" is enforceable when the task derives from a *gap* — itself a statistic over many entries. Is that a chain? |
| E0.4 | **I1 grounding / claim-level auditability.** | A | Fraction of promoted capabilities whose provenance resolves to `observed` < 1.0. |

### Root 1 · Estimator quality

| ID | Claim | Rig | Kill criterion |
|---|---|---|---|
| E1.1 | Occupied rank is a computed budget | A | **DONE — FAILS.** See §1. |
| E1.2 ★ | **Three-λ unanimity makes rank release safe** (Part III §2). | A | **Predicted break, and it is the fix that breaks:** allocation uses the short estimator, release requires unanimity across all three. The long-λ estimator releases almost nothing, so the budget fills monotonically and every region deadlocks into `reclaim`. Measure time-to-deadlock vs λ_long. Compounds with E1.1's finding that realistic traffic starts with 6 free directions. |
| E1.3 | **Posterior variance predicts real error** (Part III §1's own recommended audit). | B | Spearman ρ < 0.5 between predicted variance and realized error on held-out real queries. Note E1.4 already shows ρ = −0.99 for the epistemic reading in simulation. |
| E1.4 | Posterior variance is an epistemic gap detector | A | **DONE — FAILS.** See §1. |
| E1.5 | **"Cheap-but-not-free at depth."** Per-layer R_t via truncated SVD over an activation buffer. | B | Wall-clock and memory per layer at realistic buffer sizes; and buffer size below which the spectrum estimate is too noisy to threshold at all (feeds back into E1.1). |

### Root 2 · Verification coverage and cost — *the binding constraint*

Part III §0 argues six of ten reported problems collapse to one: improvement
rate is bounded by verifier availability and cost. This plan takes that
seriously and weights the phase schedule accordingly.

| ID | Claim | Rig | Kill criterion |
|---|---|---|---|
| E2.1 ★ | **Probe harvesting turns the Assay from a content line into a sealing policy** (Part III §5). | A→B | **Predicted break — tier laundering.** "Every real interaction that resolves with a verified outcome is a candidate probe" does not say *at what tier*. A T3-judged outcome that gets frozen and hidden becomes a T2 probe by the act of concealment. Kill: any harvested T2 probe tracing to a T3 judge. Then measure what fraction of traffic survives a strict T0/T1-only filter — **if that fraction is small, harvesting supplements exactly the domains that already have verifiers and does nothing for the T3 domains that need them.** Which would mean it does not attack the binding constraint at all. |
| E2.2 | **Verifier synthesis** (Part III §9) — the discovery/verification asymmetry as a construction recipe. Declared the highest-leverage item in the design. | manual → B | Do this by hand on one T3 domain before building any machinery, exactly as Part III's own roadmap item 6 says. Kill: synthesized verifier fails to agree with held-out human labels, or fails to reject known-bad cases the base judge accepted. Deliverable is a yes/no on whether the recipe is real, plus an honest read on which domain classes it can reach. |
| E2.3 | **The staged ladder makes an archive affordable.** | A | Spearman ρ between SMALL-suite and FULL-suite scores. If low, the cheap filter rejects good candidates and the cost story collapses — the ladder would be buying affordability with false negatives it never measures. |
| E2.4 | **Sealed-suite drift fires before the incumbent score does.** | A | Plant a metric with a known shortcut (the design's own Goodhart canary drill). Kill: incumbent score moves first. |
| E2.5 | **Harvested suites inherit traffic coverage bias** (the design admits this). | A | Quantify it: measure suite coverage against the true region distribution, and how fast bias compounds when the harvested suite also steers the curriculum. |

### Root 3 · Signature ontology and transfer

| ID | Claim | Rig | Kill criterion |
|---|---|---|---|
| E3.1 | **Ranking by net transfer accumulates abstractions; ranking by target gain accumulates patches** (Part II §B). The central generalization claim. | A | Build a world with known ground-truth transfer structure. Two arms, N promotions each. Kill: gate-B does not beat gate-A on held-out *compositional* tasks. |
| E3.2 | **Writing a τ row for every off-target Δ collapses the cold start** (Part III §4a). | A | The τ sample from rejected candidates is *censored by the same gate*. Compare T estimated from the censored sample against ground truth. Kill: bias is large and systematically favours the diagonal — that would make T confidently wrong rather than merely sparse, which is worse than cold start. |
| E3.3 ★ | **"Choose the partition that maximizes off-diagonal mass in aggregated T"** (Part III §3). | A | **Predicted break — the objective is degenerate.** Off-diagonal mass is maximised by maximally fine partitions (one region per task makes everything off-diagonal). As stated this is not an optimization, it is a drive toward atomisation. Needs a regulariser or an MDL term the design does not specify. Cheap to demonstrate. |
| E3.4 | **Storing τ at observation grain makes re-partition a re-aggregation, not data loss.** | A | Re-partition and verify T reconstructs. This one I expect to hold — it is the strongest mechanism in Part III. |

### Root 4 · Search space and loop safety

| ID | Claim | Rig | Kill criterion |
|---|---|---|---|
| E4.1 ★ | **The scalar/textual firewall** (Part I §6). "The system can talk itself into any hypothesis it likes and still cannot talk itself into a promotion." | A | **Predicted break.** The firewall governs who may *write* posteriors. But the textual channel picks which candidates get proposed, and only proposed candidates get scored — so it controls the sampling distribution over the scalar channel without ever writing to it. Kill: a biased generator shifts arm posteriors measurably while writing zero scalars. Then test the fix: force a fraction of each arm's evaluations to be generator-independent. |
| E4.2 ★ | **The blast-radius fixed point.** L9 may never edit L1's schema, the Assay, the gate's tier requirements, or the rule. | A (static analysis + injection) | **Predicted break — the seal leaks by construction.** §4 puts "L7's gate thresholds" and "retrieval policy" in the L9-*editable* set. But ε, ε_reg, `noise_leeway` and the shrinkage on w are thresholds, so L9 can lower the bar without touching any tier requirement; and T1 counterfactual replay *runs through retrieval*, so L9 owns code the T1 verifier depends on. Deliverable: the actual dependency graph of the Assay on L9-editable code, and a corrected minimal non-editable set. |
| E4.3 | **Refutation scope errors are bounded by expiry.** | A | Flood with over-broad scopes. Measure cycles-to-surface. The design admits no gap detector will ever see these holes — confirm, then test two candidate fixes: mandatory expiry sampling (periodic re-attempt regardless of clock) and a scope-width penalty. |
| E4.4 | **Refutation permanence is affordable.** I6 exempts `refuted` from decay. | A | An undecayable type grows monotonically against a bounded retrieval budget. Find the crossover where refutations dominate retrieval. |
| E4.5 | **Typed arm tuples are a genuine search-space expansion** (Part III §7). | A | Enumerate the full `artifact × edit_type` product; count semantically valid cells. Kill: <10% valid means the mechanism is mostly a rejection engine, not an expansion. |

### Rig B · The claims that genuinely need a model

| ID | Claim | Kill criterion |
|---|---|---|
| E5.1 | L2 compresses to ~1/400 of latent pages, megabytes per user | Measure ratio and retrieval quality at that ratio |
| E5.2 | Memory Interleave (retrieve → infer → retrieve) beats single-shot | No significant gain over a matched retrieval budget |
| E5.3 | K-consecutive-failed-reflections beats a learned decision-point predicate (Part II §D) | Latency/accuracy frontier vs the predicate alone |
| E5.4 | Skill-card admission at cos ≤ 0.93, and "failure despite retrieved guidance" as the sharpest distillation signal | Admission control does not prevent bank collapse; or the signal does not outperform uniform failure sampling |
| E5.5 | **Counterfactual replay is "the one signal that is free and clean"** (Part II §F Stage 1) | Not clean if a candidate write changes retrieval for *subsequent* queries — then Δ is contaminated by downstream effects and is not causally attributed to one decision. Measure the contamination directly. |

---

## 3 · Phase schedule

Ordered by *how fast a result can invalidate a tier*, not by stack order.

**Phase 1 — cheap decisive (Rig A).** E1.1 ✅, E1.4 ✅, then E0.2, E4.2, E3.3.
All five can kill a mechanism outright and none needs a model. Two already
have. E4.2 and E3.3 are close to pure analysis.

**Phase 2 — loop dynamics (Rig A).** E1.2, E3.1, E3.2, E4.1, E4.3, E4.4, E0.1,
E0.3, E0.4, E2.3, E2.4, E2.5, E4.5. This is where the simulator earns its
keep: every one of these is a question about whether a loop converges, and
none of them is a question about language.

**Phase 3 — the binding constraint.** E2.1, then E2.2 by hand on one domain.
Per Part III §0 this is the highest-leverage work in the whole programme, and
E2.2 is a research bet that should be settled manually before any machinery is
built around it.

**Phase 4 — Rig B.** E1.3, E1.5, E5.1–E5.5. Stand up the MLX harness. Feed
E1.3 and E1.5 back into E1.1: the simulator asked "under what spectrum shapes
is the budget well posed" — Rig B answers "and does a real model produce one."

**Phase 5 — Rig C.** Only mechanisms that survived. Given the E1.1 result, L7
adapter compilation should not get GPU hours until the budget question has an
answer.

**Bootstrap order note.** The design's own Part I §11 sequence (L1 types → seal
the Assay → fast path → L9 on T0 → router → L8 challenger) is a *build* order
and it is a good one. This is a *test* order and it deliberately runs the other
way, hitting the deepest assumptions first. Both should run; they are not in
conflict.

---

## 4 · Method commitments

**Pre-register kill criteria before running.** Every experiment states its
falsification threshold in its docstring before the first run. This is the
scalar/textual firewall applied to the research process: without it, a result
can always be narrated into a pass.

**Report metric bugs as findings.** Two occurred already while building E1.1
and E1.4 — a plateau detector that counted saturation regions as knees, and a
world model where competence advanced per cycle rather than per task, hiding
the cost of misallocated budget. Both changed the result. Both are in the git
history and noted in the code.

**A repaired mechanism must be re-registered and re-run.** The `reducible`
variance fix from E1.4 is a proposal that passed one test, not a fix. It needs
its own kill criteria — in particular, whether a derivative signal is stable
under noisy pass-rate estimates, which is untested.

---

## 5 · Repairs already indicated

Two, from the two experiments run:

**R1 · Replace counted free rank with an energy criterion (from E1.1).** The
failure is that ε is a hard cut on a soft spectrum, so being off by one
direction is unbounded damage. An allocation rule that does not require a knee:
allocate a candidate adapter's basis, then *measure* the resulting perturbation
on held-out traffic and reject if it exceeds a tolerance. This replaces "count
directions below ε" with "measure what the write actually disturbs" — which is
a T1 counterfactual, a verifier the design already has. It also composes with
the promotion gate instead of sitting beside it. Cost: an eval per allocation
instead of a threshold comparison. Worth testing against the current rule.

**R2 · Score gaps by reducible variance, not variance (from E1.4).** A
derivative, not a level. Passed both kill criteria at zero competence cost.
Needs its own testing for stability under noisy p̂.

Neither is built. Both belong in Phase 2 as arms to compare, not as adopted
fixes.

---

## 6 · Reference implementations to harvest

| Repo | Take | Lands in |
|---|---|---|
| [Aedelon/titans-pytorch-mlx](https://github.com/Aedelon/titans-pytorch-mlx) | **MLX port — the one that actually runs on an 8 GB M2** | Rig B backbone |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | surprise + momentum/decay write gating | L0/L4 |
| [MemTensor/Metis](https://github.com/MemTensor/Metis) | native memory state | L4 |
| [autoLearnMem/AutoMem](https://github.com/autoLearnMem/AutoMem) | typed memory actions, policy/task split | L1 types, L6 |
| [SakanaAI/continuous-thought-machines](https://github.com/SakanaAI/continuous-thought-machines) | latent ticks, halting head | L5, E5.3 |
| [VectorSpaceLab/arex-model](https://github.com/VectorSpaceLab/arex-model) | bi-level loop, critical-interval credit | E2.2, Part II §C |
| [jennyzzt/dgm](https://github.com/jennyzzt/dgm) | archive, `score_child_prop` parent selection, staged eval ladder | L9, E2.3 |

---

## 7 · Repo layout

```
rig_a/core/         spectrum.py   R_t and the three readings
                    world.py      synthetic practice world + challenger
rig_a/experiments/  one file per experiment, kill criteria in the docstring
rig_b/              MLX harness (Phase 4)
rig_c/              GPU jobs (Phase 5)
claims/claims.yaml  machine-readable claim ledger, status tracked per claim
results/            one JSON per run, seeded and reproducible
```

Run an experiment:

```bash
.venv/bin/python rig_a/experiments/e1_1_spectrum_knee.py
```
