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

## 1 · Status

| ID | Claim | Verdict |
|---|---|---|
| E1.1 | `#{σ_k > ε}` is a computed budget | **PARTIAL** — fails on 2 of 4 streams |
| E1.1b | the budget works under the energy/GPM criterion | **PASS** — all 4 streams |
| E1.4 | posterior variance is an epistemic gap detector | **FAIL** |
| E0.2 | the tombstone cascade reaches the weights | **PARTIAL** — one arm withdrawn |
| E0.2b | *rebuild:* functional ground truth + a third influence path | **FAIL** on both criteria |
| E4.2 | the blast-radius fixed point seals the Assay | **FAIL** |
| E3.3 | max off-diagonal mass is a partition objective | **FAIL** |
| E2.1 | probe harvesting widens the verifiable surface | **FAIL** on 2 of 3 criteria |

**Two published conclusions have been withdrawn.** Both are recorded in
`claims/claims.yaml` under `retracted:` and in `bugs:` — see §4a.

- E1.1's headline, *"realistic traffic leaves 6 free directions of 128, so
  allocation deadlocks into permanent reclamation,"* was an artifact of a
  generator that re-drew its domain subspaces on every call. Train and held-out
  sets came from unrelated subspaces. With it fixed the domain mixture **passes**
  with 68 free directions and zero leakage.
- E0.2's *"transitive closure fixes it completely"* was a **tautology**:
  `record_provenance(transitive)` and `true_influencers` execute the same loop
  body, so the arm could not have returned any other answer.

**And the E1.1 conclusion is reversed.** E1.1 tested only the literal
formulation — thresholding eigenvalues at ε, placed at the curvature knee —
while `rank_for_energy`, implementing GPM's actual criterion, sat in the same
module documented as "the parameter-free version of the budget question,"
computed, and excluded from the verdict. E1.1b puts it in the verdict path.
All four streams pass, including power law (which E1.1 said leaked 75%) and the
realistic mixture. The budget only empties at 5× over-subscription.

So the subspace budget is **sound**, and Part II §A states it in a form that is
ill-posed while pointing at a form that is not. That is a specification defect,
not an architectural one — which means the earlier framing of *"four of five are
spec defects, E1.1 is the exception"* was wrong in both halves. Nothing found
so far threatens the ledger-first thesis.

### E1.1 · The *literal* formulation `#{σ_k > ε}` is not well posed

**Scope, corrected.** This tests only the mechanism as Part II §A literally
states it, with ε at the curvature knee. It is not a verdict on the subspace
budget — see E1.1b, which reverses that.

Two conditions were pre-registered: **K1** a decade-wide plateau where rank is
insensitive to ε; **K2** held-out queries carry <5% energy in the sub-ε "free"
subspace.

```
stream                                   plateau  swing   r95  free    leak  interf   K1   K2  verdict
bimodal (shape the design assumes)          3.14   24.0    23   105   0.087   0.111   ok   no  FAIL
exponential (soft knee)                     1.30    1.6    26   102   0.000   0.003   ok   ok  PASS
power law alpha=1.0                         0.07   29.0    99    29   0.750   0.190   no   no  FAIL
domain mixture (realistic WAM traffic)      2.33    1.1    60    68   0.000   0.000   ok   ok  PASS
```

Two findings survive:

**Under this ε-placement, power-law features destroy the budget.** No interior
plateau (0.07 decades), rank swings 29× across two decades of ε, 75% of
held-out energy in the "free" subspace. E1.1b shows this is a property of the
*placement rule*, not of power-law spectra.

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

**What this costs:** ε as an eigenvalue threshold is unusable. That is a defect
in how §A is *written*, and E1.1b shows the mechanism it is trying to express is
fine.

### E1.1b · The budget works under its strongest faithful reading

The design points at GPM directly — *"truncated SVD over a small activation
buffer — what GPM actually does"* — and GPM does not threshold eigenvalues. It
keeps the top-r directions holding a chosen fraction ρ of feature energy. That
has no scale and cannot be knocked off by a mis-scaled spectrum.

Choosing r on a calibration sample and measuring on a fresh one (choosing and
measuring on the same sample would make leakage circular by construction):

```
  power law alpha=1.0
     retention  committed   free  leak cal  leak test   interf   usable  safe
         0.950        100     28    0.0522     0.0518   0.0480      yes   yes

  domain mixture (realistic WAM traffic)
         0.950         61     67    0.0562     0.0595   0.0494      yes   yes
         0.990         87     41    0.0112     0.0113   0.0237      yes   yes
         0.999        106     22    0.0011     0.0012   0.0092      yes   yes
```

**All four streams pass**, power law included. Test leakage tracks calibration
leakage at ~1.0×, so the spectrum estimate generalises off its own sample. A
capacity sweep over dim ∈ {128, 256, 512} and 4–64 domains finds the budget
empties only at a **fill ratio of 5.0** — five times over-subscribed. At every
realistic ratio there is ample free rank.

This is the single most important correction in Phase 1. The mechanism E1.1
declared broken is sound; §A just states it in the one form that does not work,
while citing the form that does.

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

### E0.2 · The tombstone cascade almost never fires

L7's strongest safety claim — "unlearning by construction rather than by
approximation," the one thing it offers that EWC and LwF cannot — depends on
recorded provenance covering every path an entry took to the weights. Part I §7
describes a path it may not cover: *"the card is what generates the adapter's
training set."*

```
provenance policy     incomplete  missing  cascade  leaks    rate    worst   U1   U2  verdict
direct                         6      186     0.13    186   0.959   0.2640   no   no  FAIL
transitive                     0        0     3.23      0   0.000   0.0000   ok   ok  PASS
```

**Withdrawn.** The `transitive` row is a tautology:
`record_provenance(policy="transitive")` and `true_influencers` execute the same
loop body, so U2 is `X − X = ∅` for every world, seed and parameterisation. The
arm could not have returned another answer. What it demonstrated is that a
two-hop influence model is closed under two-hop closure.

The 95.9% for `direct` is the difference between two code paths, and its
magnitude is set by how often this world's rollouts happen to cite card sources
directly — a knob, not a measurement. That a gap *exists* was established by
argument before the run; the number added nothing.

Two deeper problems, both of which require a rebuild rather than a patch:

**The ground truth shares the mechanism's model.** `true_influencers` enumerates
exactly two hop types, so a third path is invisible to the ground truth as well
as to both arms — and the design has real third paths: entry → gap statistics →
challenger task distribution → which rollouts exist; entry → retrieval ranking →
which card surfaced. A second measurement would not have caught this, because it
would have shared the same definition.

**Set membership is the wrong shape for parametric carryover.** If a tombstoned
entry shaped a card, the card would have been different, so *every* rollout
conditioned on it would have been different — not only rollouts citing that
entry.

**The finding E0.2 owed and did not produce.** Transitive cascade invalidates
~54% of adapters per tombstone, and adapters are hours-to-days per the design's
own substrate table. That implies a **tombstone-rate ceiling** above which
unlearning-by-construction is unaffordable — and that ceiling, not the leak
percentage, is what decides whether L7's strongest claim is operationally real.

### E4.2 · The seal leaks two independent ways

```
Assay depends on 11 artifacts transitively
of which L9 may edit: ['dispatch', 'retrieval_policy', 'tool_schemas']
  assay_t0     reachable-and-editable: ['dispatch', 'tool_schemas']
  assay_t1     reachable-and-editable: ['retrieval_policy']
  assay_t2     reachable-and-editable: ['retrieval_policy']
  assay_t3     reachable-and-editable: ['retrieval_policy']
```

**Reachability.** All four Assay tiers run on code L9 is permitted to edit. A
verifier L9 may not edit still executes through a retrieval policy it may. T1
counterfactual replay replays real queries — through retrieval.

**Thresholds are not requirements — and the rule bounds neither their values
nor their joint setting.** The finding is the curve, not an endpoint:

```
noise_leeway, others at baseline:     eps_reg, others at baseline:
   0.000  ->  0.086                      0.005  ->  0.002
   0.020  ->  0.063                      0.020  ->  0.005
   0.100  ->  0.005  (baseline)          0.100  ->  0.018
   0.200  ->  0.000                      0.500  ->  0.018

spread from any ONE threshold:  0.086
all three relaxed together:     0.322   (vs 0.005 baseline)
```

Quoting the 64× endpoint alone was the weaker framing: it invites the reply
that no reasonable operator would set `eps_reg` to 0.50 — which concedes the
point, because the rule does not require an operator to be reasonable, and an
L9 archive optimising for promotion count is not one. The rule enumerates
*which* thresholds are editable and says nothing about *what values* they may
take, so the reachable set is unbounded and compliance is reported across all
of it. Individually each threshold is a modest lever; the gate is a conjunction,
so they compound.

The deliverable turned out more useful than the finding. Sealing the whole
dependency closure **over-seals** — it would freeze `l1_entries`, `l2_pages`,
`l3_views` and `l7_adapters`, which the system exists to update. The closure
splits in two: *seal* the harness code a verifier executes through, and *pin*
the derived views by stamping each Assay run with the versions it ran against —
which is the mechanism WAM already uses for L2 pages and weight versions. See
R4.

### E3.3 · The partition objective is monotone in fineness

```
  regions   offdiag mass   predictive MSE   rows/cell
        1         0.0000         0.001485     30000.0
        4         0.1957         0.000910      1875.0   <- true structure
       32         0.9058         0.000978        29.3
      192         0.9908         0.001577         0.8   <- objective's argmax
```

Splitting a region can only move mass off the diagonal, never onto it, so the
objective has no opposing force and its argmax is **total atomisation** — one
region per signature, 0.8 measurements per cell. The planted 4-cluster truth
scores 0.1957, near the *bottom*: the objective does not merely fail to find
the true structure, it ranks it as nearly the worst candidate available.

Part III §3 names this exact failure — "too-fine signatures make T diagonal and
starve transfer" — and then adopts an objective that is monotone in fineness.

**Replacement, and the limit of what was shown.** Score a partition by the
held-out predictive error of the T it induces. Too coarse averages away
structure, too fine has no data per cell, and the argmin lands exactly on the
planted 4 clusters.

The degeneracy half is solid — monotonicity in fineness is a structural
property of the objective, not a fact about this world. **The repair half is
not established.** Candidate partitions are constructed nested with the planted
structure and k=4 is in the family, so recovering 4 is close to guaranteed.
That is a sanity check, not evidence the objective works when the true
structure is non-nested, overlapping, hierarchical, or absent from the
candidate family — which is the realistic case for a signature ontology.

### E0.2b · Rebuilt — and `transitive` provenance is provably incomplete

E0.2's ground truth shared the mechanism's model. This rebuild removes that two
ways: ground truth is **never enumerated** — an entry influences an adapter iff
*deleting it moves that adapter's weights*, the world run twice — and the world
adds a path no set-based closure can capture. Rollouts **retrieve** their
conditioning card by query similarity, so deleting an entry can hand a rollout
to a different card. Provenance records the card that *was* selected; the card
that *would have been* selected was never run and is recorded nowhere.

```
  policy          mean recall   always complete   over-fires
  direct               0.1259             0.083         0.00
  transitive           0.9130             0.611         0.00

  true cascade: 5.53 adapters per tombstone (69% of the fleet)
  retrieval flips: 3.41 rollouts reassigned per tombstone; 87% of tombstones cause >= 1
```

**C1 fails, and it corrects me.** `transitive` recalls **0.913, not 1.0** —
missing ~9% of adapters that deletion genuinely moves, with complete coverage on
only 61% of tombstones. My earlier claim that closure through the conditioning
card "closes it completely" was the tautological result and is wrong. **No
set-based closure can close this**, because the missing dependency is on a
counterfactual retrieval that was never executed. Note also that neither policy
*over*-fires: both under-invalidate, silently.

**C2 fails — the ceiling is real.** The correct cascade touches 69% of the fleet,
and adapters are "hours–days" per the design's own substrate table:

```
  cascade  touched     2h /day     8h /day    24h /day
       5%      3.2       15.00        3.75        1.25
      10%      6.4        7.50        1.88        0.62
      25%     16.0        3.00        0.75        0.25
      50%     32.0        1.50        0.38        0.12
      70%     44.8        1.07        0.27        0.09
```

Sustainable tombstones per day, fleet of 64. Below 1.0 deletion requests queue
faster than they clear. **Much of the plausible space sits below 1/day** — 25%
cascade at 8h is 0.75. Reported as a surface rather than a single number because
the 69% is a property of this world's card overlap, not of the design; what is
structural is that cost scales linearly with cascade breadth × recompile time,
and the design already fixes the second factor at hours–days.

### E2.1 · Probe harvesting: viable, but it launders tiers and samples the easy corner

Part III §0 argues verification coverage is the binding constraint on the whole
architecture, and §5 is the mechanism offered against it — so this is the
cheapest measurement with the largest consequence in the plan.

```
  corr   suite  laundered    rate  strict yield  mean bias   worst  covered   blind
  0.00   38053      10989   0.289         0.677     0.0100  0.0518    0.975   0.025
  0.25   37543      13934   0.371         0.590     0.0337  0.1020    0.974   0.026
  0.50   36938      16843   0.456         0.502     0.0515  0.0712    0.943   0.057
  0.75   36437      19624   0.539         0.420     0.0947  0.1160    0.945   0.055
  1.00   35956      22581   0.628         0.334     0.1756  0.2090    0.942   0.058
```

**H1 fails — tier laundering is real and large.** "Resolves with a verified
outcome" names no tier, so with no filter **29–63% of the sealed "T2" suite is
T3-adjudicated judge opinion**, carrying the authority of a sealed probe while
T3 is the one tier forbidden from promoting anything. Cheap to fix with an
explicit tier filter; the policy as written does not have one.

**H2 passes, contradicting this plan's own prediction.** I pre-registered that
a strict T0/T1 filter would yield too little to matter, and therefore that
harvesting "does nothing for the T3 domains that need it." **That was wrong.**
The strict filter yields 33–68% of traffic and covers 94–97% of traffic mass.
Harvesting is viable as a suite source; its problem is not yield.

**H3 fails at high correlation — the suite tests each domain's easy corner.**
Mean per-domain difficulty bias reaches 0.095 at correlation 0.75 and 0.176 at
1.0.

**The caveat that governs the whole result.** The correlation between
checkability and easiness is *swept, not measured*, and the per-domain
checkability profile is an assumption stated in the source file. What this
establishes is the shape of the damage as a function of that correlation, and
which regime harvesting requires. Locating real traffic on that curve is Rig B
work, and it is what actually decides whether §5's optimism is warranted.

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

**Phase 1 — cheap decisive (Rig A). ✅ Complete.** E1.1, E1.4, E0.2, E4.2,
E3.3. All five failed; the three predicted breaks broke for the predicted
reason. Total runtime under a minute on the M2.

**Phase 2 — loop dynamics and repair validation (Rig A).** Two tracks now,
run together:

*New claims:* E1.2, E3.1, E3.2, E4.1, E4.3, E4.4, E0.1, E0.3, E0.4, E2.3,
E2.4, E2.5, E4.5. Every one is a question about whether a loop converges, and
none is a question about language.

*Repair validation, framed adversarially.* Not "run R2–R5 against the arm they
replace" — a repair that beats the arm it replaced, in the world that motivated
it, has told you nothing. **Build the world designed to defeat each repair, from
outside the model class the repair assumes:**

| Repair | The world built to kill it |
|---|---|
| R2 reducible variance | a never-practised region (no prior variance to difference) and a region on a long plateau that later breaks through |
| R3 transitive provenance | a third influence path — entry → retrieval ranking → which card surfaced — that closure through cards cannot see |
| R5 predictive-error partitioning | true structure that is non-nested, overlapping, hierarchical, or absent from the candidate family |
| R4 value bounds | an L9 archive optimising promotion count against the bounds, to find whether any admissible setting still moves the bar materially |
| R6 tier filter | a domain where T0 verification exists but is systematically available only for the easy slice |

R4 additionally needs the cost of its *sealing* component measured separately
from its *bounding* component, since only the former narrows L9's search space.

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

**Ask of every kill criterion, before the run: is there any world that could
produce the other verdict?** If not, it is not a measurement. This costs about
five minutes per experiment and it catches E0.2 in the first thirty seconds —
`record_provenance(transitive)` and `true_influencers` are the same loop body,
so no world, seed or parameterisation could have returned anything but zero.

This is a *separate* discipline from the two-measurement rule below, and the
two catch different things. Two measurements catch **contradiction** — which is
exactly how B3 surfaced. They do not catch **correlated error**: when the
ground truth and the mechanism share a definition, both measurements agree and
both are wrong. A second measurement of `true_influencers − provenance` would
also have returned zero. Unit tests do not help either, because the code is
correct; it is the criterion that is empty.

**Treat a perfect in-rig score as a warning.** Three repairs currently score
perfectly — 0 leaks, exact recovery, 1.0× fair share. In a simulator whose
ground truth is authored alongside the hypothesis, a perfect score is more often
a signature of tautology than of strength. R3's perfect score was one.

### 4a · Rig bugs — four so far, every one produced a confident number first

All are in `claims/claims.yaml` under `bugs:`, in the code, and in git history:

- E1.1's plateau detector counted *saturation* regions (rank pinned at 1 or at
  `dim`) as knees, which credited a power-law spectrum with a 3.82-decade
  plateau it does not have.
- E1.4's world advanced competence per *cycle* rather than per *task*, so a
  region starved to one task learned as fast as one given a hundred, hiding
  the entire cost of misallocated budget.
- E0.2 first measured whether weights *changed* when the deleted value was
  substituted, and reported **zero leakage for a policy with 186 missing
  influence paths**. Cached weights are constants — they are stale, not
  sensitive — and only comparison against a clean recompile detects the leak.

- **B4** `stream_domain_mixture` re-drew its domain subspaces on every call, so
  train and held-out sets came from unrelated subspaces — 99% of train energy
  needed 37 directions while the same fraction of test energy needed 127 of 128.
  This **manufactured a headline finding** that was reported and has now been
  withdrawn.

B3 is instructive for one reason and B4 for another.

B3 produced a confident PASS on the claim this plan most expected to fail, and
was caught only because U1 and U2 contradicted each other — the argument for two
measurements.

B4 is worse and the lesson is different. It biased a result **against** the
design, survived a full write-up, and no second measurement would have caught it
because every measurement in that experiment drew from the same broken
generator. What caught it was asking whether a striking result might be an
artifact of a parameter I had chosen — the capacity sweep — which is a question,
not a cross-check. Both failures so far have been in the rig, not in the
architecture.

**A repaired mechanism must be re-registered and re-run.** The `reducible`
variance fix from E1.4 is a proposal that passed one test, not a fix. It needs
its own kill criteria — in particular, whether a derivative signal is stable
under noisy pass-rate estimates, which is untested.

---

## 5 · Repairs indicated

**None is adopted.** Each survived the test that motivated it, which is not the
same as a fix — see the Phase 2 framing below for why that bar is too low.

**R1 · State the subspace budget by cumulative energy, not by thresholding
eigenvalues (E1.1 → E1.1b).** Keep the top-r directions holding fraction ρ of
held-out feature energy; pick r on a calibration sample and measure on a fresh
one. **Demonstrated, not sketched** — all four streams workable, budget empty
only at 5× over-subscription. This supersedes the original R1 (a
measured-perturbation allocation rule), which is still worth having as a
belt-and-braces check at allocation time but is no longer needed to rescue the
mechanism.

**R2 · Score gaps by reducible variance (E1.4).** A derivative, not a level.
Passed both criteria at zero competence cost. Open: stability under noisy p̂.

**R3 · Record provenance as the transitive closure through the conditioning
card (E0.2 → E0.2b).** **Necessary but provably insufficient.** E0.2b measures
it at 0.913 recall against functional ground truth: it is a large improvement
over `direct` (0.126) and it cannot reach 1.0, because the residual dependency
is on a retrieval that was never run. Take it, but do not treat it as closing
the question.

**R7 · Verify unlearning instead of inferring it (E0.2b).** After a cascade,
recompute the affected adapters and compare against the held weights — the same
operation E0.2b uses as ground truth. Affordable because it runs once per
deletion, not once per promotion. Structurally the same move as R1: measure what
happened rather than count what a bookkeeping rule predicts. Open: verification
detects an incomplete cascade, it does not make the correct cascade affordable,
so C2 stands regardless.

**R4 · Bound threshold values, and split the boundary into seal and pin
(E4.2).** Three parts, in descending order of leverage:
  1. **Bound the values** permitted thresholds may take, not merely which
     thresholds are editable. This is the sharpest of the three and it costs
     nothing — it does not remove anything from L9's search space.
  2. **Pin** derived views by stamping each Assay run with the versions it ran
     against, reusing the L2 weight-version mechanism.
  3. **Seal** the harness code a verifier executes through. This one has a real
     cost: sealing the retrieval policy removes it from L9's search space,
     which is exactly where SIA found scaffold edits pay. Prefer (1) and (2)
     first and measure whether (3) is still needed.

**R5 · Score partitions by held-out predictive error, not off-diagonal mass
(E3.3).** Open, and the open part is the important part: validated only on
partitions nested with the planted truth, which is the case where it cannot
fail.

**R6 · Filter harvested probes by verification tier (E2.1).** One predicate,
removes 29–63% of a harvested suite that is laundered judge opinion. The
cheapest repair on the list. It does not address the representativeness problem,
which is the part with no repair yet.

### What the Phase 1 results say collectively

**Almost every failure so far is a specification defect**, and the two original
candidates for "architectural" both dissolved on closer testing:

- E1.1 looked architectural until the energy criterion was put in the verdict
  path. It is a defect in how §A is written.
- E0.2's severity was never measured; the number was a knob.

The defects are real and worth fixing — a provenance set recorded at the wrong
point, a boundary that constrains membership but not values, an objective with
no opposing force, a gap signal reading the wrong quantity, a harvest policy
with no tier predicate. Each repairs with machinery already in the design.

**One finding is not a specification defect.** E0.2b's C2 — the tombstone-rate
ceiling — is arithmetic on the design's central move. "Weights are a derived
view of the ledger" plus "real deletion via tombstone + cascade" plus "adapters
recompile in hours–days" jointly fix deletion throughput at
`capacity / (cascade breadth × recompile time)`. No rewording changes that. You
can shrink cascade breadth with more specialised adapters and less card overlap,
but you cannot drive it to zero, and the design offers no mechanism for bounding
it. **This is the first genuinely architectural cost found so far**, and unlike
the others it has no repair on the list — R7 detects an incomplete cascade but
does not make a correct one cheaper.

Nothing found so far threatens the ledger-first thesis. The deletion ceiling
constrains the *rate* at which one of its promises can be kept, not whether the
promise is coherent.

The honest counterweight: **both serious errors found in Phase 1 were in the
rig, not in the architecture**, and one of them manufactured a headline that
survived a full write-up. The base rate of my own error is currently higher
than the base rate of confirmed architectural error, and that should be priced
into how much weight any single Rig A result carries.

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
