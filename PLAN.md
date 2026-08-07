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
| E1.1b | the budget works under the energy/GPM criterion | **PASS** — but see E1.1c |
| E1.1c | …does it hold *per domain*, not just on the traffic mean? | **FAIL** — partially reverses E1.1b |
| E1.1d | tail-safe free rank vs subscription ratio | **the missing curve** — boundary sits at 1.0× |
| E1.2 | three-λ unanimity makes rank release safe | **FAIL** on both criteria — architectural |
| E1.4 | posterior variance is an epistemic gap detector | **FAIL** |
| E0.1 | **I4** — competence regenerates from provenance | **FAIL** on K1, K4, KB — blindness **12.4×** |
| E0.2 | the tombstone cascade reaches the weights | **PARTIAL** — one arm withdrawn |
| E0.2b | *rebuild:* functional ground truth + a third influence path | **FAIL** on both criteria |
| E4.2 | the blast-radius fixed point seals the Assay | **FAIL** |
| E3.3 | max off-diagonal mass is a partition objective | **FAIL** |
| E0.2c | *does the ceiling hold under better policies?* | **PASS** — it does not; E0.2b demoted. Its D3 superseded |
| E0.2d | is admission control a lever on cascade breadth? | **FAIL** — inverts E0.2c's D3. A **missing control surface** |
| E3.1 | net-transfer ranking accumulates abstractions | **PARTIAL** — T1–T3 pass, T4 fails |
| E3.1b | is T4's inversion caused by spillover, or by *asymmetry*? | **FAIL** — asymmetry artifact. Partially rehabilitates §B |
| E3.1c | does the transfer *matrix* beat a one-line breadth penalty? | **PARTIAL** — the statistic wins; the matrix is unevidenced |
| E3.1d | is E3.1's comp-only figure real? | **FAIL** — tiebreak artifact, spread 0.775. Figure withdrawn |
| E2.1 | probe harvesting widens the verifiable surface | **FAIL** on 2 of 3 criteria |
| E5.1 | **joint feasibility** — do the constraints intersect? | **~1%** of the space, and **0** at the design's own profile |

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

### E1.1c · E1.1b's PASS was a traffic-average artifact

**This partially reverses E1.1b by the same move E1.1b used on E1.1: not a new
mechanism, a corrected aggregation.**

`rank_for_energy` sums energy over calibration queries drawn at Zipfian rates,
and `leakage_by_rank` / `interference_by_rank` return a **mean over the same
distribution**. Choice and measurement are weighted identically — so protection
is allocated in proportion to frequency, a rare domain's subspace lands below
the cut and is classified free, and the metric that should catch it averages
over the distribution that made the domain rare.

Same G1 bar, asked per domain on equal-sized probe sets:

```
--- 16 domains, rank 8, rate ratio 16x ---
     rho    r  free  int_traffic  int_worst  leak_worst   failing
  0.9500   65    63       0.0430     0.1033      0.2049     12/16   <- E1.1b's operating point
  0.9900   90    38       0.0215     0.0727      0.0549      4/16
  0.9990  110    18       0.0089     0.0355      0.0079      0/16   <- tail-safe
```

**P1 fails** — 6/8, 12/16 and 25/32 domains exceed the bar at ρ=0.95 while the
traffic mean reads 0.043–0.058 and passes. **P2 fails and makes it structural**:
Spearman(rate, leakage) = **−1.000, −0.965, −0.917**, monotone in rarity across
all three configs. A single unlucky domain would not correlate.

**The number to carry forward:** tail-safe retention is ρ=0.999, free rank
**18/128 and 17/128** — 13–14% of dimension against the ~50% E1.1b reports at
ρ=0.95. Room for about two rank-8 adapters. *"Empty only at 5× over-subscription"
was measured at a retention that is not tail-safe* and is withdrawn as a
headline.

This matters here more than it would elsewhere: L7's admitted failure mode is
coverage bias, and the rare-domain tail is exactly the long-tail personal
knowledge the ledger exists to preserve.

**Both obvious repairs fail.** R-a (union of per-domain top-r) cuts worst
leakage ~7× and still misses the interference bar; R-b (frequency-balanced `R_t`)
barely moves. The honest repair is that **ρ stops being a free parameter and
becomes a measured one**, set from worst-domain interference — the same move as
R1 and R7, and affordable since it needs domain labels only at calibration time.

**Leakage and interference are not interchangeable, and interference binds.**
`interference_by_rank` writes into `U_free[:, :r]`, the *largest* sub-cut
directions. G1 should always be the interference arm; any repair evaluated on
leakage alone will look better than it is.

**The falsifier partly succeeds.** Panel C sweeps domain-subspace overlap: at
overlap 0.73 tail-safe relaxes to 0.995 and free rank triples to 57. So severity
is a function of **real per-region subspace overlap** — a Rig B measurement, and
the same activation pass §3 already schedules.

### E3.1b · T4's inversion is an asymmetry artifact — §B is partly rehabilitated

T4 applies spillover only where `kind == "patch"`; skills get none. But net
transfer is a **sum** over off-target regions, so adding the same constant to
every candidate shifts every score equally and cannot change the ranking. Only
an asymmetry can invert the margin — and T4's stated justification, *"a low-rank
update fit to one region is not region-confined,"* does not distinguish the two
candidate types. A skill adapter is also low-rank, fit to a *broader*
distribution, so on that reasoning it should spill **more**.

```
regime                              target  transfer   margin
clean                                0.264     0.349   +0.085
noisy + spurious                     0.075     0.206   +0.131
spill 0.02  patch only  (E3.1 T4)    0.015     0.006   -0.010
spill 0.02  symmetric                0.040     0.216   +0.176
spill 0.05  symmetric                0.034     0.245   +0.210
```

Symmetric spillover **preserves and improves** the margin. So the condition §B
owes is **not** the one this plan published. Not *"narrow patches must have
near-zero real off-target effect"* — that is the absolute condition and it is
implausible. The actual condition is **differential**: patches must not spill
more than skills by more than ~0.03, which is `0.5 × GAIN_SKILL` in this world's
units and should be carried as a **ratio**.

That condition is much weaker, arguably favourable a priori, and it makes the
Rig B measurement **cheaper**: measure the *difference* in off-target spillover
between a narrow adapter and a broad one, paired on the same probe set, rather
than the level of either.

### E3.1c · The transfer *statistic* is evidenced; the transfer *matrix* is not

The comparison that decides whether Root 3 is justified is not target-vs-transfer
— it is net transfer against the cheapest thing that also removes the reward for
narrowness: a scale-free breadth penalty that aggregates no per-region magnitudes.

Net transfer wins in all three regimes, by 0.093 / 0.031 / 0.088. The one-line
alternative loses, so the statistic earns its keep.

**But no arm in E3.1 ever consults accumulated history.** `transfer` scores
`d.sum() − d.max()` on the *current* candidate's measured deltas. So the
**statistic** has evidence and the **matrix** has none, in either direction —
and everything expensive hangs off the matrix: τ storage at observation grain,
the partition objective, the censoring bias, and the L3 compiled
signature-ontology view. T's three remaining uses — L8 curriculum prior, L7 merge
prior, diagonal-T as a memorisation diagnostic — are unevidenced and were never
the stated reason for building it.

---

### E1.1d · The whole record was measured on the cliff edge

Every free-rank number here is taken at one point. E1.1c's configs are 64, 128
and 128 directions of traffic against dim 128; E1.2's world is 16 × 8 = 128.
Both are **exactly Σr_d/d = 1**. Joint feasibility takes free rank as its primary
axis, so a feasible region computed at one subscription slice would repeat the
level-vs-ratio error one level up.

```
  overlap    subscr   tail-safe rho   committed   free   usable
  0.0          0.50           0.999          62     66     yes
  0.0          1.00           0.999         109     19     yes
  0.0          1.50           0.999         123      5      NO
  0.7          1.50           0.995          89     39     yes
  0.7          3.00            0.99          87     41     yes

  feasibility boundary (last usable subscription)
    overlap 0.0  ->  1.0    (19 free)      first unusable 1.5
    overlap 0.4  ->  1.0    (21 free)      first unusable 1.5
    overlap 0.7  ->  3.0+   (41 free)      no unusable point swept
```

**The boundary at low overlap is subscription 1.0 — which is exactly where every
prior measurement was taken.** One step up to 1.5× and the budget is gone (5–6
free); one step down to 0.5× and it is comfortable (66 free). So "17–18 free at
tail-safe" and "6–15 under the release rules" are readings *at the cliff*, and
that is the single most important context for both.

**Overlap moves the boundary by at least 3×**, so it is not a refinement — it
decides whether an operating envelope exists at all. Together with the real
subscription ratio it determines whether any L7 budget finding bites in
practice, which makes both Rig B measurements rather than one.

(At overlap 0.7 free rank is non-monotone in subscription — 97, 88, 69, 39, 50,
41 — most likely the discrete retention grid, since tail-safe ρ jumps between
0.99 and 0.995 there. Not interpreted.)

---

### E1.2 · The three-λ rule is unsafe in both directions at once

Part III §2's asymmetry — allocate on the short estimator, release only on
unanimity — is argued from blast radius and it is sound as far as it goes. What
it never prices is the **rate**. Release is what replenishes the free pool, and
unanimity means the pool replenishes at the speed of the *slowest* estimator.

```
  traffic-only free rank, no adapters: 69/128

  release rule    mean free  starve rate     sd   live  unsafe dirs  of which rare
  unanimity             6.3        0.711  0.040    1.1            0              0
  short_only           15.2        0.216  0.046    6.1         5222             57
  medium_only           8.7        0.507  0.062    2.8         2051             69
  long_only            10.7        0.678  0.097    1.3            0              0
  (mean of 12 seeds)
```

**U1′ fails.** The budget starts at 69 free directions and unanimity collapses it
to **6.3** — the ratchet destroys 91% of it. Requests starve at 0.711 against
short_only's 0.216, a **3.29×** ratio, and unanimity sustains 1.1 live adapters
where short_only sustains 6.1. *The rule adopted to protect the budget is what
empties it.*

**U2 is not a simulation result — it is a composition, and stronger for it.**
Allocation ranks on the short estimator in *every* arm, so no λ_long and no
release rule can touch it. Compose that with E1.1c and it is a two-line proof:

> allocation consults only the fast estimator · the only downstream catch is
> traffic-weighted · **therefore allocation errors in low-mass domains are
> undetectable, for any parameter setting**

That cannot be argued down by a seed count. The simulation only *sizes* it:
under `short_only` a traffic-weighted check reports 16.6% damage where the worst
domain is at 85.8% — a **5.2× blindness factor**; unanimity is less damaging
(0.517) and *blinder* (7.2×).

**The ratio is the finding, not the level.** Domains partition the space
disjointly here, so every allocation damages someone by construction and the
level cannot come out low — it is a distributional statistic at subscription
1.0 and overlap 0, which E1.1d shows is precisely the feasibility boundary and
the pessimistic end of the overlap axis. The *ratio* is two views of the same
damage, so it survives whether or not the damage was avoidable.

So §2's justification, *"if the direction was actually committed, the promotion
gate's non-regression test catches it,"* is false for structural reasons no
tuning reaches.

**Allocation order: predicted to be the mechanism, measured as a minor one.**
The allocator takes least-visited free directions first, and least-visited is
identical to rarest-domain — so the prediction was that it preferentially
targets the tail. Directionally confirmed and much weaker than expected:
ascending 0.858 damage at rank 11.1, descending 0.815 at 9.2, random 0.841 at
10.5. All three concentrate on the rarer half.

The reason is more damning than the prediction. Ordering only selects *within*
the free pool, and **the pool is rare-domain-dominated under every rule**,
because low traffic is what "free" looks like to a visit-weighted estimator.
Pool composition is the mechanism; ordering is a detail — which means allocation
order is not available as a repair.

**Why this is not a threshold to retune.** The two failures have *opposite*
repairs. Tighten release and the budget starves; loosen allocation and you write
into committed directions nothing downstream can see. No setting of the
three-λ rule is safe in both directions, and the design offers no third option.

**And it generalises past item 10.** Four measured symptoms of one anisotropy,
all in L7: the energy cut classifies rare-domain subspace as free (E1.1c); the
allocator's free pool is rare-domain-dominated under every ordering (E1.2); the
non-regression gate cannot see the damage there (E1.1c, E1.2 U2); and the gate's
`qᵀR⁻¹q` reads practised-but-unprobed directions as known (analytic, untested).

That is not four findings, it is one finding four times — so amendment item 10's
"two weightings of one estimator" is too narrow. The rule is: **any statistic
used for protection or allocation must be computed per region on equal-sized
samples; only statistics used for routing may be traffic-weighted.** Enumerating
Parts I–III against it finds **eleven protection sites**, three measured and
eight inferred — written up in
[docs/wam_amendment_weighting_rule.md](docs/wam_amendment_weighting_rule.md).
The pattern is that every statistic in the stack defaults to traffic-weighted,
because traffic is what the system sees: frequency weighting is not a choice
anyone made, it is what you get by not making one.

**The λ_long sweep says more than starvation.** U1′ fails across the *entire*
swept range — even the best point starves at 0.530 against `short_only`'s 0.216.
And that best point is **λ_long = 0.98, which is λ_medium**: unanimity is least
bad exactly where the three-timescale structure degenerates to two, so **the
third timescale contributes nothing but cost at every setting tested.** That is
a sharper indictment of §2 than the starvation number. (The sweep is
non-monotone; SEs are 0.011–0.028, so it is probably real, unexplained, and
irrelevant to the verdict.)

**Registered honestly:** U1's pre-registered absolute form does *not* fire,
because it assumed `short_only` would be comfortable and it is not. Re-registered
as relative; both verdicts print every run.

---

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

### E0.1 · I4 fails, and a pooled metric would have reported a pass

The largest unmeasured claim in the design — Part II §G calls I4 "the invariant
the entire design leans on" and "a design intention rather than a tested
property." Tested against the rev-2 amendment: three support categories,
per-region reporting, arms A4 and A6.

```
  arm                     ident  partial  over pool  over worst  blind   under
  A0 control                yes    0.000     0.0000      0.0000   0.00  0.0000
  A1a drift, uncapped        no    0.000     0.0674      0.7977  12.40  0.0000
  A1b drift, usage cap      yes    0.000     0.0000      0.0000   0.00  0.0000
  A2 tombstone               no    0.411     0.0000      0.0000   0.00  0.0000
  A3 harness drift           no    0.000     0.2160      0.4672   2.18  0.0000
  A4 ontology, usage draw    yes    0.000     0.0000      0.0000   0.00  0.0000
  A4 ontology, strat draw     no    0.000     0.2167      0.4163   1.92  0.0000
  A6 draw uniform            no    0.000     0.1649      0.6244   3.85  0.0000
  A6 draw stratified         no    0.000     0.2455      0.5195   2.13  0.0000
```

**K1 and KB fail together, and the blindness factor is 12.4× — the largest in
the programme.** Under ledger drift, pooled over-forgetting is **6.7%** while the
worst region loses **79.8%** of the competence it had. A pooled-only E0.1 reports
"6.7%, acceptable" while one region is essentially wiped. This is the predicted
failure precisely: use-based decay evicts what is rarely referenced, so the
provenance that decays is the sparse rare-region kind — **the arm most likely to
show over-forgetting is the arm a pooled metric is blindest to.**

**K4 fails, and it was flagged as the arm most likely to surprise.** A pure
signature re-partition — ledger, weights and harness all held fixed — changes
competence by **21.7% pooled, 41.6% worst**. The ontology *is* load-bearing for
adapter competence, so Root 3 reaches into Root 1's problem exactly as the
amendment warned.

That only became visible after fixing a defect of the now-familiar class: `A4`
originally ran under the usage draw, which never consults the partition, so it
was **inert by construction** and K4 "passed." Same shape as unanimity reporting
zero unsafe on a quantity its own rule excludes (B8).

**I4 is a *relative* invariant, and nothing in Parts I–III says so.** A1b is
inert: under a usage-capped draw, tail decay changes nothing because those
entries were never in the draw — the competence was already absent at *compile*
time, and I4 compares A against A′ with both equally impoverished. **I4 checks
recompile fidelity, not compile adequacy.**

**41.1% of items fall in the partially-tombstoned category**, about which the
amended I4 asserts nothing. That is the honest size of what an item-level test
cannot see.

**A3 vindicates the component-granular stamp.** Changing one harness component
produces 21.6% pooled over-forgetting, so without pinning components E0.1 would
confound the property under test with harness drift.

**A6 resolves the `|P|` question, then relocates it.** Capping the draw *does*
bound cost (450 → 300), so provenance IDs can be retained in full and the
cost/breadth tension **dissolves** rather than resolving. But *which* entries the
cap admits is a protection decision, and a usage-weighted draw is exactly the
forbidden weighting. Stratified drawing trades a higher pooled rate (0.246 vs
0.165) for a lower worst (0.520 vs 0.624) and lower blindness (2.13 vs 3.85) —
the weighting rule's tradeoff — and still does not reach 1.0×.

*Scope limit:* this world has no retrieval-mediated card selection, so E0.2b's
~8.7% irreducible under-forgetting cannot appear. K2 passing is a statement about
this influence model, not a contradiction of E0.2b.

---

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

### E3.1b · The compositional-only figure was measuring pool order

A number this plan published in three places turns out to measure nothing about
the ranking rule. Under net transfer:

```
regular skill (>=2 regions)   target= 0.060   transfer= 0.060
comp-only skill               target= 0.000   transfer= 0.000
patch                         target= 0.140   transfer= 0.000
already-learned skill         target= 0.000   transfer= 0.000
```

Three classes tie at exactly zero, and the selection loop uses strict `>`, so
the tie goes to whichever candidate the pool emitted first. Varying only the
tiebreak rule:

```
tiebreak           comp_only   compositional   skills
first                  0.550           0.333     9.88   <- the published number
random                 0.515           0.329     9.78
prefer_patch           0.005           0.053     5.65
prefer_skill           0.780           0.582    11.85
spread                 0.775
```

**The contamination is wider than the one figure.** Clean-regime *compositional
scores* span 0.053–0.582 across the same rules, so T1's margin rests on an
unspecified tiebreak too. It survives under neutral rules (first 0.333, random
0.329, both over target's 0.264) and inverts under an adversarial one. The
noisy regimes show spread 0.000 — noise breaks the ties — so T2's result is
untouched.

This is the same defect as B3, B6 and the empty kill criterion: **a quantity
that cannot vary with the thing it claims to measure.** It passed the
postcondition lint's letter (the metric does vary) while failing its intent
(it varies with something else entirely). The lint needs the stronger form:
*name the input that would flip it, and check nothing else flips it more.*

---

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

### E5.1 · 7.1% feasible, and the binding constraint is not the one I studied

Every experiment before this moved one thing with everything else at defaults.
That finds mechanisms; it cannot tell you whether a design is buildable, because
the constraints are coupled and each tightens under the others.

Six constraints over **97,200 configurations** spanning region count,
subscription, subspace overlap, provenance overlap, draw fraction, batch, decay
policy, decay rate and deployment profile. **6858 feasible — 7.1%**, with
feasible values strictly interior to every axis, so the density figure means
something.

**The verdict at the design's own profile is conditional, not absolute.**

```
  profile               max beta   window at 0.30   window at measured beta
  as specified              0.31             5..5                    EMPTY
  fast recompile            1.00            1..11                    1..11
  high parallelism          1.00            1..11                    1..11
  small fleet               1.00            1..11                    1..11
  tolerant latency          1.00            5..49                     6..49
```

At fleet 64 / 8h recompile / 4-way parallelism / 7-day tolerance, the C3∧C4
window is non-empty only for cascade breadth **≤ 0.31** — and E0.2d's lowest
breadth is **0.65** — though "measured" is doing more work there than the number
supports, since 0.65 came from `entry_multiplicity` in E0.2c's world and inherits
that chosen range. The direction is not in doubt at a 0.65-versus-0.31 gap; the
precision is. So the
window is empty there, but conditionally on breadth. Easing any one hardware
constraint opens it at any breadth.

**This is what promotes R9.** Four routes out: faster recompiles, more
parallelism, a smaller fleet, or breadth below ~0.31 — and only the last is not
a hardware purchase. E0.2d established the design has *no lever* on breadth, so
joint feasibility turns **provenance-aware admission from a missing mechanism
into the mechanism that decides whether an operating envelope exists.**

**The ranking I drew from the attribution table is withdrawn.** An earlier
version concluded "C1 is the least binding constraint" and that "most of this
programme's effort went to the non-binding constraint." Both are wrong, and my
own lint catches why: **the elimination fraction measures grid freedom, not
bindingness.** C1's drivers sit on a balanced 45-point grid; C4's and C6's
tolerances were single pinned numbers, and PROFILES was six named bundles rather
than a factorial. A constraint whose drivers are finely swept will always look
less binding, because the grid hands it room to be satisfied.

The readable form is a curve against each constraint's *own* tolerance:

```
  C6 over-forget tol   0.05 0.10 0.20 0.35 0.50  ->  0.78 0.62 0.43 0.22 0.11
  C4 latency days         3    7   14   30   60  ->  0.79 0.58 0.25 0.25 0.00
  C3 deletions/day     0.25 0.50 1.00 2.00 4.00  ->  0.06 0.08 0.25 0.29 0.53
  C2 probe budget      2000 4000 6000  10k  20k  ->  0.40 0.20 0.20 0.00 0.00
```

Every one spans most of [0,1]. At a single tolerance the ranking is a statement
about the tolerances, not about the design.

**And the self-criticism attached to it was wrong on its own terms.**
Feasibility-binding and correctness-relevant are different properties, and a
feasibility sweep can only see the first. What the C1 work produced was three
*soundness* findings — the budget specified in a form that does not work
(E1.1/b), protection allocated by frequency (E1.1c), and allocation writing into
still-committed directions undetectably (E1.2 U2). A configuration can satisfy
C1 comfortably while the mechanism satisfying it is wrong in all three ways.
Part III §0's warning is about where improvement *rate* is bottlenecked, not
about where specification errors live.

### What this study actually produced

Not the 9.1%. **The design's uncertainty is now localised into four quantities,
none of which has been measured:**

| Quantity | Gates | How to resolve |
|---|---|---|
| **`H`, recompile wall-clock** | the C3∧C4 window | Rig B — an afternoon |
| **support redundancy + threshold** | C6 | harvested-probe provenance, the same interaction sample as the tier audit |
| **latency tolerance `L`** | C4 | not a measurement — a product requirement to write down |
| **cascade breadth `β`** | whether any window exists at all | **R9** — a mechanism that does not exist |

Support redundancy is C6's version of `H`: a 3-entry, 2-of-3 support was pinned
in every earlier run and loses ~16% of items to the decay rate alone. Putting it
on an axis moved feasibility from 7.1% to 9.1%, and it gates the constraint with
the steepest tolerance curve.

Three of the four are cheap or free; the fourth is the architectural gap. That is
the correct shape for a feasibility study's answer, and it is a better result
than a percentage.

**Three errors corrected here, and one changed the headline.** The union was
modelled as *linear* in batch and clipped at fleet; adapters are touched
independently, so it saturates *exponentially* — at β=0.1, batch 10, the linear
form claimed the whole fleet where the truth is 65%, overstating C3's cost most
severely in the low-breadth regime where feasibility lives. The window was then
evaluated at the *saturated* union, pinning breadth at its worst value and
letting the result be reported as "independent of every other axis" — the
pessimistic-corner pattern for the third time, after E1.2's disjoint domains and
E1.1c/d's overlap 0. And the box was too small on two axes, so the earlier "~1%"
was a statement about where I looked.

*(The earlier anomaly — use-based decay appearing to beat stratified — was an
artifact of the linear union form. Corrected: stratified 3510 vs use-based 3348,
the direction the weighting rule predicts.)*

**What this does not establish.** C1 and C6 are measured; C2, C3, C4 and cascade
breadth are computed from relationships measured elsewhere. This composes prior
measurements under chosen tolerances — it is not an independent measurement, and
F3's ordering is only as good as those tolerances.

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

## 3 · Phase schedule — reordered

**Rig B is no longer a later phase. It is the calibration input every Rig A
result is conditional on, and it moves ahead of the remaining claims.**

The reason is the error record. Generator error is now the dominant error
source in this programme — B4 and B5 were both generators, and B4 manufactured
a headline that survived a full write-up. No amount of additional Rig A work
reduces it, because the generators are the unmeasured input to all of it. A
single real activation spectrum would have killed B4 on sight.

It is also the measurement that decides whether E1.1b's empty-intersection
result matters. The question is *not* which spectrum shape a transformer
produces — it is **how much shape varies across regions**, because that variance
is exactly what determines whether one ρ can serve a fleet. Four synthetic
shapes cannot answer that. One model's per-region activation spectra can.

**Phase 1 — cheap decisive (Rig A).** Complete: E1.1, E1.1b, E1.4, E0.2,
E0.2b, E0.2c, E4.2, E3.3, E2.1.

**Phase 2 — Rig B calibration (moved up, now blocking).** The smallest set that
retires generator risk and settles what Phase 1 left conditional:

| Measurement | Settles |
|---|---|
| **E1.3 — predicted variance vs realized error on held-out queries** | **first, not fifth.** The weighting rule fixes protection statistics by computing them per region; it cannot reach anisotropy *within* a region. Partitioning finer to convert within- into between-region walks straight into E3.3's atomisation result, so partitioning cannot solve it at *any* grain. E1.3 is the only instrument left that does not depend on a partition |
| **per-region subspace overlap** and the **subscription ratio** of real traffic | E1.1d makes these jointly decisive: they move the feasibility boundary from 1.0× to beyond 3.0×, and every Rig A free-rank number was taken at 1.0× |
| per-region activation feature spectra from a small MLX model | whether any Rig A spectrum stream resembles a real one — retires B4-class risk |
| the **spillover differential** — narrow adapter vs broad, paired on one probe set | E3.1b makes this the measurement that decides §B, and much cheaper than a level |
| observed correlation between checkability and difficulty on real traffic | where E2.1 sits on its own sweep — the number that decides Part III §5 |
| **recompile wall-clock `H`** — time to compile one adapter at target rank on target hardware | **the cheapest decisive item in the block.** The C3∧C4 window turns on `H = 8h`, which is an assumption, not a measurement. At 30 minutes the window opens with no architectural change — and this gates the only infeasibility currently presented as unconditional. An afternoon's work |
| E1.5 per-layer cost | Root 1's remaining claim, which needs a model anyway |

**Rig A is closed here, deliberately.** Nine rig bugs in nineteen experiments is
a stable error rate, not a converging one; every remaining Rig A number is
conditional on generators whose realism is unmeasured; and three of the four
quantities that now gate the design are resolved *outside* the simulator. The
failure mode to guard against is continuing to run Rig A because Rig A is where
the machinery is. The next two items are afternoons and each moves more than
another simulation can:

1. **`H`, recompile wall-clock** — Rig B. Gates the only infeasibility currently
   presented as unconditional. If it is 30 minutes rather than 8 hours the
   window opens with no architectural change.
2. **Support redundancy and pass threshold** — from harvested-probe provenance,
   the same interaction sample as the tier audit. Gates C6, whose tolerance
   curve is the steepest of the four.

Then **E2.3**, the last untested claim in the architecturally-consequential
class — *after* those two, not before.

**Phase 3 — remaining Rig A claims**, re-run against whatever Phase 2 says the
generators should look like: E1.2, E3.1, E3.2, E3.4, E4.1, E4.3, E4.4, E4.5,
E0.1, E0.3, E0.4, E2.3, E2.4, E2.5.

**Phase 4 — adversarial repair validation.** Not rematches — worlds built to
kill each repair, from outside the model class it assumes (see §5).

**Phase 5 — Rig C.** Survivors only.

**Bootstrap order note.** The design's own Part I §11 sequence is a *build*
order and a good one. This is a *test* order and it deliberately runs the other
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

**A constraint that never binds and a constraint that always binds are equally
uninformative, and both are more likely a pinned parameter than a property.**
This generalises the manipulation check from a per-*arm* question to a
per-*constraint* one, and it is now the most productive single question in the
programme — it found B7, B8, and both of B9's near-publications. The standing
instrument is the per-constraint attribution table (E5.1 F3): eliminated, and
eliminated *alone*. A constraint at 100% or 0% is a bug report about the sweep
until proven otherwise.

**Verify the manipulation took before interpreting a comparison.** The lint below
checks a criterion's logical *form*. B7 passed that check and still failed: U1
*could* have returned the other answer in principle, but in that run no arm ever
had free rank, so the arms never differed on the quantity being varied and U1
"passed" on a comparison that never happened. Assert that the arms actually
differ on the manipulated quantity — in B7's case, `free_rank > 0` for at least
one arm. This is a standard manipulation check and it is the third distinct
instance here of *the run did not exercise the criterion*.

**Every mutation and every criterion must have a postcondition that some
possible input would violate.** Three failures in this programme are the same
defect — *an operation that cannot report failure*:

| Where | The operation | Why it could not fail |
|---|---|---|
| kill criterion | `true_influencers − provenance` | both sides were the same loop body, so the difference was `∅` for every world |
| code | E1.1b's `safe` flag *(as first drafted)* | leakage on the calibration sample is `1−ρ` by construction |
| tooling | `str.replace` patching `claims.yaml` | fails open — a missing anchor silently does nothing |

Hit in a criterion, in code, and in tooling, which is decent evidence it is the
general one.

**E3.1b forces a stronger form.** The comp-only figure *did* vary — it passed
the lint's letter — but it varied with pool emission order rather than with the
ranking rule it was cited as evidence for. So the lint is:

> For every mutation, assert it changed something. For every criterion, name the
> input that would flip it — **and check that nothing irrelevant flips it more.**

The cheap version of the second clause is a nuisance sweep: vary something that
*should not* matter (tiebreak order, emission order, seed, iteration order) and
confirm the metric is stable. Had that run on E3.1, the 55% would never have
been published.

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

- **B5** the card-overlap knob in `influence.py` shrank the pool of entries
  feeding any card, instead of raising how many cards each entry feeds — so
  cascade breadth moved the *wrong way* and would have reported the L3/L7
  coupling with an inverted sign. Caught because the direction was implausible.
- **B6** *(process, not code)* three `str.replace` calls patching
  `claims/claims.yaml` anchored on text that was not there, and silently did
  nothing — R6, R7 and R8 were described in this document for two commits while
  missing from the machine-readable ledger. Python's `str.replace` fails open.
  Every scripted patch now asserts its anchor exists.

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

### What the results say collectively — and what the record cannot support

**The "every failure is a specification defect" line was selection, not
evidence, and it is withdrawn.** The first eight experiments all asked *local
mechanism* questions — is this formula right (E1.1, E1.4), is this list complete
(E4.2), is this objective right (E3.3), does this provenance set cover the paths
(E0.2), does this filter yield enough (E2.1). Local mechanism errors are
*by construction* locally repairable, so the record was close to tautological
given which claims were chosen. Cheap-to-test correlated with
locally-repairable throughout.

The claims that can fail architecturally are the ones where failure has no local
repair. Two have now run:

- **E3.1** (net transfer accumulates abstractions) — passes conditionally, on a
  condition the design does not state.
- **E0.2d** (admission control bounds cascade breadth) — fails, and the repair
  is a mechanism that does not exist rather than a threshold that needs moving.

**E1.2** and **E0.1** have now both run and both fail. E0.1 is the heaviest:
I4 fails on over-forgetting with a **12.4× blindness factor**, the ontology turns
out to be load-bearing for adapter competence, and I4 is revealed as a *relative*
invariant that cannot see competence never compiled in the first place. **One
remains untested: E2.3** (whether the cheap rung preserves the ranking the
expensive rung would give).

So the accurate summary is: *most* failures found are specification defects,
one is a missing control surface, one central claim holds conditionally, and
**three of the five architecturally-consequential claims have not been tested.**

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

**E0.2b's ceiling was demoted, and E0.2c's replacement was then itself
corrected.** E0.2c takes the ceiling apart correctly:

- **Correctness is not rate-limited at all.** Disabling an affected adapter is
  ~28,800× cheaper than recompiling it, and disabling is what makes deletion
  sound — recompilation restores *competence*. E0.2b reported a service-quality
  limit as a correctness limit.
- **Batching makes throughput independent of cascade breadth.** Per-deletion
  cost is `|union|/b × recompile` and `|union|` saturates at fleet size, so at
  window 16 every breadth gives 24 deletions/day against 1.57–2.53 eager.
- **Breadth is a knob the design already holds.** Cascade breadth rises
  monotonically with card-bank duplication, 63.1% at one card per entry to
  98.8% at six.

But E0.2c's own conclusion — that L3's admission threshold is the lever on
breadth — was a **common cause, not an intervention**. It varied entry
multiplicity and watched card cosine and cascade breadth rise together. E0.2d
separates them by holding provenance fixed and moving content cosine alone:

```
Panel B - provenance held at overlap 0.307, vary content rotation
      rotation  prov overlap  card cosine   cascade
          0.00         0.307        0.519     97.0%
          0.50         0.307        0.205     96.9%
          1.00         0.307        0.033     96.7%
                                    ^^^^^      ^^^^
                              moves 0.498   moves 0.009
```

**Content cosine is not a lever on cascade breadth.** SESA's cos ≤ 0.93 is a
redundancy filter on card *content*; breadth is set by *provenance overlap* —
how many cards share a source entry. E0.2c's world coupled them only because
card content was built as the mean of its source entries. In a real bank they
come apart: two cards can capture different patterns in the same entries and be
near-orthogonal in content while sharing every source.

**This is a missing control surface, and it does not reduce to a spec defect.**
The design has no stated lever on cascade breadth at all. The repair is not
retuning 0.93 — it is provenance-aware admission, capping source-entry overlap
between admitted cards, which is a new mechanism. A missing control surface
cannot be fixed by rewording. **This is the closest thing the programme has
produced to an architectural finding.**

### E3.1 · Net-transfer ranking works — under a condition the design never states

The first claim tested here whose failure would have had no local repair.

```
regime                        target   transfer   margin
clean                          0.264      0.349    +0.085
noisy probes                   0.176      0.277    +0.101
noisy + spurious               0.071      0.219    +0.147   (173% of clean)
patches with REAL spillover    0.021      0.013    -0.008   <- T4 fails
```

T1–T3 pass. The margin *grows* under measurement noise, which I did not predict:
target ranking takes a max over noisy per-region deltas, net transfer takes a
sum, and summing is variance-reducing.

**Quote the noisy rows, not the clean one.** E3.1b shows the clean regime is
tiebreak-contaminated — transfer's compositional score there spans 0.053 to
0.582 depending on how exact ties are resolved. The tied set exists only because
a noiseless world produces exact ties. In the noisy regimes spread is 0.000
across all four tiebreak rules, so the +0.147 margin is artifact-free while the
+0.085 one is not.

**But the mechanism is not the one Part II §B describes.** Net transfer does not
*detect* generality — it removes the reward for *narrowness*. The argument is
**analytic and needs no number**: a first-order statistic over per-region deltas
cannot distinguish a candidate whose per-region deltas are all zero from one
already applied, and compositional-only skills are exactly that case.

> **Withdrawn.** An earlier version of this section cited "transfer acquires
> compositional-only skills at 55% while learning 72% overall" as evidence.
> E3.1b shows that figure measures pool emission order, not the ranking rule —
> see below. It has been struck from all documents.

**Which means §B is conditionally true, and the condition is unstated:** narrow
patches must have near-zero *real* off-target effect. Give patches a systematic
+0.02 spillover and the margin inverts and both arms collapse — every candidate
looks good on net transfer, the signal saturates, and the gate stops
discriminating at all (3.6–4.2 skills learned against 9–10). Whether real
adapters have that spillover is empirical and not obviously favourable: a
low-rank update fit to one region is not region-confined. **Measuring real patch
spillover decides whether the architecture's central generalization claim holds
in practice**, and it is a Rig B/C measurement.

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
