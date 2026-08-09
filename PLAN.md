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
| E0.2e | **R9** — provenance-aware admission has an acceptable point | **FAIL** — the curve cannot be drawn as specified |
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
| E2.3 | the staged ladder makes an archive affordable | **FAIL** on both — and not for the predicted reason |
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

### E0.2e · R9 cannot be evaluated as specified

The last architectural gap. E0.2d found no lever on cascade breadth; E5.1 made
breadth the quantity deciding whether an operating envelope exists. R9 —
provenance-aware admission — is the only proposed repair, and it does not exist
yet. So it was specified here as a **curve** with an acceptable region marked,
because every mechanism in this design specified without stating its tension has
been found out by an experiment.

Tightening the cap does two things at once, and they push **opposite ways at
every pool structure swept**:

```
        conc   cards/entry   fleet reach   directions
         0.2         +4.60        -0.169   OPPOSED
         0.5         +5.80        -0.286   OPPOSED
         0.9         +8.18        -0.265   OPPOSED
```

Cards-per-entry falls 9.18 → 1.00 — **the direct mechanism works**. But a tighter
cap also *shrinks the bank*, and each adapter still draws the same number of
rollouts, so a surviving card is used by more adapters and fleet reach *rises*.

Which dominates depends on how the fleet composes from the bank as the bank
shrinks, and **Parts I–III specify none of it.** Three things must be stated
before R9 can be designed: how many distinct cards an adapter's draw uses,
whether that is absolute or a fraction of the bank, and whether fleet size tracks
bank size.

**And the expected tension is not the binding one.** R9 was predicted to trade
against I11 — reject a card and its distinction is never available. Coverage stays
at **1.000** across every tau and pool structure swept, so that tension does not
bind here. The blocking unknown is the fleet coupling.

So the result is neither "R9 works" nor "R9 has no acceptable point." It is that
**R9 arrives with the same defect the repairs were meant to fix** — proposed
without stating the tension it trades against. Choosing a coupling to obtain a
number would have been choosing the answer, which is why G1 was re-registered
from "is there an acceptable point" to "can the curve be drawn at all."

---

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
Parts I–III against it finds **twelve protection sites**, four measured and
eight inferred — the count is derived from the table by
[tools/check_record.py](tools/check_record.py), not typed beside it — written up in
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
competence by 21.7% pooled, 41.6% worst. **[WITHDRAWN — see the K4 differencing below.]** The ontology *is* load-bearing for
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
produces 21.6% pooled over-forgetting **[WITHDRAWN — B20; rebuilt at 8.1 points over the floor]**, so without pinning components E0.1 would
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

### E2.3 · The ladder's filter is the wrong shape, and it isn't a weighting problem

The last untested claim in the architecturally-consequential class. Its failure
mode is structurally invisible: a candidate rejected at the cheap rung is never
measured on the full one, so the archive reports what it promoted and never what
it discarded.

**L1 fails, on a re-registered criterion.** The original — Spearman(cheap, *true*)
— compared the cheap rung against truth rather than against the **full rung**,
so a broken gate objective would be charged to the ladder. And rank correlation
is the wrong shape for a filter: a ladder's job is **recall**, not ranking.
Re-registered as recall of the full rung's top decile, which falls to **0.349** —
the ladder discards two thirds of what the expensive rung would have ranked top.

*The obvious worry about the original criterion does not materialise.* If the
full rung ranked by the same mean, ρ(cheap, full) would stay high while
ρ(cheap, true) collapsed — the ladder working and the **gate** being broken,
collapsing E2.3's two findings into one. Measured, they track: mean gap +0.026,
and ρ(cheap, full) falls to 0.322 alongside ρ(cheap, true) at 0.361. So E2.3
retains an independent result.

**L2 fails: ~100% of good rare specialists are dropped at the cheap rung against
~36% of generalists.** A 2.81× concentration.

**And my prediction about *why* was wrong.** I expected the concentration to come
from a traffic-weighted cheap rung — the weighting rule arriving at the Assay.
It doesn't. `ladder_random`, the unbiased control with uniform probe allocation,
concentrates at **exactly the same 2.81×**. The docstring pre-registered that
contingency ("if BOTH arms concentrate, the effect is the ladder itself rather
than the weighting, and the repair is different"), which is the only reason the
wrong prediction didn't become the finding.

The real mechanism is simpler and worse:

```
  generalist      +0.04 in 16 of 16  ->  cheap score 0.0400
  rare specialist +0.30 in  1 of 16  ->  cheap score 0.0187
```

A cheap rung ranks by a **mean over covered regions**, so a specialist loses by
more than half on *any* suite — uniformly sampled or traffic-weighted, noisy or
exact. Probe allocation never enters the comparison. **Ranking by a mean discards
specialists structurally**, so this is not a weighting-rule site and the
weighting rule does not reach it.

**Why the objective had to be named first.** Under the design's gate as written
— unweighted mean of per-region deltas — dropping a rare specialist is *correct*,
not a defect: it is worth half a generalist. The loss is only a defect under I11's
coverage objective, where a specialist lifting a region above the competence
floor is exactly what closes the gap. **The design never says which objective
promotion serves**, and until it does, "the ladder loses good candidates" is not
even well posed. That is the finding underneath the finding.

*Separately, and not about the ladder:* regret under the coverage objective is
near-constant across every ladder setting, because the final pick is made on the
full suite's **mean** score. That regret is a property of the gate's objective,
not of the ladder, and is reported so it is not mistaken for one.

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
| ~~**`H`, recompile wall-clock**~~ | the C3∧C4 window | **RESOLVED — EB.1.** 18 min at the established draw cap, not 8h. The window is **OPEN** at the measured value |
| **support redundancy + threshold** | C6 | **procedure written** ([docs](docs/support_redundancy_procedure.md)) with a decision rule attached ([E5.2](rig_a/experiments/e5_2_support_decision_table.py)): if real probes rest on **≥5 supporting entries, C6 stops binding**. Awaiting an interaction corpus — not measurable here |
| **latency tolerance `L`** | C4 | not a measurement — a product requirement to write down |
| **cascade breadth `β`** | whether any window exists at all | **R9** — a mechanism that does not exist |

Support redundancy is C6's version of `H`: a 3-entry, 2-of-3 support was pinned
in every earlier run and loses ~16% of items to the decay rate alone. Putting it
on an axis moved feasibility from 7.1% to 9.1%, and it gates the constraint with
the steepest tolerance curve.

**One of the four is now resolved** — `H`, by EB.1, and it was the one carrying
the infeasibility verdict. Of the remaining three, one needs interaction data
this project does not yet have, one is a product requirement rather than a
measurement, and the third is the architectural gap. That is the correct shape
for a feasibility study's answer, and a better result than a percentage.

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

### EB.1 · `H` measured — and it was carrying E5.1's infeasibility verdict

The first Rig B measurement. Part I §7's substrate table gives L7 adapters a
single timescale, "hours–days", and that sentence was the only thing standing
behind `H = 8h` in E5.1 — where the entire C3∧C4 window turns on it.

**`H` is not a constant.** It is `draw × tokens_per_entry × epochs / throughput`,
and Parts I–III state none of the three. Measured on an M2 with MLX, Qwen2.5 at
4-bit, LoRA over 8 layers:

```
  model                 tok/s     100     300    1000    3000   draw for 8h
  Qwen2.5-0.5B-4bit       973   0.04h   0.13h   0.44h   1.32h        18,237
  Qwen2.5-1.5B-4bit       417   0.10h   0.31h   1.02h   3.07h         7,826
```

**At E0.1's established draw cap of 300 entries, `H` is 18 minutes at 1.5B — not
8 hours.** Reaching 8 hours needs ~7,800 drawn entries, more than an order of
magnitude past the cap. And the consequence for E5.1 is direct:

```
  assumed  H = 8h      drain 5.33d   window [5.33, 3.33]   EMPTY
  measured H = 0.31h   drain 0.20d   window [0.20, 13.59]   OPEN
```

**E5.1's infeasibility at the design's own profile was carried entirely by the
assumed `H`.** One of the four gating quantities is resolved, and it was the one
carrying the verdict.

**H2 passes decisively:** rank 4→32 moves wall-clock +4.5% and +2.2%. Rank is a
*budget* quantity, not a *time* quantity, so the subspace budget and `H` are not
coupled and E5.1 needs no extra edge.

**H1 passes within the memory envelope** (+9.9% per-token drift across 128–512),
so capping the draw does bound `H` and E0.1's A6 resolution holds — but getting
there required correcting my own verdict. See §4a, **B12**.

**Scope, stated rather than laundered.** An M2 at 1.5B is not production
hardware and these seconds are not production `H`. What transfers is the
structure — `H` scales in drawn tokens, is bounded by the cap, and is
insensitive to rank — plus a throughput anchor a production run can be checked
against. Whether production `H` is 19 minutes or 8 hours depends on model size
and hardware the design never states, **which is itself the finding**: "hours–
days" was never a checkable claim.

*Rig limit worth recording:* a memory cliff at seq_len 1024 (1.5B) and 2048
(0.5B), where per-token cost jumps 17×. It scales with model size, so it is
paging on 8 GB rather than quadratic attention. That bounds what Rig B can
measure here.

---

### E0.5 · The margin certificate is sound, narrow, and not a subset

Rev 2 §2's answer to E0.2b's residual is that you never need the counterfactual —
whether a selection *would have flipped* is decidable from scores the system
already computed. Three claims, three different answers, and collapsing them into
one verdict would destroy the distinction a reader needs.

**Sound, and that is the half that carries.** No certified selection ever flipped,
across 2 arms × 5 ledger sizes × 3 batch sizes × 8 worlds × 6 draws. Soundness is
a property of the algebra and the implementation, not of the traffic, so unlike
the certified *fraction* it transfers out of the rig. The audit performs the
deletion for real rather than trusting the derivation, because a bound that is
sound on paper and mis-implemented in code is exactly B3's shape.

**Narrow.** Certification reaches 0.81 at single tombstones with well-supported
cards and collapses to ~0.00 at **every batch size ≥ 4**, while the world stays
stable at 0.60–0.95. The gap is bound slack, not fragility — which matters,
because slack is an instrument problem and fragility is a world problem, and they
call for opposite responses.

**Two drivers, one predicted.** `k` is the bound's denominator, so support per
card was pre-registered as the governing variable. It is — and so is a second one
of comparable size. Holding touched-fraction pinned, k from 3 → 60 buys **+0.457**;
holding k pinned, touched-fraction from 0.33 → 0.02 buys **+0.419**. The control
arm is the only reason the second one is visible. The cross-arm comparison at
fixed ledger reads +0.102 and is the wrong number to quote, because k rises 7.5×
while touched-fraction rises 6× and the figure nets two opposing effects — the
same defect as B19, caught this time before it was published.

**Not a subset, and this is the finding.** §2 claims uncertified selections are a
strict subset of transitive-closure edges, so the journal shrinks the cascade.
False: 24,598 uncertified selections lie outside the closure and **2,178 of them —
8.9% — actually flipped.** Those moved because a *rival* card rose while the
chosen card was untouched. Provenance records the card that was selected; it
cannot record that a competitor gained. So the journal **enlarges** the cascade,
and part of the enlargement is real influence no set-based closure can see. That
is E0.2b's missing recall approached from the other side, and it is a better
result than the one claimed — just not the one claimed.

**A tension neither document states.** E0.2c's deletion economics rest on
batching: window 16 is what makes throughput independent of cascade breadth at 24
deletions/day. Batch 16 is also where certification is 0.000 in every
configuration tested. **The mechanism that makes deletion affordable is the
mechanism that destroys the certificate.** §2 and R8 cannot both be adopted at
their stated settings.

### E0.6 · The cover is not total, and rev 2 understates its own repair

R-c says the fleet residual set must hold probes for *all* provenance no live
owner covers, not only what retirement orphaned. This is in Phase 1 rather than
Phase 3 for a reason that is structural: every other gap in this record announced
itself with a number going the wrong way, and this one cannot. An unowned surface
produces **no statistic at all**, so a register shipping without R-c would report
only its wins.

**R-c is not a refinement of R-b — it is most of the job.** Never-owned provenance
exceeds retirement-orphaned provenance by **3.3× to 5.5×** at every traffic level.
R-b alone ships a cover with a hole in it.

**The surface is load-bearing, and thinly.** Unowned entries do move adapter
weights — 12–41% move at least one — but at 0.12 adapters each against 7.71 for
covered entries. The honest claim is not "a third of competence is undefended"; it
is that a real, small, structurally invisible surface exists.

**The finding under the finding.** Unowned provenance runs **0.483 at 16 rollouts
to 0.027 at 512**. The cover's *reach* is set by traffic even though every
statistic computed inside it is an equal draw. I8 holds within an owner and says
nothing about how far the owners collectively reach — so a register that wins on
tail safety wins over whatever surface traffic happened to buy, and the gap is
largest exactly where traffic is thin, which is the tail the register exists to
protect. That is not an argument against the register. It is why `unowned
fraction` belongs on the dashboard beside the win.

**And the new metric has the old defect.** At card grain the worst cell runs to
**1.000 against a pooled 0.483** — blindness 2.1, peaking at 4.2 — with 37.5% of
cards having every source unowned at low traffic. A pooled protection number
hiding a concentrated hole — the same defect as E0.1 KB, E1.1c and E2.3, and the
first time it appears inside the statistic proposed to fix the problem. Reported worst-cell alongside pooled, in E0.1 KB's form.

**The tail statistic has its own blind spot, stated rather than discovered.** An
entry in no card's source list has no cell to sit in, so `worst card` reads 0.000
at high traffic while 6.7% of entries are in no card at all — and those are the
hardest case, since no card cites them and no owner's probe draw reaches them by
any route. It gets its own column.

### E0.7 · The oracle line, and a statistic that was measuring its denominator

**The number this experiment first reported is withdrawn.** Run 1 gave a "77%
oracle saving" as `1 − distinct/(fleet × probes)`. Back the pool out of its own
two numbers — `N(1 − e^{−2048/N}) = 473` solves at **N ≈ 480** — and 98.5% of the
eligible pool was already consumed. The numerator was pinned at its ceiling while
the denominator grew with fleet, so the saving rises monotonically forever and its
argmax is unbounded granularity. That is E3.3's degeneracy exactly: monotone in
the wrong direction, with an argmax nobody would accept if it were stated as a
recommendation.

| | | |
|---|---|---|
| `1 − distinct/(fleet × probes)` | **withdrawn** | monotone in fleet; best where the pool is most exhausted |
| `distinct / \|pool\|` | consumption | saturates at 1, cannot be gamed by adding owners |
| `distinct × (1 − yield)` | the scarce line | what is actually authored rather than harvested |

**The governing quantity is `fleet × probes` against `|pool|`, and the crossover
is now in range.**

| pool | distinct | of draws | pool consumed |
|---|---|---|---|
| 240 | 209 | 40.8% | 86.9% |
| 4102 | 497 | **97.1%** | 12.1% |

Below the crossover the line is flat and a new owner authors almost nothing.
Above it, **essentially every owner pays**. Run 1 sat above the crossover at every
point it sampled, which is why every point looked like a win — and a real ledger
sits far above it, in the regime where granularity is oracle-*expensive*.

**KN still fails.** The distinct count sits on the birthday curve at every pool
size, largest departure 3.2%. The mechanism §1.3 gives — a new owner pays only for
provenance not already covered — is unsupported at any pool size. Overlap was
never going to be the governing variable.

**What rescues the verdict is harvest yield, and E2.1 already measured it.** A
harvested T0/T1 probe carries a verified outcome that came free with the
interaction; only the adjudicated remainder is scarce. So the oracle line is
`distinct × (1 − yield)`, and E2.1's strict filter yields **0.68 falling to 0.33**
as checkability–difficulty correlation rises — a ~2× move in the oracle bill,
larger than pool size does over most of this sweep and far larger than provenance
overlap does at all. If probe sets are harvest-first, §1.3 survives for a reason
neither the document nor this record had tested, and worklist 2.3 needs no new
curve.

**And what a probe *is* resolves as a pair, not a key.** A probe is a stimulus
plus an expectation. The stimulus is oracle-priced and entry-keyed, so it shares;
the expectation is *whose floor does this count against*, which is bookkeeping and
free. That reconciles the two measurements rather than choosing between them —
sharing is real on the oracle line and absent on the compute line, and both are
correct. The falsifiable condition: **the expectation must be derivable from the
stimulus.** True for ground-truth-correct outcomes, false where a probe tests an
owner-specific target behaviour, because then the expectation is itself
oracle-priced — and at `owner_specific = 1.0` distinct probes equal draws at every
pool size. What restricting to ground-truth-keyed stimuli costs is **coverage**,
and that is unmeasured.

### E0.5 revisited · The batch-16 tension was mine

E0.5 asked *"what fraction certifies at batch b"* and read 0.000 at b ≥ 16 as a
conflict with E0.2c's window-16 economics. That question conflates two operations
that sit on different clocks: **disabling is per tombstone and is what makes
deletion sound; recompiling is what gets batched.**

| | |
|---|---|
| disable load, per tombstone | **0.138** |
| recompile queue, batch reading | 0.929 |
| recompile queue, per-tombstone reading | 0.821 |
| actually flipped over the window | 0.114 |

The certificate is needed on the fast path, where it costs 13.8%. **Running
certification is sound and, at window close, algebraically the same set as the
batch reading** — the bound sums `src_dev` over hit sources and divides by
`(k − m)` whether accumulated in one step or sixteen. So it does not improve the
number; it changes *when* the answer is needed, which is the point.

Two things not to overclaim. The cheap per-tombstone reading produced **no**
violation here, and that is not soundness — it is provably unsound as a batch
guarantee (`e1` alone cannot flip a selection, `e2` alone cannot, both together
can), so zero in six worlds is a fact about these worlds. And the honest cost:
**92.9% of selections are uncertified over the window while 11.4% actually
flipped**, so the certificate is nearly useless for pruning the recompile queue.
It earns its place on the disable path and nowhere else — narrower than §2 claims.

### E0.8 · Depth governs coverage and not the oracle line — they are two curves

E0.6 and E0.7 looked like one quantity seen from two ends, both set by traffic
depth per entry. Holding ledger and fleet fixed so depth is the only thing moving:

| depth | pool | unowned | distinct | consumed | /draws |
|---|---|---|---|---|---|
| 0.05 | 388 | 0.692 | 227 | 58.7% | 88.6% |
| 1.00 | 960 | 0.000 | 226 | 23.6% | 88.3% |
| 4.00 | 960 | 0.000 | 225 | 23.4% | 88.0% |

**KD fails.** Unowned provenance collapses to zero by depth 1.0 while distinct
probes are **flat at ~225 across the entire range** and sharing never appears at
all. Consumption *does* fall with depth — because its **denominator** grows, the
pool running 388 → 960 while the numerator is pinned by draws. Reading
consumption alone would have shown a curve moving with depth and concealed that
the oracle cost never moved. Same defect as the withdrawn 77%, a ratio whose
denominator is doing the work; caught this time only because the absolute count
was carried alongside the ratio.

**So the corollary dies, and the corollary was the interesting part.** *"Both
currencies degrade together in the tail"* is not what happens: coverage degrades
in the tail and the oracle bill does not move. If anything the tail is where
sharing would *begin* to help, because that is where the pool shrinks toward the
draw count — the opposite direction from the prediction. Reaching `pool < draws`
needs fewer than one rollout per adapter here (B18), so that regime is recorded
as unmeasured rather than asserted either way.

What survives for worklist 2.3: not the unification. Overlap is not the governing
variable for the oracle line (E0.7) and neither is depth. The governing quantity
is `fleet × probes` against the pool, which is arithmetic. Depth governs coverage,
which is E0.6's result standing alone. **Two axes, two quantities.**

### E0.1 revisited · K4 is withdrawn, and A3 varied nothing

Worklist 2.4 and 2.5. Two of this record's published numbers do not survive their
own null rows, and one incoming correction did not survive being checked.

**K4 passes. The 21.7% was the draw policy.** K4 was scored on `identical`, which
is `False` for *any* stochastic arm regardless of what the ontology does — so it
scored the stratified draw, not the re-partition. Against a null-treatment row
under the same policy, the ontology's effect is **−0.0118 pooled and −0.0069
worst, against a null seed spread of 0.0218**: indistinguishable from resampling,
and negative besides.

So *"the signature ontology is not load-bearing for adapter competence"* is
**restored**, Root 3 does not reach into Root 1 by this path, and rev 2 §1.2's
conditional retention of the ontology for routing and curriculum has its
condition satisfied — *"a wrong partition costs efficiency rather than
competence"* survives. Worklist 2.4's kill criterion was written to fire if K4
survived differencing. It did not.

**A3 varied nothing at all (B20).** `arm()` passed one `policy` to both
`w.compile()` calls, so A3 — `policy="uniform"` and no other treatment —
recompiled under the same policy it compiled under. Nothing about the harness
differed; the rng advanced, the second stochastic draw came out different, and
the 21.6% was the uniform **resampling floor** wearing a treatment's name. That is
stronger than the review's *"A3 may be resampling"*: it is resampling by
construction, because no harness component can differ across two calls receiving
the same argument.

Rebuilt with a `recompile_policy` lever — compile under `usage`, recompile under
`uniform` — A3 gives **0.2969 against the uniform null's 0.2160**: an 8.1-point
effect at a null spread of 0.018, roughly 4.5σ. Component-granular stamping keeps
measured support, at about a third of the size once claimed.

**B19 is withdrawn, and how it got in matters more than the number.** I recorded
that the stratified-vs-usage tradeoff crossed a decay difference. Those two
numbers are A6-stratified and A6-**uniform**; both carry `decay=0.25` and differ
only in draw policy, so the comparison is matched and citable. What is true is
narrower — A6 is not a matched control for *A4*, which bears on K4's differencing
and not on the tradeoff at all.

I took the review's sentence, which conflated the two, and recorded it as a bug in
my own rig without checking which arms produced the numbers. The results file and
the code were both to hand, and a caveat sat next to a sound number for two
commits. **An incoming correction is a hypothesis about the record, not an
observation of it** — the same discipline the record applies to its own results,
applied to criticism of them. Second wrong attribution in this record, after B11.

Going to check whether B19 was true is what surfaced B20. A false bug report led
to a real one.

**And the nulls are not a universal comparator.** Both A6 arms carry decay, which
shrinks provenance 600 → 450 against a fixed 300-entry cap, so their recompile
draw covers a *larger* fraction of a smaller live set than the null's does.
A6-uniform lands **−0.0511** against its null and A6-stratified **+0.0170** —
differences that measure the draw fraction, not the policy. The within-A6
comparison is the valid one, because decay is held equal across it. A null row is
matched to a policy, not to every arm that uses that policy.

### E1.6 · The hinge, ratio half — §1's central claim survives

Rev 2 registered the hinge as one experiment with one kill criterion, and it
bundles two quantities with different sensitivities. **The ratio** — blindness,
and the ordering of per-domain exposure — comes from *traffic weighting*, so it
is readable at any subscription. **The level** — free rank, and whether the
register starves — is exactly the subscription-sensitive number E1.1d showed
cannot be read at 1.0×. The half that decides whether §1's thesis is right runs
today; only the half that decides whether it *fits* waits on Rig B. Precedent is
this record's own: E1.2's finding was the ratio, not the level.

At ρ = 0.95, over 3 configurations × 4 seeds:

| | spectrum | register | R-b freq-balanced |
|---|---|---|---|
| blindness (worst / traffic-weighted) | 2.88 | **1.33** | 1.45 |
| Spearman(rate, leakage) | **−0.971** | **−0.056** | −0.243 |

**The rarity monotone is broken, not shifted.** Spearman moves from essentially
perfect anti-correlation to zero, and the exposure *ordering* correlation between
the two arms is **+0.045** — the register protects a different set of domains, not
the same set with nicer numbers, which is what H3 was there to rule out. At
ρ = 0.99 the spectrum leaves 12.8 domains above the interference limit and the
register leaves **0.0**.

**The architectural point is not the number.** R-b lands close to the register
(1.45 against 1.33), and that agreement *is* the finding rather than a redundancy:
**R-b must estimate the rate vector it corrects for** — the traffic distribution
the weighting rule says protection must not depend on — while the register needs
only each owner's own provenance, which I1/I4 already require it to carry. Same
protection, no estimate, and the register is slightly the better of the two.

Free rank is printed and **deliberately not scored**. This decides whether the
register is a better protection *instrument*, not whether it *fits*.

### E0.5 · Why the uncertified set is uncertified — and it names one lever

Decomposing the 92.9% over a 16-tombstone window: **75.9% is rival rise alone** —
the chosen card was not even touched and a rival's upper bound rose to meet it —
24.1% is both, and **0.0% is chosen-card magnitude alone**. So tightening the
chosen side with the journal's recorded signed contributions buys ~nothing; the
lever is recording a wider candidate set past top-k. Consistent with KC, since
rival rise is exactly what a closure over selected artifacts cannot see.

**And this rig flatters the mechanism.** Retrieval here is argmax over *all* cards,
so every candidate's score is recorded. A real top-k journal cannot bound a rival
sitting outside k — it needs the (k+1)-th score as an entry threshold — so these
are the best-case numbers.

### E0.8 · Equal count is not equal coverage — the third level of one anisotropy

`probes_per_owner` is a fixed **count** while owner provenance grows with depth,
so coverage — probes drawn over that owner's provenance — falls **0.500 → 0.009**
across the sweep, a 55× fall.

The between-owner spread is only **1.7×** under uniform rollouts, and that is a
property of the world rather than a finding: equal rollouts give every owner the
same provenance size. Under **Zipfian experience per owner**, same totals and
nothing else changed, the spread reaches **37.6×**.

So an "equal-N per owner" draw is equal in count and unequal in coverage by up to
38× once owners differ in experience — while every owner is still reported as
having had its equal draw. **I8 is satisfied and the tail inside an owner is not
protected.** Between-region is fixed by I8, between-owner was found by E0.6,
within-owner is this and it is open.

### E2.2 · Verifier synthesis by hand — soundness decomposes, value does not

The design's own declared highest-leverage item, done as Part III §9 and rev 2 §11
both specify: by hand, one T3 domain, no machinery. **L3 skill-card admission** —
*is this card a faithful, useful distillation of its sources?*

**The decomposition exists.** Eight constraint-wise checks, each executable,
deterministic and individually falsifiable, and all of them the design's own
invariants restated as predicates: provenance grounding, no contradiction,
citation validity, content-cosine admission, fan-out compliance, I4
recompilability, schema validity, I6 refutation consistency.

**What no check reaches:** usefulness, salience, abstraction-vs-average,
calibration. Every reachable check answers *is this card true, grounded,
well-formed, permitted*; every unreachable one answers *is it worth having*. That
is E2.3's boundary arriving from the other side — the decomposable half is the
constraint half, and the non-decomposable half is **the objective E2.3 already
found the design never states.**

**So `A` has a computable ceiling, and §4.2 assumed it did not.** A candidate
failing a screen costs zero labels; one passing still needs a label for the value
judgment. With `r` the sound-screen rejection rate, `A = 1/(1−r)` — and `r` is
bounded by the fraction of candidates that are *unsound*, not the fraction that are
*bad*, because a sound screen cannot reject something true, grounded, well-formed
and worthless.

**Which gives an uncomfortable dynamic.** A better generator emits fewer unsound
candidates, so `r` falls and `A → 1`. **L10's amplification is largest when the
system is worst.** §4.2's "categorical gain" is withdrawn: L10 does not change the
exponent, it applies a factor that shrinks as it succeeds.

**And §4 and §6 disagree about what an instrument may be.** The escape is for an
instrument to score *quality*, which reaches the value half — but §6 forbids
exactly that for a cheap rung, and E2.3 measured the cost of estimating instead
(recall 0.349, ρ 0.322). Either L10's instruments are screens (`A` bounded,
self-limiting) or estimates (E2.3's recall problem one level up). Neither document
connects them, because §6 was written about the promotion ladder and applies
verbatim to the verifier field — a synthesised instrument *is* a cheap rung under
another name.

**What this makes runnable:** the registered kill criterion needs labels and stays
blocked, but §4's viability now turns on `r`, which needs **no labels and no L10**.
Every check above is executable against any generator's output today.

D3 was performed — eight constructed bads, each rejected by its intended check and
no other — and is **not counted as evidence**: a decomposition tested against cases
its own author constructed is generator and evaluator sharing weights, one level
up, which is the failure mode §4.1 exists to prevent.

### E1.7 · Churn under R-a…R-d — the rules hold, and the bill is mild

**KM passes with 0 violations across 600 cycles, and it is the least interesting
result here.** R-b holds by construction under these operations — merge unions,
retire moves, promotion adds — so zero violations confirms the implementation
matches the rules rather than confirming the rules. Worth the four lines because
this record has been wrong three times about what holds by construction.

**KR fails, consistent with E0.6.** The residual is **52.5%** of everything defended
at the end. R-c is not a refinement of R-b, it is the larger half — which is what
E0.6 measured at 3.3–5.5× from the other direction.

**The bill, which §1.2b names and does not price.** Growth is **linear, not
accelerating**: 4.17 evals/cycle over the first half against 3.58 over the second,
ratio 0.86. Unions and the residual each add a bounded amount per operation and the
operation rate is fixed, so the bill accumulates rather than compounds. Cycles to
reach a budget, extrapolated linearly and marked as extrapolation: 1000 at ~223,
4000 at ~997, 16000 at ~4094. **The brake on merge chains is real and mild in this
regime.** The budget is a *decide* quantity, so a range is swept rather than one
figure passed or failed.

**And a claim I asserted and had to withdraw in the same run.** The first version
of the output said the oracle and compute lines *"diverge by roughly an order of
magnitude"*. They sit at 480 against 598 — a ratio of **1.25**. Caught by the
method commitment added one turn earlier: a structural relationship between two
quantities is a measurement, not an argument. The separation §1.3 needs is real and
small at this fleet size, and would widen with the merge rate, which is untested.

### EB.2 · The hinge survives real geometry — and two of the generator's assumptions do not

Phase 4.1, registered since PLAN was written as *"per-region activation spectra
from a small MLX model (retires B4-class risk)"*. It mattered more than when it
was registered, because E1.6 had just used the generator it calibrates. Every
spectrum result in this record — E1.1, E1.1b, E1.1c, E1.1d, E1.6 — runs on
`DomainMixture`, which asserts a `k^-0.5` within-domain decay and independent,
near-orthogonal domain bases. Neither had been measured.

`RealMixture` presents DomainMixture's interface backed by Qwen2.5 hidden states,
so E1.6's instruments run **unchanged** on real vectors. Same instrument,
different world.

**KB3 holds at every point** — two models, three layers each, raw and
outlier-removed: twelve independent points. The register reduces blindness and
breaks the rarity monotone on measured geometry. §1's central claim survives
calibration.

**KB1 and KB2 both fail, and the models disagree about how.**

| | decay `a` | effective rank | overlap | massive share |
|---|---|---|---|---|
| DomainMixture | 0.50 assumed | 8 by construction | ~0 assumed | — |
| Qwen 0.5B | **0.237** — flatter | ~37 | **0.309** | 6.5% |
| Qwen 1.5B | **1.070** — steeper | ~1.5 | **0.292** | **71.5%** |

Cross-domain overlap is ~0.30 in both against an assumed ~0, so E1.1c's panel C
concern is **confirmed rather than swept**, and free-rank numbers taken on
independent bases are pessimistic.

**The direction survives; the magnitude was inflated by the generator.**

| register advantage (spectrum blindness ÷ register blindness) | |
|---|---|
| synthetic DomainMixture | **2.17×** |
| real, Qwen 0.5B | 1.37× |
| real, Qwen 1.5B | **1.09×** |

The spectrum is far less blind on real geometry because real domains *overlap* — a
rare domain's directions are partly retained for a common domain's sake, exactly
the mechanism panel C predicted. And **the advantage shrinks with model size**
across the only two sizes tested, so whether it survives at production scale is
untested and must not be extrapolated from two points.

**A finding about the criterion itself, independent of the register.** GPM's
energy criterion commits directions in order of retained energy. On the 1.5B,
**71% of that energy sits in five dimensions that are largely input-independent**
and carry no domain information at all — documented massive-activation behaviour,
confirmed by direct diagnosis (max |a| 227 against the 0.5B's 10.4). So the
criterion spends its retention budget first on directions that distinguish
nothing, and only the rank left over protects capabilities. **No synthetic
generator would have produced this**, and it is unaddressed.

**The level does not move.** Domains above the 0.05 interference bar at ρ = 0.95:
spectrum 7.5, register 7.3 of 8. Both arms leave nearly every domain unsafe. The
register wins the ratio and neither wins the level — a *different* unreadable from
E1.1d's, set by the interference bar against the real spectrum rather than by
subscription. Reporting the blindness improvement alone would have implied a
safety gain that is absent.

**Method, and it is B12 recurring.** The first run used one model and pooled
across layers: KB1 read NO at 0.237. Adding the second model flipped KB1 to *ok*
at a pooled 0.654 — a mean across two models that disagree by 5× on decay and 30×
on effective rank. That is the pooled-hides-tail defect committed inside this
experiment's own scoring. Now scored **per model, never pooled**.

### EB.3 · The spillover differential — the manipulation check fired, and the format control is the contribution

R10's condition, which §B rests on and the design never stated: *net-transfer
ranking separates abstractions from patches only while narrow patches have
near-zero real off-target effect.* E3.1's T4 showed +0.02 systematic spillover
inverts the margin. This is the measurement, and it did not produce the number.

**The first run reported every delta negative and argmax on the trained domain in
0 of 3 arms.** That is not huge spillover — it is broken training, and the check
that asks *did the narrow adapter most improve its own domain* caught it before
anything was published. **EB.1's hyperparameters had been inherited without
validation**: EB.1 used them to *time* steps and never checked that training
converged. Swept — every configuration that fits seven sentences drives train loss
to ~0.1 and makes held-out loss **rise**.

**Then the format control, which is the part worth keeping.** All eight domains
are short declarative factual prose: they differ in *topic*, not in *form*. So an
adapter trained on any one improves all of them by learning the shared form, and
that registers as spillover while having nothing to do with domain transfer. A
control arm trained on the same form and a topic no probe touches measures exactly
that:

| | on-target | off-target | ratio |
|---|---|---|---|
| raw | +0.5495 | +0.3408 | 0.62 |
| **format-adjusted** | +0.1230 | **−0.0315** | **−0.26** |

**78% of the raw spillover was format.** Without the control this would have been
published as *"narrow adapters leak 62% of their gain"*, which is false. It is the
correlated-error trap in a new place — train and probe sets sharing a property
that is not the property under test — and that is E0.2's shape.

**What it suggests, and must not be quoted as.** Format-adjusted, off-target
effect is slightly negative against a positive on-target gain, which is
directionally consistent with §B's condition. It is not readable as a result:
argmax landed on the trained domain in only 1 of 3 arms. Reported as a direction,
not a number.

**What it specifies.** R10 is not answered and is now precisely specified —
measuring real spillover needs per-domain text in enough volume to learn domain
*content* rather than form, and domains differing in *form* as well as topic. That
is the corpus the Rig B trip collects anyway, so Phase 4.2 joins 2.1, 2.2 and the
checkability sweep on **one acquisition** rather than being the model-only
measurement it was registered as.

The format control generalises: any transfer measurement between same-format
domains needs one, or it measures form.

### EB.4 · `R_t` is not cheap, and the buffer floor scales with dimension

E1.5, registered as *"measures wall-clock/memory, and the buffer size below which
the spectrum is too noisy to threshold"*. Both halves come back against the
design, and the second reaches backwards into EB.2.

**Cost — and the memory is not the expensive part.**

| | dim | layers | `R_t` all layers | weights | ratio | eig / layer |
|---|---|---|---|---|---|---|
| Qwen 0.5B | 896 | 24 | 154 MB | 278 MB | **0.55** | 69 ms |
| Qwen 1.5B | 1536 | 28 | 528 MB | 868 MB | **0.61** | 300 ms |

In the float64 the estimator holds; float32 halves it to ~28–30% and leaves the
conclusion. The rank-one update is microseconds, but the **eigendecomposition —
which every threshold read needs — is per layer**, so a full budget read costs
**~1.7 s at 0.5B and ~8.4 s at 1.5B**. Part II §A never states that number, and it
is the one that binds a budget read often enough to matter.

**The noise floor, and it is the half that reaches further.** Split-half agreement
of the top-r subspace never reaches 0.90 in the measurable range — at n = 256 it
is 0.71 (dim 896) and 0.65 (dim 1536) for r = 8. Stability rises ~linearly in
log₂(n), so the crossing is estimable, and marked as extrapolation:

> **r = 8 floor ≈ 586 at dim 896, ≈ 1026 at dim 1536** — that is **0.65× and 0.67×
> the dimension**. The floor tracks dim. KC3's good outcome, a constant floor,
> does not occur.

**What that does to I8.** I8 requires protection statistics on equal-sized
per-owner draws and never says how large. This answers it: below the floor a
per-owner spectrum is sampling noise, and any committed rank read off it is a
property of the draw. So **I8's equal N has a minimum set by the estimator, not by
policy — it is ~0.65 × dim, and it grows with the model.** At dim 1536 with a
64-owner fleet that is ~66,000 activation vectors before any protection statistic
is readable. Neither document connects the register's audit cost to model
dimension.

**And it reaches backwards into EB.2.** That experiment estimated per-domain bases
from ~118 vectors against an estimated floor of ~586, so its top-8 subspaces sat
at split-half stability around 0.5. Its decay and overlap *magnitudes* are noisier
than the caveat it carried, and this replaces the caveat with a number.

**KB3 is unaffected**, and the reason is worth stating: it compared spectrum
against register on the *same buffers*, so sampling noise moved both arms
together — which is why the direction held across all twelve points. **A paired
comparison survives a noisy estimate; absolute geometry figures do not.**

### EB.5 · The centering hypothesis — right about the small model, wrong about the large one

Worklist v2's item 1.1, posed as a hypothesis: massive activations are large and
roughly *input-independent*, so enormous in an uncentered `E[aaᵀ]` and ~0 in a
centered covariance — in which case EB.2's 71% is an artifact and the fix is one
line.

**The answer is in two places and they disagree**, which is why reading one wasn't
enough. `rig_a/core/spectrum.py` accumulates `R = λR + outer(a, a)` — **uncentered**,
and that is the estimator the *design* uses. EB.2 subtracted the mean before every
SVD — **centered**, and that is what EB.2 *measured*. So the hypothesis is wrong
about EB.2's number and right about the mechanism: the deployed estimator sees a
worse version than EB.2 reported.

| | uncentered | centered | mean²/var | dims shared |
|---|---|---|---|---|
| 0.5B | 22.2% | **6.0%** | 4.6 | 1/5 |
| 1.5B | 78.3% | **77.4%** | 0.19 | 4/5 |

**Right about the small model, wrong about the large one — the wrong way round for
scaling.** At 0.5B the concentration is mostly a mean offset and centering removes
it. At 1.5B it is real variance and centering does nothing. One model would have
answered this confidently and wrongly in either direction: B12 and EB.2's own
pooling defect, a third time.

**And the filter is justified with a number.** On the 1.5B the top-5 directions
hold **77.4% of centered variance and carry 0.2% between-domain share**, against
2.0% for every other dimension — a ratio of 0.13×. They consume three quarters of
the energy budget and discriminate essentially nothing, and `rank_for_energy`
cannot see the difference because it ranks on *total* energy.

**Two repairs, and the first is not sufficient.** Center `R_t` — one line with a
running mean, buys 16 points at 0.5B and ~1 at 1.5B. Then **rank by between-owner
variance rather than total energy**, which is the actual repair and is what the
design means by *committed*. Under the register that is **free**: owners are the
grouping, and per-owner buffers are what I8 already requires. The spectrum has no
grouping to compute it over — **an argument for §1 that §1 does not make.**

### R12 · Bound rank by data, not only by budget

EB.4's floor is a **data** floor, not a compute floor. Entries are the unit of
independence, so `n ≤ |provenance|` however long the texts and no budget buys more.
It therefore binds hardest on exactly the rare, small-provenance owners the
register exists to protect.

`f` read off EB.4's fitted curves — largest `r` whose subspace is stable at 0.90:

| n (entries) | dim 896 | dim 1536 |
|---|---|---|
| 40 / 100 / 250 | **none** | **none** |
| 1000 | 8 | 4 |
| 5000 | 16 | 16 |

> **`rank_o ≤ f(|provenance_o|)`.** A rare owner gets a small basis because that is
> all its data supports. If that is not enough competence, the answer is more
> provenance, not more rank.

It converts a silent failure into a refusal: without it a 40-entry owner requests
rank 8, gets it, and its basis is fitted to noise — while every statistic computed
against that basis is reported as an equal draw and satisfies I8. It is also the
missing bound on allocation, which §1.2 leaves unspecified. Full derivation and
the three things it does not settle: [docs/data_bounded_rank_rule.md](docs/data_bounded_rank_rule.md).

### EB.6 · `r` was not blocked, and it does not decay the way the pessimistic reading says

Worklist v2 §2 asked whether `r` is really corpus-blocked. **It is not**, and this
confirms it by measuring it: `r` needs a **generator and predicates**, not a corpus
and not traffic. E2.2 wrote the predicates; the models were already local. One of
five blocked measurements comes off the list without an acquisition.

| | r | r_generic | reached | r_specific \| reached | A ceiling |
|---|---|---|---|---|---|
| Qwen 0.5B | 1.00 | 0.78 | 9 | 1.00 | *inf* |
| Qwen 1.5B | 0.72 | **0.00** | 40 | **0.72** | 3.64 |

**The decomposition holds directionally.** Format compliance is *solved* between
these two sizes — `r_generic` goes to zero. The design screens do not: conditioned
on reaching them, rejection falls 1.00 → 0.72, nowhere near zero.

**And the residual concentrates in one screen.** `fan_out` fires on **68% of 1.5B
candidates** — a cap *stated in the prompt* and still violated two times in three.
That is the clearest instance of a design constraint no generator is trained on,
and stating it does not fix it.

So §4's pessimistic reading is not what the size axis shows. The component that
collapses is the one generators are trained on; the component encoding design
constraints does not — and it is additionally a **design variable**, since every
further invariant expressed as an executable necessary condition raises
`r_specific` soundly by construction.

**A conditioning defect I had to fix.** The first run reported `r_specific_only`
unconditionally — 0.23 at 0.5B against 0.72 at 1.5B — which reads as specific
rejection *rising* with model quality. Artifact: only 22% of 0.5B candidates
passed the format screens at all, so its unconditional specific rate is capped by
its generic rate. Two rates with different denominators in one column.

**Scope.** A 0.5B and a 1.5B are a short axis and neither is a production
generator, so the *level* of `r` is worthless as a design figure. The *direction*
across the pair is what transfers — the same split as EB.2. And the 0.5B's
conditional figure rests on the **9** candidates that reached the design screens.

### R13 · Specify the acquisition as predicates over the asset

EB.3 ran, its manipulation check fired twice, and the deliverable turned out to be
a specification for what the run needed. **That specification cost a run.** A
predicate over the asset costs nothing and is checkable before anything is
acquired — so asset B's spec is written as executable predicates, self-tested,
before the acquisition rather than after it.

**B carries three measurements, not four.** EB.6 took `r` off the blocked list, so
asset C is already spent: M1 checkability–difficulty (which locates E2.1 and
therefore harvest yield, which §1.3's oracle line now turns on entirely), M2
certified fraction, M3 entry-degree distribution.

**Every predicate is named after a failure in this record**, not an imagined one:

| | Named after | Blocks |
|---|---|---|
| **P0** independent resolution | E2.1's laundering — the only one that contaminates every measurement at once, and it **fails silently** | all three |
| **P1b** difficulty independent of outcome | E0.2's trap: if difficulty is read off success, the correlation is between a variable and itself | M1 |
| **P2a** decomposable scorer | §2's stated tension, better as an admission check — a cross-encoder means M2 **cannot be run at all** | M2 |
| **P2b** rival tail bounded | E0.5's **75.9% rival rise** — a rival outside top-k cannot be bounded, so without the (k+1)-th score M2 omits its own dominant failure mode | M2 |
| **P3a** bank unconstrained | E0.2e: a pre-filtered bank's degree distribution is the **filter's**, not the structure's | M3 |

**Self-tested, and that is not optional.** Each predicate is checked against an
asset built to violate exactly it, and all six fire. A validator whose checks
cannot fire is the defect this record found four times — and here it would fail
silently *after* an acquisition, which is the most expensive place for it.

**The two that cannot be retrofitted are P1b and P3a**, because both need a
decision at *capture* time: difficulty stamped after the fact is not independent,
and a bank filtered before capture cannot be unfiltered. Those are the ones to get
right first, and they are why the spec precedes the acquisition.

**What it does not do:** it checks *computability*, not representativeness. A log
passing every predicate can still be one deployment's traffic, and M1 would then
locate that deployment on E2.1's sweep rather than reality — which §10 already
concedes has no repair. And asset A (form-diverse expository text, for 4.2) is a
separate acquisition already specified by EB.3; A and B do not overlap and should
not be reconflated into "a corpus".

### §4 restated · `A` was never the binding quantity, and EB.6 is not a rescue

EB.6 measured `r_generic → 0.00` and `r_specific → 0.72`, which reads as a rescue
of §4. Working the accounting through says it is not. With `v = P(valuable | sound)`:

- unsound candidates cannot be promoted, so **promotions = `v × sound` either way**;
- with screens, labels = sound, so **labels per promotion = `1/v`**;
- without screens, labels = `sound/(1−r)`, so labels per promotion = `1/(v(1−r))`.

**Sound screens recover exactly the factor the generator's unsoundness cost, and
nothing more.** As `r → 0`, `A → 1` — but `1/v` never moved. The generator converts
wasted candidates into sound ones at the same oracle cost per promotion.

> **Screens do not amplify the oracle budget. They stop generator unsoundness from
> consuming it.**

So the constraint was always `v`, and `v` moves only under an instrument that scores
**value** — the thing E2.2 found no constraint-wise check reaches, §6 forbids as a
cheap rung, and E2.3 priced at recall 0.349. My pessimistic reading (`A → 1`) and the
`r_specific` refinement were both arguing about a factor that sits *beside* the
constraint rather than on it.

**The one number that decides §4** is the correlation between soundness and value.
The invariance assumes they are independent; if they correlate, screening genuinely
raises `v`. Register `P(valuable | sound)` against `P(valuable)` on the first
labelled set — same acquisition as M1, one column.

**Two things from EB.6 survive it.** `r_specific` is **one screen deep**: fan-out
fires on 0.68 against a specific-rejection union of 0.72, so only **0.05** is
rejected without fan-out firing — fan-out is **93%** of it, and "design constraints
don't come free with scale" rests on a single constraint. And there is **no perverse
incentive to withhold the checker from the generator**: under the invariance a
self-checking generator produces more sound candidates at the same `1/v`.

### R13 revisited · six defects in my own validator, two concealing each other

Found by reading the code at HEAD. The first version had **two instances of
"an operation that cannot report failure" hiding each other**, plus four more:

| | Defect |
|---|---|
| **1a.1** | no `domain` field — M1 yields only a *pooled* correlation, while E2.1's quantity is per-domain bias (0.095 mean, 0.176 worst). And a domain label applied after capture is an **inferred partition** — E0.1-K4 and E3.3's territory. Joins the cannot-retrofit set |
| **1a.2** | `P3c` evaluated `"cards" in rows[0]` while its stated job was E0.2f's ratio recoverability — the three ratio fields were **not in the schema at all**, so stripping them left it passing |
| **1a.3** | the self-test covered **6 of 11** while the docstring claimed every one — and the unfireable predicate sat *inside* the untested set |
| **1a.4** | `self_test()` failed **open** on the pass direction: `ok` was printed and never read, so a compliant asset failing a predicate printed *"validator is wrong"* and exited 0 |
| **1a.5** | `P3b`'s floor of 500 sat exactly on the cliff its own consequence string named. Now derived: separating a 10× tail from a 3× one needs ≥16 entries in the top 1%, so **n ≥ 1600** |
| **1a.6** | `P0`'s 20% was hardcoded. Now derived: reading E2.1's sweep at steps of 0.25 at 2 SE of a correlation needs **n ≥ 259** independent resolutions |

Now **14 predicates, all covered**, and `check_record.py` derives the count from
`predicates()` and asserts the break set covers every one — verified by deleting a
break case and watching the gate fire.

Two added from the review: **P0a outcome provenance** (the resolver must be disjoint
from the scorer, because a predicate over the asset cannot otherwise catch P0's
silent failure) and **P4 two independent sources** (representativeness can't be
checked within one log, but heterogeneity across logs can — bounding it rather than
closing it).

### EB.7 · EB.5's repairs built — and only one of them is the repair

EB.5 diagnosed and did not build. Both are now in `spectrum.py`, and measured on
*directions* rather than axes — `discrimination(u) = u'Bu / u'Tu`, the fraction of
what a direction moves that distinguishes one owner from another.

| top-5 directions | uncentered | centered | between-group |
|---|---|---|---|
| Qwen 0.5B | 0.0650 | 0.0914 | **0.4292** |
| Qwen 1.5B | 0.0355 | 0.0380 | **0.1579** |

**Centering is not the repair.** It moves discrimination barely at either size.
Ranking by between-owner variance is what moves it — **4.7× and 4.2×** against the
centered baseline — and it is the repair that needs a *grouping*, which under the
register is recorded and, under I8, already paid for.

**What it costs, beside what it buys.** The between-group basis captures less total
energy at the same rank — 10.5% against 21.3% (0.5B), 67.9% against 89.8% (1.5B).
It protects what distinguishes owners and leaves more of the shared bulk in the
free pool.

**And a ceiling nobody has stated.** The gain decays with rank and *inverts*:

| | r=5 | r=8 | r=16 | r=32 |
|---|---|---|---|---|
| 0.5B | 4.7× | 4.7× | 2.5× | 1.4× |
| 1.5B | 4.2× | 2.3× | 1.0× | **0.5×** |

`B` is a sum of `n_groups` outer products of deviations that sum to zero, so
`rank(B) ≤ n_groups − 1` — measured at exactly 7 with 8 groups. Past that the
"between-group" eigenvectors lie in **B's null space**, carry no between-group
variance at all, and the ranking selects arbitrary directions. At r=32 it is
*worse* than ranking by energy.

> **Committed rank chosen by between-owner variance is bounded by owner count, not
> by dimension.** A 64-owner fleet can commit at most 63 directions this way.

That is a third data-shaped bound on the same budget, beside R12's per-owner
provenance floor and E1.1d's subscription cliff — and it is the only one that gets
**looser** as the fleet grows, which is the opposite direction from R12's.

**What it does not settle:** discrimination is not tail safety. The next step is
E1.6's instruments with the committed basis chosen by `B` rather than `T` — whether
the *blindness ratio* improves, not just whether the directions discriminate.

### R14 · The §0 re-audit — the thesis predicts, weakly, and gains a criterion

The only free item that tests a **prediction** rather than checking a mechanism.
§0's table found eight *estimate-where-a-record-exists* sites against the records
of the time. Four records exist now that did not — the selection journal, the
commitment register, the residual set, the coverage cascade — and the thesis says
new records should reveal new deletions. That had never been tested, so it could
have returned nothing.

Each candidate had to clear two checks, and **both are places the audit can fail**:
is the estimate real, and is the record *sufficient* — same referent, same grain,
**available at the same time**.

| Site | Estimate today | Record | Verdict |
|---|---|---|---|
| merge prior | basis overlap between owners | the register's `basis: [direction_id]` | **clean** |
| deletion risk | `F_max` by entry *category* | tombstone events in L1 | **clean, and off-list** |
| gap set | posterior variance per signature | coverage cascade's unowned fraction | **partial** |
| routing | which region a query is in | the journal's winning card | **partial** |

**Merge prior** is the cleanest: overlap becomes a set intersection. Conditional on
the id form — the schema also allows `sketch`, under which overlap is estimated
again, so this is an argument for recording ids that did not exist before the
register did.

**Deletion risk was not among the three proposed candidates.** §3 prices `F_max` by
entry category — *personal data, user-corrected facts* — which is a taxonomy
standing in for a rate the ledger already records as tombstone events. Found by
asking §0's question of §3 rather than of the four new records. It substitutes
*eventually* rather than immediately: a cold ledger has no history, so the category
judgment is still needed as a prior — a different shape from all eight originals.

**Gap set decomposes rather than deletes.** The cascade records the never-compiled
half — I11's quantity, 0.483 at thin traffic in E0.6 — and says nothing about
owned-but-weak. The two halves have different repairs: one is closed by promoting
something, the other by improving what is there.

**Routing fails on *time*, not on referent.** Routing decides whether to use weights
or retrieve — *before* retrieval. The winning card is known *after*. What survives
is offline policy fitting on recorded outcomes, and any decision taken
post-retrieval.

> **The criterion the audit produced: a record substitutes for an estimate only if
> it is available when the decision is made.** All eight of §0's original sites
> satisfied that silently, so nobody had to state it. Routing is the first site
> where it binds.

**What it says about the thesis.** The prediction has content — two clean
deletions, one of them off-list — so §0 is not merely a description of eight
findings already in hand. But the yield has changed shape: all eight originals were
immediate and total; of four here, two are clean (one only *eventually*) and two
are partial. Consistent with the easy sites having been found first, and the
remaining yield per site is smaller than §0's table implies.

**What would have falsified it:** all four failing either check. Two did fail
sufficiency, which is why the answer is *confirmed, weakly* — and that is the more
useful answer.

### The last three free items — 1.3, 1.4, 1.7

None carries a kill criterion. All three are recording or record-correction, and
they close the free list.

**1.3 · R12's open question has a third answer, and it beats both I had.** I had
inheriting a neighbour's basis (which reintroduces the clause E0.7 killed) and
pooling (which costs the owner its independent statistic).

> **A sub-threshold owner does not become an owner.** It stays an L3 skill card and
> is served by retrieval.

No similarity judgment, so the neighbourhood clause stays dead; no pooling, so
nothing loses an independent statistic. And `|provenance| ≥ threshold` becomes a
**promotion criterion** — recorded, checkable *before* promotion, hence a **sound
screen** in §6's sense: one more hoisted conjunct, free.

The cost is honest and lands on the tail — rare capabilities stay retrieval-served,
which is the regime L4's gate `g` already handles — and it is **measurable rather
than arguable**: compiled against retrieval-served on the same probes.

**And a coupling neither mechanism knows about.** R12 pushes a thin owner to
broaden its provenance to qualify; §3's fan-out cap constrains which entries it may
broaden *into*. A thin owner seeking promotion and a saturated entry are in direct
conflict.

**1.4 · Within-owner coverage is now reported.** `ProbeRegistry.coverage()` gives
worst/best/spread per owner beside the count. I8 checks the count; coverage is what
protection depends on, and nothing reported it. On Zipfian provenance it reads
**40× spread** — matching E0.8's 38×, from a different code path.

**1.7 · Record sync — the format control reaches §8.** The abstraction test is
*"a merged owner must beat both parents on a third region neither targeted"*,
offered as the only definition of generalised a non-regression gate cannot fake.
If all three regions share a form, **the merged owner wins by learning form** and
the test certifies an average as an abstraction. The third region must differ in
**form as well as topic** — which makes its corpus requirement the same one asset A
already carries. First mechanism the EB.3 generalisation was *pointed at* rather
than derived from.

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

**Annotate what is true by construction, at the point it is constructed.** E0.1
reports `identical: yes` in three arms with three different statuses — a
*control* (A0, true by construction and that is its job), a *finding* (A1b, the
relative-invariant result), and a *bug* (A4-usage, B8). Nothing distinguished
them except which draw path each took, and determinism was a fact about the code
that had never been written down — so it had to be re-derived by reading, and a
clean result sat under a cloud in between. The annotation is now in `draw()`
itself, not in a review. This is the record's own thesis applied to the
programme: **recorded, not inferred.**

**One null-treatment row per draw policy, not one control per arm table.** A
single control stands for a table only if every arm shares its nuisance
parameters. E0.1's do not: draw policy and decay both vary while A0 alone stands
for nine arms. So A4-stratified confounds relabelling with resampling — the
manipulation and the redraw arrive together — and there is no row that isolates
either. The rule is a null row *per policy that reaches the stochastic branch*,
plus an RNG-seed nuisance sweep on those arms reported beside the treatment
effect. Had that sweep run on A4, B8 would have been the experiment's own output
rather than something found by reading the code two rounds later.

**Before registering a control, check whether the null is derivable from the code
path.** A control on a deterministic path is dead weight, and worse: it casts
doubt on a clean result. A1a is `cap=None`, which exits `draw()` before the
policy branch, so its null is *derivable* — alive unchanged ⇒ equal draws ⇒
over-forgetting exactly 0 — and no run was needed to establish it. Registering
one anyway put E0.1's 12.4× blindness factor under a cloud for two rounds for
nothing. This is the exact mirror of B7, where a control was registered on a
quantity the run never exercised: **both are the same failure to ask what the
code can do before asking what the world did.**

**One null per arm, differing in exactly one respect — and "one respect" has to
include the parameters the treatment perturbs downstream.** This rule has now
been found too coarse twice, by its own runs. Per-table failed when A4-stratified
confounded relabelling with resampling. Per-policy failed when both A6 arms
carried decay, which shrinks provenance 600 → 450 against a fixed 300-entry cap,
moving the draw fraction 0.50 → 0.67 — so the null measured the draw fraction
rather than the policy, and A6-uniform landed *below* its own null. The stable
form is one null per arm. These runs take seconds; the coarser forms were
economising against nothing.

**An incoming correction is a claim and gets the same check as an internal one.**
Verify it against the artifact it names before recording it. B19 entered this
record because a reviewer's sentence conflated a true statement with a figure it
did not apply to, and I recorded the conflation without opening the results file
that was already to hand. Accepting a correction feels like humility, which is
exactly why it draws less scrutiny than making one — and a wrongly-accepted
correction is worse than a wrongly-made claim, because it also discredits a sound
number. Second wrong attribution here, after B11.

**"Inert" is three different things, and only one of them is a finding.** The word
was itself an overloaded identifier, which is why `identical: yes` could conflate a
control, a finding and a bug — and why K4 was scored on the wrong statistic for
two rounds. The taxonomy:

| mode | what happened | status |
|---|---|---|
| **not mutated** | the treatment changed no state, and both calls took the same arguments | **bug** — B20, and A0 by design |
| **mutated but never read** | state changed; nothing on the path to the measurement reads it | **bug** — B8, and A4-usage by design |
| **read but output-invariant** | state changed, was read, and the metric still did not move | **finding** — A1b |

Only the third says anything about the world. The first two say something about the
code, and they are answerable *statically, before any data exists*. Declare which
arms are expected inert; anything else reporting inert is a bug report.

**Worth noting how this one was found: by building an instrument.** That is a third
detection mechanism alongside *a measurement going the wrong way* (7 bugs) and
*reading the code* (B20, the I7 collision). It is the only one that forces
definitional precision **before** any data exists — the taxonomy above did not
exist until something had to decide `inert: yes/no` in a column.

**"X and Y are the same quantity" and "A is a subset of B" are measurements, not
arguments.** A structural relationship between two quantities is exactly as
checkable as a number, and it is the kind of claim that reads as insight and gets
waved through. This programme has now falsified four of them by measuring:
§2's *uncertified ⊆ closure edges* (false, and 8.9% of the excess really flipped),
§1.3's *sharing follows provenance neighbourhoods* (false at every pool size, the
count is on the birthday curve), the *E0.6/E0.7 unification* (false — depth governs
coverage and not the oracle line), and *B19's confound* (false — the two figures
were within-arm and matched). Every one was checkable from data already collected
before it was asserted.

**Ask whether an arm COULD have moved its own measurement, before reading what it
did.** B7, B8, B20 and A4-usage are one defect at four sites, and inspection found
all four — after three of them had produced a published number. The question is
not *did the number move* but *could it have*, and that is answerable statically:
an arm is inert when nothing it mutated is read on the path to the measurement
**and** both calls took the same arguments. `rig_a/core/trace.py` is that check in
about forty lines, and E0.1 now asserts it over every arm before any metric is
read. Arms that are supposed to be inert are declared; anything else is a bug
report. Four instances is enough to stop finding these by inspection.

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

- **B12** EB.1 reported `H1 NO` on +1986% per-token drift, all of it from one
  point — and the run's own output was self-contradictory, printing the failure
  and then asserting the opposite two lines later. Quadratic attention predicts
  ~2× from 512→1024, not 19×. The 0.5B model put the identical cliff at 2048
  instead of 1024, so the blowup **scales with model size** — paging on 8 GB, not
  the algorithm. Caught by *"why is this number so extreme?"*, which is now the
  fourth bug that question has found and the mechanism that keeps working.

- **B13** E5.2's first decision table conflated compile adequacy with decay
  robustness and reported over-forgetting *conditional on passing*, so the
  denominator moved with `k` and **k=10 came out worse than k=3** — backwards for
  anything called redundancy, and unusable as the lookup it claimed to be. Caught
  by the table being non-monotone in the direction it informs.

- **B14/B15** E2.3 scored its rare-region criterion against an empty population
  **twice**, then against a wrong baseline a third time. First against the mean
  objective, under which no rare specialist is ever "good"; then against a
  gap-closing objective that still favoured generalists 0.56 to 0.166; then
  against frequent specialists, which lift zero regions and are never good
  either. The manipulation check caught all three. The rule that came out of it:
  **name the objective before scoring a criterion, and verify every compared
  population is non-empty before running the sweep** — not after.

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
