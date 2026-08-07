# Review of the WAM falsification harness

Two experiments added and run (`e1_1c`, `e3_1b`), two analytical findings without
runs. One of the new results partially reverses E1.1b; one partially rehabilitates
E3.1's central claim.

---

## First, what the harness gets right

Stating this because it changes how much weight the findings below should carry.

**Pre-registering kill criteria in the module docstring, before the first run,**
is the single practice that makes the rest trustworthy. Nearly every result here
could have been narrated into a pass otherwise, and the E1.1 → E1.1b reversal is
proof the discipline binds: the reversal happened because the criterion change was
made *visible and auditable* rather than quietly substituted.

**The retraction record is better than the results.** B4 — a generator re-drawing
its domain subspaces per call, so train and held-out came from unrelated worlds —
manufactured a headline that survived a full write-up and biased *against* the
design. Catching it, publishing it, and then reasoning correctly about why no
second measurement would have caught it, is the part of this repo that would be
hardest for someone else to reproduce.

**The escalation of detection methods is the right thing to be tracking.**
Inspection → contradiction between two measurements → asking whether a striking
number was a parameter you chose → implausible direction → counting what should
have been there. The observation that only the fourth scales is correct, and it
is what produced both of my findings below: E1.1c came from asking what the mean
was averaging over, E3.1b from asking whether the world's asymmetry was argued
for or assumed.

**Withdrawing "every failure is a specification defect" as selection rather than
evidence** is the most self-aware thing in the document. The first eight
experiments all asked local-mechanism questions, and local errors are locally
repairable by construction. Recognising a tautology in your own summary statistic
is rarer than finding a bug.

Two process notes worth keeping: `str.replace` failing open (B6) is a whole class
of bug, and asserting the anchor exists is the right fix. And "a repaired
mechanism must be re-registered and re-run" is the rule that stops R1–R8 from
becoming folklore.

---

## E1.1c · The budget's PASS is a traffic-average artifact

**This partially reverses E1.1b, by the same move E1.1b used on E1.1: not a new
mechanism, a corrected aggregation.**

E1.1b chooses `r` with `rank_for_energy(est, calib, rho)`, which sums energy over
calibration queries drawn at `DomainMixture`'s Zipfian rates, then measures safety
with `leakage_by_rank` and `interference_by_rank` — both of which return a **mean
over held-out traffic queries**. Choice and measurement are weighted by the same
distribution.

That is not a rounding concern. It is what "energy" means: **protection is
allocated in proportion to frequency.** A rare domain's subspace therefore lands
in the tail, below the cut, and is classified free — and the metric that should
catch it cannot, because it averages over the distribution that made the domain
rare. At 32 domains the tail domain contributes 0.8% of the mean it is judged by.

Same G1 bar (interference ≤ 0.05), asked per domain on equal-sized probe sets:

```
--- 16 domains, rank 8, rate ratio 16x ---
     rho    r  free  int_traffic  int_worst  leak_worst   failing
  0.9500   65    63       0.0430     0.1033      0.2049     12/16   <- E1.1b's operating point
  0.9900   90    38       0.0215     0.0727      0.0549      4/16
  0.9950   98    30       0.0169     0.0505      0.0271      1/16
  0.9990  110    18       0.0089     0.0355      0.0079      0/16   <- tail-safe

Spearman(traffic rate, domain leakage) at rho=0.95: -0.965
per-domain leakage, most -> least frequent:
  0.001  0.008  0.014  0.032  0.041  0.066  0.135  0.103  0.154  0.112  ...  0.205  0.189
```

**P1 fails.** At ρ = 0.95, 6/8, 12/16 and 25/32 domains exceed the interference
limit while the traffic mean reads 0.043–0.058 and passes.

**P2 fails, and this is the part that makes it structural.** Spearman(rate,
leakage) = **−1.000, −0.965, −0.917**. Monotone in rarity across all three
configurations. A single unlucky domain would not correlate; this is the criterion
doing exactly what it is defined to do.

**P3 — the number worth carrying forward.** Tail-safe retention is ρ = 0.999, and
free rank there is **18/128 and 17/128** at 16 and 32 domains — 13–14% of
dimension, against the ~50% E1.1b reports at ρ = 0.95. Against `RANK_REQUEST = 8`
that is room for roughly two adapters.

**Why this matters more here than it would elsewhere.** L7's own admitted failure
mode is coverage bias, and the rare-domain tail is precisely the long-tail personal
knowledge the ledger exists to preserve. A budget that is safe on average and unsafe
in the tail is unsafe for WAM's stated purpose specifically. "Empty only at 5×
over-subscription" was measured at a retention that is not tail-safe.

**Both obvious repairs fail.**

```
repairs at rho=0.95, 32 domains
  baseline                 r= 63 free= 65  worst=0.1315  25/32 failing
  R-a worst-case rank      r=100 free= 28  worst=0.0718   2/32 failing
  R-b freq-balanced R_t    r= 76 free= 52  worst=0.0870  24/32 failing
```

R-a (take the union of every domain's own top-r) helps a great deal on leakage
(worst 0.33 → 0.047) and still misses the interference bar. R-b (accumulate `R_t`
with observations importance-weighted by 1/rate) barely moves it. So the honest
repair is not a reformulation — it is that **ρ stops being a free parameter and
becomes a measured one: set it from worst-domain interference.** That is
affordable, since it needs domain labels only at calibration time, and it is the
same move as R1 and R7 — measure what happened rather than trust the bookkeeping
rule.

**Leakage and interference are not interchangeable, and interference binds.** R-a
cuts worst leakage by 7× and leaves worst interference above the bar, because
`interference_by_rank` writes into `U_free[:, :r]` — the *largest* sub-cut
directions. That is the right conservative choice and worth stating explicitly:
G1 should always be the interference arm, and any future repair evaluated on
leakage alone will look better than it is.

**Panel C — the falsifier for my own claim, and it partly succeeds.** Random bases
in dim 128 are near-orthogonal, which is pessimistic for free rank; real domains
share structure. Sweeping a shared-subspace fraction:

```
 domains  share  overlap  tail-safe rho    r  free
      16    0.0    0.062          0.999  110    18
      16    0.4    0.142          0.999  107    21
      16    0.7    0.728          0.995   71    57
      32    0.7    0.723          0.995   71    57
```

At overlap 0.73 the tail-safe retention relaxes to 0.995 and free rank triples.
**So the severity is a function of real domain-subspace overlap, which is a Rig B
measurement** — and it is the same measurement PLAN.md §3 already identifies as
the binding input ("how much shape varies across regions"). Add per-region
overlap to that measurement; it is the same activation pass.

**Consequence for E1.2.** The retracted B4 headline was "realistic traffic leaves
6 free directions," and E1.2's predicted deadlock explicitly noted it "compounds
with" that number. The number is now non-retracted and only ~3× larger: **17–18
free directions at the tail-safe point, against a rank-8 request.** E1.2 should be
re-prioritised — three-λ unanimity releasing rank slowly against a budget this
tight is now the most likely place for a genuine architectural failure.

---

## E3.1b · T4's inversion is an asymmetry artifact, not a spillover finding

**This partially rehabilitates the central generalization claim.**

E3.1's T4 applies spillover only when `kind == "patch"`. Skill adapters get none.
The justification offered is *"a low-rank update fit to one region is not
region-confined"* — but that argument does not distinguish the two candidate
types. A skill adapter is also a low-rank update, fit to a *broader* distribution,
so on that reasoning it should spill **more**, not less.

The arithmetic makes the concern sharp: net transfer is a sum over off-target
regions, so adding the same constant to every candidate shifts every score
equally and leaves the ranking untouched. Only an asymmetry can invert the margin.

```
regime                                 target  transfer   margin  sk(t)  sk(x)  comp_only(x)
clean                                   0.264     0.349   +0.085    9.2   10.1         0.550
noisy + spurious                        0.075     0.206   +0.131    6.3    8.5         0.283
spill 0.02  patch only  (E3.1 T4)       0.015     0.006   -0.010    3.9    3.1         0.017
spill 0.02  symmetric                   0.040     0.216   +0.176    5.2    8.6         0.350
spill 0.05  patch only                  0.000     0.000   +0.000    0.5    0.0         0.000
spill 0.05  symmetric                   0.034     0.245   +0.210    4.7    8.9         0.408
```

**S2 confirmed.** Patch-only spillover inverts the margin; symmetric spillover of
the same magnitude gives **+0.176** — more than double the clean margin — and at
0.05 gives **+0.210**. Compositional-only acquisition goes 0.017 → 0.350.

The differential sweep, skill spillover held at 0.02:

```
 patch spill   excess   target  transfer   margin
       0.020    0.000    0.040     0.216   +0.176
       0.030    0.010    0.014     0.062   +0.047
       0.040    0.020    0.005     0.007   +0.002
       0.050    0.030    0.001     0.001   -0.000
```

**So the condition Part II §B owes is not the one PLAN.md states.** Not "narrow
patches must have near-zero real off-target effect" — that is the absolute
condition, and it is implausible. The actual condition is **"patches must not
spill over more than skills do, by more than ~0.03"** — which in this world's
units is roughly `0.5 × GAIN_SKILL` (0.06) or `0.21 × GAIN_PATCH` (0.14), so it
should be carried as a ratio rather than an absolute.

That is a differential condition, it is much weaker, and it is arguably favourable
a priori for the reason given above. It also changes the Rig B measurement, and
makes it cheaper: **measure the difference in off-target spillover between a
narrow adapter and a broad one, not the level of either.** A paired measurement
on the same probe set, which also removes the between-run variance that would
dominate two separate level measurements.

**What survives from T4 unchanged, and it is the more interesting half.** Both
arms collapse in absolute terms under patch-only spillover (3.9/3.1 skills against
9.2/10.1 clean). And PLAN.md's mechanism reading still stands and is not affected
by any of this: net transfer *does not detect generality, it removes the reward
for narrowness* — transfer acquires compositional-only skills at 55% while
learning 72% of skills overall. That is the sharper finding in E3.1 and I would
lead with it over the T4 regime.

---

## R2 relocates the gap detector from Root 1 to Root 2

Analytical, no run needed, but it changes which root E1.4 belongs to.

`world.py` has `epistemic_var() = 1.0 / (1.0 + self.visits)` — a deterministic,
monotone, region-independent function of visit count. And `reducible` is
`max(0, prev_predictive − cur_predictive)` where
`predictive = epistemic + aleatoric = 1/(1+v) + p(1−p)`.

Decompose the decline. The `1/(1+v)` term falls identically in every region,
learnable or not, and decays as `1/v²`. So it cannot discriminate — it only
*exhausts*, which is why the noise region eventually stops asking. **The
discriminating signal is entirely the `p(1−p)` term, i.e. competence improvement.**

Which means R2 is competence-improvement measurement wearing variance clothing.
And competence improvement requires a per-region outcome signal at two time
points — a verifier in that region.

**So the "free gap detector, no new estimator" claim in Part I §3 does not survive
its own repair.** It dies in exactly the domains that most need a gap detector:
the T3 domains, where by design nothing can measure whether practice helped. E1.4
is not a Root 1 estimator problem with a Root 1 fix. It is Root 2 wearing a Root 1
costume, which is one more instance of PLAN.md §0's collapse and mild evidence for it.

**Cheap Rig A test, and I would run it before adopting R2.** Gate the competence
signal to a verified subset of regions; put the noise region in the *unverified*
subset. Prediction: the magnet returns for unverified regions, because `reducible`
falls back to the undiscriminating `1/v²` term there. If that holds, R2's kill
criteria need a third condition — that it works where no verifier exists — and it
will fail it.

## `epistemic_var = 1/(1+visits)` is optimistic in a way that matters

Real posterior variance is `q^T R^{-1} q`: it depends on **where** the visits
landed in feature space, not how many there were. A region whose features are
collinear keeps high variance in unexplored directions no matter how often it is
practised, and a region can be heavily visited and still epistemically wide open
along a direction traffic never probed.

The current model cannot express that, so it understates the difficulty of every
variance-based reading. Both E1.4's failure and R2's pass are therefore measured
in a world kinder than the real one. Worth one line in `world.py`'s docstring, and
worth folding into E1.3 — which is already the right experiment for it, since
Spearman(predicted variance, realized error) on real queries is exactly the
quantity a visit-count model cannot predict.

Cheap partial fix inside Rig A: give each region a within-region feature basis and
sample practice tasks non-uniformly over it, so visits accumulate anisotropically.
Then `epistemic_var` becomes a function of *coverage* rather than count, and the
distinction between "practised a lot" and "known" becomes representable.

---

## Two smaller notes

**E1.1c's `n_domains = 8` case passes P3 comfortably** (65 free directions at
ρ=0.999) while 16 and 32 do not. Free rank at the tail-safe point falls steeply
between 8 and 16 domains and then flattens. Worth one sweep to locate that knee,
because if it sits at a realistic domain count the finding is severe and if it
sits above one it is not.

**`interference_by_rank` draws a fresh random `delta_W` per call**, so every
number in E1.1b and E1.1c carries single-draw variance from the adapter direction
as well as from the world. It did not change any verdict here — the effects are
large relative to it — but the metric would be tighter as a mean over a few draws,
and E1.1b's `safe` flag is a threshold comparison on a single draw.

---

## Where I would go next

1. **Re-run E1.1b's verdict with a per-domain worst case.** Not a new experiment —
   E1.1c is the measurement. What it needs is for the *reported* operating point to
   move from ρ=0.95 to the tail-safe one, and for the headline to become free rank
   at that point rather than at ρ=0.95.
2. **E1.2, promoted.** It now compounds with a live number instead of a retracted
   one, and it is the strongest remaining candidate for a genuine architectural
   failure.
3. **The R2 verifier-gating test above,** before R2 is adopted. It is ten lines.
4. **Rig B: measure per-region activation-spectrum overlap** in the same pass as
   the spectrum shape. Panel C makes overlap the parameter that decides E1.1c's
   severity, and it is free once the activations are captured.
5. **Rig B: measure the spillover *differential*,** not patch spillover. Paired,
   same probe set, narrow adapter versus broad. This is the measurement that
   decides §B and it is now much cheaper than PLAN.md assumes.
6. **E0.1 remains untested and is the largest unmeasured claim in the design.** I4
   is what every other tier's safety argument reduces to, and Part II §G calls it
   an intention. Three of five architecturally-consequential claims untested is
   the right thing for the status table to say loudly.
