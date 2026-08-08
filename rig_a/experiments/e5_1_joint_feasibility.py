"""E5.1 -- Joint feasibility. Does the feasible region have anything in it?

Every experiment in this programme has moved one thing with everything else at
defaults. That is the right way to find a mechanism and the wrong way to find
out whether a design is buildable, because the constraints are coupled through
shared parameters and every one of them tightens under the others:

    tail-safe protection needs rho ~ 0.999, leaving 13-14% of dimension  (E1.1c)
    and that boundary sits at subscription 1.0x unless overlap rescues it (E1.1d)
    bounded recompile needs a capped L3 draw                             (E0.1 A6)
    but WHICH entries the cap admits is a protection decision            (weighting rule)
    and a stratified draw still leaves 52% worst-region over-forgetting  (E0.1)
    cascade breadth sets deletion cost and the design has no lever on it (E0.2d)
    batching amortises deletion but bounds restoration latency           (E0.2c)
    per-region protection costs probe budget linear in region count      (weighting rule)
    and re-partitioning itself costs 21.7% competence                    (E0.1 K4)

A thesis of this kind does not fail by refutation. It fails by the feasible
region emptying. No experiment here has tested the conjunction.

THE OUTPUT IS A REGION, NOT A VERDICT. What matters is which constraints bind
where, and whether any configuration satisfies all of them at once.

WHAT IS MEASURED AND WHAT IS COMPUTED -- stated because it decides how much the
result is worth:

    MEASURED here, by running the world:
        C1 tail-safe free rank, per (region count, subscription, subspace overlap)
        C6 worst-region over-forgetting, per (region count, draw-cap fraction)
    COMPUTED arithmetically from relationships already measured elsewhere:
        C2 probe budget            = regions x probes-per-region   (weighting rule)
        C3 deletion throughput     = capacity x batch / (|union| x recompile)  (E0.2c)
        C4 restoration latency     = batch / arrival + drain                   (E0.2c)
        cascade breadth from provenance overlap                               (E0.2d)

THE AXIS THAT MAY NOT BE FREE. C1 improves with SUBSPACE overlap between domains
(E1.1d: overlap 0.7 moved the boundary from 1.0x to past 3.0x). C3 worsens with
PROVENANCE overlap between cards (E0.2d: breadth 63% -> 99%). Those are different
quantities -- feature geometry versus which ledger entries feed which cards -- so
they are swept independently here. But they are plausibly correlated, since
domains that share subspace plausibly share source material. If they are, the
two constraints pull opposite ways on ONE knob and the feasible region is the
diagonal slice, not the product. Both are reported.

KILL CRITERIA (pre-registered):
    F1 fails if the feasible region is EMPTY over the whole swept space. That
       would be the first result in this programme that the ledger-first thesis
       could actually fail on.
    F2 fails if the feasible region is non-empty in the product but EMPTY on the
       correlated diagonal -- feasible only if you can pick subspace overlap and
       provenance overlap independently, which you may not be able to.
    F3 reports which constraint binds most often. Not pass/fail: it says where
       the next repair is worth spending.

Is there a world that produces the other verdict? Yes and it is the expected
one: at 0.25x subscription with a full draw and generous latency tolerance every
constraint is slack, so a non-empty region is the default expectation and F1
firing would be the surprise.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.spectrum import (  # noqa: E402
    RtEstimator,
    interference_by_rank,
    rank_for_energy,
)

DIM = 128
SEED = 20260806
N_OBS = 3000
N_CALIB = 400
N_PROBE = 250
RANK_REQUEST = 8
INTERFERENCE_LIMIT = 0.05
RETENTIONS = (0.95, 0.99, 0.995, 0.999, 0.9995)

# -- swept axes --------------------------------------------------------------
# 2 and 4 added after the first interiority check: the feasible set sat at the
# minimum swept region count, so the boundary was outside the box.
REGIONS = (2, 4, 8, 16, 32)
DOMAIN_RANKS = (4, 8, 16)              # with REGIONS gives subscription 0.25x-4x
SUBSPACE_OVERLAP = (0.0, 0.4, 0.7)
PROVENANCE_OVERLAP = (0.1, 0.4, 0.7)   # -> cascade breadth, via E0.2d
# 0.10 added for the same reason.
DRAW_FRACTION = (0.10, 0.25, 0.5, 0.75, 1.0)
BATCH = (1, 4, 16, 64)
DECAY_POLICY = ("use_based", "stratified")
# Swept, because fixing it at 0.25 made C6 fail in 95.8% of configurations
# and dominate the sweep: a 3-entry support with a 2-of-3 pass threshold
# loses ~16% of items to 25% decay INHERENTLY, whatever the draw does.
DECAY_RATE = (0.05, 0.15, 0.25)
# C6's version of H: a pinned structural choice that gates 57% of eliminations.
# (support size, pass threshold as a fraction). 3-entry/2-of-3 was the only value
# in earlier runs, and a 3-entry support loses ~16% of items to decay rate alone.
SUPPORT_SHAPES = ((3, 0.5), (6, 0.34), (10, 0.30))

# -- tolerances --------------------------------------------------------------
PROBE_BUDGET = 6000          # evaluations available per protection statistic
DELETIONS_PER_DAY_MIN = 1.0
RESTORE_LATENCY_MAX_DAYS = 7.0
OVER_FORGET_MAX = 0.20       # worst-region
DELETION_ARRIVAL_PER_DAY = 2.0

# Deployment profiles. These were fixed constants in the first two runs, and
# fixing them is what produced "EMPTY" -- C3 and C4 turn out to be jointly
# unsatisfiable BY ARITHMETIC at (fleet 64, 8h recompile, 4 parallel, 7-day
# tolerance), independent of every swept axis:
#
#     C3 needs  b >= D * U * H / C          (=5.33 there)
#     C4 needs  b <= A * (L - U * H / C)    (=3.33 there)
#     feasible iff  D*U*H/C <= A*(L - U*H/C)
#
# with U saturating at the fleet size. The drain term U*H/C alone consumed 5.33
# of the 7-day budget. Reporting emptiness at one such point would have been a
# statement about the constants, not the design, so the profile is now swept and
# the closed-form window is reported alongside.
# The concurrent : promoted ratio is a DEPLOYMENT POLICY, so it is swept rather
# than chosen -- same class as L and the I7 density floor.
CONCURRENT_RATIO = (0.05, 0.15, 0.35, 1.0)
# Part II section A requires disjoint bases, which implies
#     concurrent_fleet * RANK_REQUEST <= DIM
# At rank 8 in 128 that is 16 adapters, hard. The design states neither this
# bound nor its alternative (that adapters serving different regions share
# basis, which section A's interference argument does not cover).
PROFILES = (
    ("as specified",     64,  8.0,  4.0,  7.0),
    ("fast recompile",   64,  2.0,  4.0,  7.0),
    ("high parallelism", 64,  8.0, 16.0,  7.0),
    ("small fleet",      16,  8.0,  4.0,  7.0),
    ("tolerant latency", 64,  8.0,  4.0, 30.0),
    ("all four eased",   64,  2.0, 16.0, 30.0),
)


# ---------------------------------------------------------------- C1, measured
def tail_safe_free_rank(n_regions, domain_rank, share, rng) -> int:
    common = np.linalg.qr(rng.normal(size=(DIM, domain_rank)))[0]
    bases = []
    for _ in range(n_regions):
        own = np.linalg.qr(rng.normal(size=(DIM, domain_rank)))[0]
        bases.append(np.linalg.qr(share * common + (1 - share) * own)[0])
    within = np.arange(1, domain_rank + 1, dtype=float) ** -0.5
    rates = 1.0 / np.arange(1, n_regions + 1, dtype=float)
    rates /= rates.sum()

    def draw(n):
        lab = rng.choice(n_regions, size=n, p=rates)
        f = np.empty((n, DIM))
        for i, d in enumerate(lab):
            f[i] = bases[d] @ (rng.normal(size=domain_rank) * within)
        return f

    est = RtEstimator(dim=DIM, lam=1.0)
    est.update(draw(N_OBS))
    calib = draw(N_CALIB)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)
    probe = {d: (rng.normal(size=(N_PROBE, domain_rank)) * within) @ bases[d].T
             for d in range(n_regions)}
    for rho in RETENTIONS:
        r = rank_for_energy(est, calib, rho)
        worst = max(interference_by_rank(est, r, probe[d], readout, RANK_REQUEST, rng)
                    for d in range(n_regions))
        if worst <= INTERFERENCE_LIMIT:
            return int(DIM - r)
    return 0


# ---------------------------------------------------------------- C6, measured
def worst_over_forgetting(n_regions, draw_fraction, decay_policy, decay_rate,
                          support_shape, rng) -> float:
    """E0.1's mechanism: a bounded draw over decayed provenance.

    `decay_policy` is an axis because the first run of this experiment had C6
    failing in 100% of configurations, including at an UNCAPPED draw -- which
    means it was measuring the Consolidator's decay, not the draw cap. A global
    use-based cut removes 60-70% of the rarest regions' entries while leaving
    frequent regions untouched, because Zipfian region rates span ~10x while
    within-region usage variation spans ~3x, so the distributions barely overlap
    and a global threshold partitions almost exactly by region.

    That is enumeration site 9 of the weighting rule -- "decay unused entries" is
    a P-class statistic computed over frequency of reference -- inferred there
    and measured here for the first time.
    """
    per_region = 50
    n_entries = n_regions * per_region
    region = np.repeat(np.arange(n_regions), per_region)
    rates = 1.0 / np.arange(1, n_regions + 1, dtype=float)
    rates /= rates.sum()
    usage = rates[region] * rng.uniform(0.5, 1.5, size=n_entries)

    sup_size, sup_thresh = support_shape
    items = []
    for _ in range(30 * n_regions):
        r = int(rng.choice(n_regions, p=rates))
        pool = np.where(region == r)[0]
        k = min(sup_size, len(pool))
        items.append((r, set(rng.choice(pool, size=k, replace=False).tolist())))

    cap = max(int(draw_fraction * n_entries), 1)

    def stratified(live):
        per = max(cap // n_regions, 1)
        out = []
        for r in range(n_regions):
            p = live[region[live] == r]
            if len(p):
                out.extend(rng.choice(p, size=min(per, len(p)), replace=False).tolist())
        return set(out[:cap])

    live0 = np.arange(n_entries)
    before = stratified(live0)
    if decay_policy == "use_based":
        # the design as written: one global threshold on reference frequency
        keep = np.argsort(usage)[int(decay_rate * n_entries):]
    else:
        # weighting-rule compliant: each region sheds the same FRACTION of its
        # own least-used entries, so decay cannot preferentially empty the tail
        keep = []
        for r in range(n_regions):
            pool = np.where(region == r)[0]
            order = pool[np.argsort(usage[pool])]
            keep.extend(order[int(decay_rate * len(order)):].tolist())
        keep = np.array(sorted(keep))
    after = stratified(np.sort(keep))

    per_region_rate = []
    for r in range(n_regions):
        sel = [(s) for (rr, s) in items if rr == r]
        if len(sel) < 5:
            continue
        lost = [1.0 for s in sel
                if len(s & before) / len(s) >= sup_thresh
                and len(s & after) / len(s) < sup_thresh]
        per_region_rate.append(len(lost) / len(sel))
    return max(per_region_rate) if per_region_rate else 0.0


# ------------------------------------------------------- C2/C3/C4, computed
# E0.2f reconciled the two breadth instruments over the same admitted bank and
# found they AGREE at 0.087, while E0.2d's 0.63-0.98 came from a world with 8
# adapters, 8 cards and 60 entries. Fitting 0.60 + 0.45*po to that point and
# making the result decide feasibility was fitting to one very small world.
# Swept instead, spanning both measurements, because which holds depends on the
# real bank : fleet : rollouts-per-adapter ratio and nobody has measured it.
BREADTH_BETA = (0.087, 0.20, 0.31, 0.50, 0.65)


def union_size(fleet, beta, batch):
    """Adapters touched by a batch of `batch` tombstones.

    Each adapter is touched independently with probability beta per tombstone,
    so the union saturates EXPONENTIALLY: fleet * (1 - (1-beta)^batch). An
    earlier version used fleet * beta * batch clipped at fleet, which is linear
    and always at or above the correct value -- at beta 0.1 and batch 10 it
    claimed the whole fleet where the true figure is 65%. That overstated C3's
    cost most severely in the LOW-breadth regime, which is exactly where
    feasibility lives.
    """
    return max(fleet * (1.0 - (1.0 - beta) ** batch), 1.0)


def deletions_per_day(prof, beta: float, batch: int) -> float:
    _, promoted, h, par, _ = prof
    return (24.0 * par) * batch / (union_size(promoted, beta, batch) * h)


def restore_latency_days(prof, batch: int, beta: float) -> float:
    _, promoted, h, par, _ = prof
    fill = batch / DELETION_ARRIVAL_PER_DAY
    drain = union_size(promoted, beta, batch) * h / (par * 24.0)
    return fill + drain


def batch_window(prof, beta: float, bmax: int = 400) -> list[int]:
    """Batch sizes satisfying C3 AND C4 -- a FIXED POINT, not an interval.

    U depends on b, so the two conditions are

        D * U(b) * H / C  <=  b  <=  A * (L - U(b) * H / C)

    and both bounds move with b. An earlier version evaluated them at the
    SATURATED union U = FLEET, which pins cascade breadth at its worst possible
    value and then reports the result as holding "independent of every other
    axis". It is independent of the other axes only because breadth was pinned --
    the pessimistic-corner pattern for the third time in this programme, after
    E1.2's disjoint domains and E1.1c/d's overlap 0.
    """
    _, fleet, h, par, lat = prof
    c = par * 24.0
    out = []
    for b in range(1, bmax + 1):
        u = max(fleet * (1.0 - (1.0 - beta) ** b), 1.0)
        if DELETIONS_PER_DAY_MIN * u * h / c <= b <= DELETION_ARRIVAL_PER_DAY * (lat - u * h / c):
            out.append(b)
    return out


def breadth_threshold(prof, lo=0.01, hi=1.0, steps=200) -> float | None:
    """Largest cascade breadth at which the C3/C4 window is still non-empty."""
    best = None
    for i in range(steps + 1):
        beta = lo + (hi - lo) * i / steps
        if batch_window(prof, beta):
            best = beta
    return best


def probe_cost(n_regions: int) -> int:
    return n_regions * N_PROBE


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(f"\nE5.1  Joint feasibility   (dim={DIM})\n")

    print("  measuring C1 (tail-safe free rank) ...")
    c1 = {}
    for R, dr, ov in itertools.product(REGIONS, DOMAIN_RANKS, SUBSPACE_OVERLAP):
        c1[(R, dr, ov)] = tail_safe_free_rank(R, dr, ov, rng)

    print("  measuring C6 (worst-region over-forgetting) ...")
    c6 = {}
    for R, f, dp, dc, sh in itertools.product(REGIONS, DRAW_FRACTION, DECAY_POLICY,
                                              DECAY_RATE, SUPPORT_SHAPES):
        c6[(R, f, dp, dc, sh)] = float(np.mean([
            worst_over_forgetting(R, f, dp, dc, sh, np.random.default_rng(SEED + i))
            for i in range(2)]))

    rows, feasible = [], []
    for prof, R, dr, so, beta, cratio, f, b, dp, sh in itertools.product(
            PROFILES, REGIONS, DOMAIN_RANKS, SUBSPACE_OVERLAP, BREADTH_BETA,
            CONCURRENT_RATIO, DRAW_FRACTION, BATCH, DECAY_POLICY, SUPPORT_SHAPES):
        dc = DECAY_RATE[1]
        sub = R * dr / DIM
        free = c1[(R, dr, so)]
        over = c6[(R, f, dp, dc, sh)]
        dpd = deletions_per_day(prof, beta, b)
        lat = restore_latency_days(prof, b, beta)
        probes = probe_cost(R)
        # recompile cost is bounded by the draw cap -- E0.1 A6 -- so C5 holds
        # whenever f < 1.0; at f = 1.0 the draw is the whole provenance.
        # C1 governs the CONCURRENT fleet; C3/C4 govern the PROMOTED fleet.
        # E5.1 previously asked C1 as "free >= RANK_REQUEST" -- whether the
        # budget fits ONE adapter -- while C3/C4 used fleet = 64. One symbol,
        # two populations, 64x apart.
        promoted = prof[1]
        concurrent = max(1, min(int(round(cratio * promoted)),
                                DIM // RANK_REQUEST))   # section A's bound
        checks = {
            "C1 tail-safe free rank": free >= concurrent * RANK_REQUEST,
            "C2 probe budget": probes <= PROBE_BUDGET,
            "C3 deletion throughput": dpd >= DELETIONS_PER_DAY_MIN,
            "C4 restoration latency": lat <= RESTORE_LATENCY_MAX_DAYS,
            "C5 recompile bounded": f < 1.0,
            "C6 over-forgetting": over <= OVER_FORGET_MAX,
        }
        row = {"profile": prof[0], "beta": beta, "concurrent_ratio": cratio,
               "concurrent_fleet": concurrent, "promoted_fleet": promoted,
               "regions": R, "domain_rank": dr, "subscription": round(sub, 2),
               "subspace_overlap": so,
               "draw_fraction": f, "batch": b, "decay_policy": dp,
               "decay_rate": dc, "support_shape": sh,
               "free_rank": free, "over_forget_worst": round(over, 3),
               "deletions_per_day": round(dpd, 2), "latency_days": round(lat, 2),
               "probe_cost": probes,
               "failed": [k for k, v in checks.items() if not v],
               "feasible": all(checks.values())}
        rows.append(row)
        if row["feasible"]:
            feasible.append(row)

    n = len(rows)
    print(f"\n  swept {n} configurations; {len(feasible)} feasible"
          f" ({100*len(feasible)/n:.1f}%)\n")

    # F3 -- per-constraint ATTRIBUTION, not just a fail count. The useful output
    # is how much each constraint eliminates ALONE, because that is the repair
    # ordering: a constraint that only ever fails alongside others buys nothing
    # when fixed.
    binds, alone = {}, {}
    for r in rows:
        for k in r["failed"]:
            binds[k] = binds.get(k, 0) + 1
        if len(r["failed"]) == 1:
            alone[r["failed"][0]] = alone.get(r["failed"][0], 0) + 1
    print("  F3  per-constraint attribution -- the repair ordering")
    print(f"      {'constraint':<26}{'eliminates':>12}{'of sweep':>10}"
          f"{'ALONE':>8}{'-> fixing it gains':>20}")
    for k, v in sorted(binds.items(), key=lambda kv: -kv[1]):
        a = alone.get(k, 0)
        print(f"      {k:<26}{v:>12}{100*v/n:>9.1f}%{a:>8}{a:>20}")
    print("      'ALONE' is configs this constraint is the ONLY thing blocking.")
    print()
    print("      DO NOT RANK CONSTRAINTS ON THESE NUMBERS. Each constraint's")
    print("      elimination fraction depends on how finely ITS OWN drivers were")
    print("      swept. C1's drivers sit on a balanced 45-point grid (regions x")
    print("      domain_rank x subspace_overlap); C4's and C6's tolerances were")
    print("      single pinned values until this run. A constraint whose drivers")
    print("      are finely swept will always look less binding, because the grid")
    print("      hands it room to be satisfied. The curves below are the readable")
    print("      form; the column above is not.")

    # -- elimination curves: each constraint against its OWN tolerance ---------
    print("\n  Elimination as a curve against each constraint's own tolerance.")
    print("  This is the only form in which cross-constraint comparison means")
    print("  anything, and it is 'report the curve, not the endpoint' applied to")
    print("  the tolerances themselves.\n")
    curves = {}

    def frac_failing(pred):
        return sum(1 for r in rows if pred(r)) / len(rows)

    print(f"      {'C6 over-forget tol':<22}" + "".join(f"{t:>9.2f}" for t in
          (0.05, 0.10, 0.20, 0.35, 0.50)))
    c6c = [round(frac_failing(lambda r, t=t: r["over_forget_worst"] > t), 3)
           for t in (0.05, 0.10, 0.20, 0.35, 0.50)]
    curves["C6"] = c6c
    print(f"      {'  fraction eliminated':<22}" + "".join(f"{v:>9.3f}" for v in c6c))

    print(f"\n      {'C4 latency tol (days)':<22}" + "".join(f"{t:>9.0f}" for t in
          (3, 7, 14, 30, 60)))
    c4c = [round(frac_failing(lambda r, t=t: r["latency_days"] > t), 3)
           for t in (3, 7, 14, 30, 60)]
    curves["C4"] = c4c
    print(f"      {'  fraction eliminated':<22}" + "".join(f"{v:>9.3f}" for v in c4c))

    print(f"\n      {'C3 deletions/day floor':<22}" + "".join(f"{t:>9.2f}" for t in
          (0.25, 0.5, 1.0, 2.0, 4.0)))
    c3c = [round(frac_failing(lambda r, t=t: r["deletions_per_day"] < t), 3)
           for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
    curves["C3"] = c3c
    print(f"      {'  fraction eliminated':<22}" + "".join(f"{v:>9.3f}" for v in c3c))

    print(f"\n      {'C2 probe budget':<22}" + "".join(f"{t:>9.0f}" for t in
          (2000, 4000, 6000, 10000, 20000)))
    c2c = [round(frac_failing(lambda r, t=t: r["probe_cost"] > t), 3)
           for t in (2000, 4000, 6000, 10000, 20000)]
    curves["C2"] = c2c
    print(f"      {'  fraction eliminated':<22}" + "".join(f"{v:>9.3f}" for v in c2c))

    print("\n      Every one spans most of [0,1] across a defensible tolerance")
    print("      range, which is the point: at a single tolerance the ranking is")
    print("      a statement about the tolerances, not about the design.")

    # -- interiority: is the feasible set touching the edges of the box? -------
    print("\n  Is any feasible configuration INTERIOR to each axis?")
    print("  (all-corner feasibility is a warning about where I looked, not a")
    print("   measurement of density -- the true boundary may be outside the box)")
    axes = {"regions": REGIONS, "domain_rank": DOMAIN_RANKS,
            "subspace_overlap": SUBSPACE_OVERLAP,
            "beta": BREADTH_BETA, "concurrent_ratio": CONCURRENT_RATIO,
            "draw_fraction": DRAW_FRACTION, "batch": BATCH, "decay_rate": DECAY_RATE}
    interior = {}
    print(f"      {'axis':<20}{'swept':>22}{'feasible values':>26}{'interior?':>11}")
    for name, vals in axes.items():
        got = sorted({r[name] for r in feasible})
        lo, hi = min(vals), max(vals)
        inside = [v for v in got if lo < v < hi]
        interior[name] = {"swept": list(vals), "feasible_values": got,
                          "has_interior": bool(inside)}
        print(f"      {name:<20}{str(list(vals)):>22}{str(got):>26}"
              f"{('yes' if inside else 'EDGE'):>11}")

    # the correlated diagonal: subspace and provenance overlap move together
    # The correlated diagonal: subspace overlap helps C1 while cascade breadth
    # hurts C3/C4, and they are plausibly correlated -- regions sharing feature
    # space plausibly share source material. If so the feasible region is this
    # slice, not the product.
    diag = [r for r in rows if abs(r["subspace_overlap"] - r["beta"]) < 0.25]
    diag_ok = [r for r in diag if r["feasible"]]
    print(f"\n  F2  correlated diagonal (subspace overlap ~ provenance overlap):"
          f" {len(diag_ok)}/{len(diag)} feasible")

    print("\n  The C3-and-C4 window is a FIXED POINT in batch size, and it is a")
    print("  function of cascade breadth. Solved per profile:")
    print(f"      {'profile':<20}{'max beta':>10}{'window at b=0.30':>19}"
          f"{'window at measured beta':>26}")
    windows = []
    for prof in PROFILES:
        thr = breadth_threshold(prof)
        w30 = batch_window(prof, 0.30)
        wmeas = batch_window(prof, BREADTH_BETA[0])        # E0.2f's reconciled value
        windows.append({"profile": prof[0],
                        "max_breadth": round(thr, 3) if thr else None,
                        "window_at_0.30": [min(w30), max(w30)] if w30 else None,
                        "window_at_measured_beta": [min(wmeas), max(wmeas)] if wmeas else None})
        f30 = f"{min(w30)}..{max(w30)}" if w30 else "EMPTY"
        fm = f"{min(wmeas)}..{max(wmeas)}" if wmeas else "EMPTY"
        print(f"      {prof[0]:<20}{(f'{thr:.2f}' if thr else 'none'):>10}{f30:>19}{fm:>26}")
    print(f"\n    E0.2f's reconciled breadth is {BREADTH_BETA[0]:.3f}; E0.2d's world"
          f" gave 0.63-0.98. Swept, not fitted.")
    print("    (")
    print("    multiplicity achievable. So the binding question is not batch size:")
    print("    it is whether cascade breadth can be brought below the threshold at")
    print("    all, and no admission threshold reaches it. That is R9.")

    by_profile = {}
    for r in rows:
        by_profile.setdefault(r["profile"], []).append(r)
    print("\n  Feasible configurations per profile:")
    for prof in PROFILES:
        sub = by_profile[prof[0]]
        print(f"      {prof[0]:<20}{sum(x['feasible'] for x in sub):>6} / {len(sub)}")

    # split by decay policy -- the design as written vs weighting-rule compliant
    as_written = [r for r in rows if r["decay_policy"] == "use_based"]
    compliant = [r for r in rows if r["decay_policy"] == "stratified"]
    print(f"\n  feasible under the design as written (global use-based decay): "
          f"{sum(r['feasible'] for r in as_written)}/{len(as_written)}")
    print(f"  feasible with per-region decay (weighting rule applied):        "
          f"{sum(r['feasible'] for r in compliant)}/{len(compliant)}")

    f1 = len(feasible) > 0
    f2 = len(diag_ok) > 0

    if feasible:
        print("\n  Feasible configurations, widest-margin first:")
        seen, best = set(), []
        for r in sorted(feasible, key=lambda r: (-r["free_rank"], r["over_forget_worst"])):
            k = (r["profile"], r["regions"], r["subscription"], r["draw_fraction"],
                 r["batch"], r["decay_rate"])
            if k not in seen:
                seen.add(k)
                best.append(r)
            if len(best) == 8:
                break
        h = (f"      {'profile':<18}{'reg':>4}{'sub':>6}{'so':>5}{'beta':>6}{'draw':>6}"
             f"{'batch':>7}{'conc':>6}{'free':>6}{'over':>7}{'del/day':>9}{'lat':>7}")
        print(h)
        for r in best:
            print(f"      {r['profile']:<18}{r['regions']:>4}{r['subscription']:>6.2f}"
                  f"{r['subspace_overlap']:>5.1f}{r['beta']:>6.3f}"
                  f"{r['draw_fraction']:>6.2f}{r['batch']:>7}"
                  f"{r['concurrent_fleet']:>6}{r['free_rank']:>6}"
                  f"{r['over_forget_worst']:>7.3f}{r['deletions_per_day']:>9.2f}"
                  f"{r['latency_days']:>7.2f}")

    print(f"\n  F1 feasible region non-empty:            {'ok' if f1 else 'EMPTY'}")
    print(f"  F2 non-empty on the correlated diagonal: {'ok' if f2 else 'EMPTY'}")
    print("  F3 is a map, not a verdict -- it says where the next repair pays.\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e5_1_joint_feasibility.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_configs": n, "n_feasible": len(feasible),
         "binding_counts": binds, "diagonal_feasible": len(diag_ok),
         "diagonal_total": len(diag),
         "tolerances": {"probe_budget": PROBE_BUDGET,
                        "deletions_per_day_min": DELETIONS_PER_DAY_MIN,
                        "restore_latency_max_days": RESTORE_LATENCY_MAX_DAYS,
                        "over_forget_max": OVER_FORGET_MAX},
         "batch_windows": windows, "attribution": {"eliminates": binds, "alone": alone},
         "elimination_curves": curves,
         "interiority": interior,
         "F1_non_empty": bool(f1), "F2_diagonal_non_empty": bool(f2),
         "feasible_use_based": sum(r["feasible"] for r in as_written),
         "feasible_stratified_decay": sum(r["feasible"] for r in compliant),
         "feasible": feasible[:60]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
