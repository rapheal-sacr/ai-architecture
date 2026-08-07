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
REGIONS = (8, 16, 32)
DOMAIN_RANKS = (4, 8, 16)              # with REGIONS gives subscription 0.25x-4x
SUBSPACE_OVERLAP = (0.0, 0.4, 0.7)
PROVENANCE_OVERLAP = (0.1, 0.4, 0.7)   # -> cascade breadth, via E0.2d
DRAW_FRACTION = (0.25, 0.5, 0.75, 1.0)
BATCH = (1, 4, 16, 64)
DECAY_POLICY = ("use_based", "stratified")
# Swept, because fixing it at 0.25 made C6 fail in 95.8% of configurations
# and dominate the sweep: a 3-entry support with a 2-of-3 pass threshold
# loses ~16% of items to 25% decay INHERENTLY, whatever the draw does.
DECAY_RATE = (0.05, 0.15, 0.25)

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
                          rng) -> float:
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

    items = []
    for _ in range(30 * n_regions):
        r = int(rng.choice(n_regions, p=rates))
        pool = np.where(region == r)[0]
        items.append((r, set(rng.choice(pool, size=3, replace=False).tolist())))

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
                if len(s & before) / len(s) >= 0.5 and len(s & after) / len(s) < 0.5]
        per_region_rate.append(len(lost) / len(sel))
    return max(per_region_rate) if per_region_rate else 0.0


# ------------------------------------------------------- C2/C3/C4, computed
def cascade_breadth(prov_overlap: float) -> float:
    """E0.2d measured 63.1% at one card per entry rising to 98.8% at six."""
    return float(np.clip(0.60 + 0.45 * prov_overlap, 0.0, 1.0))


def union_size(fleet, prov_overlap, batch):
    return max(min(fleet, fleet * cascade_breadth(prov_overlap) * batch), 1.0)


def deletions_per_day(prof, prov_overlap: float, batch: int) -> float:
    _, fleet, h, par, _ = prof
    return (24.0 * par) * batch / (union_size(fleet, prov_overlap, batch) * h)


def restore_latency_days(prof, batch: int, prov_overlap: float) -> float:
    _, fleet, h, par, _ = prof
    fill = batch / DELETION_ARRIVAL_PER_DAY
    drain = union_size(fleet, prov_overlap, batch) * h / (par * 24.0)
    return fill + drain


def batch_window(prof, prov_overlap: float) -> tuple[float, float]:
    """The closed-form C3-and-C4 window on batch size, at saturated union."""
    _, fleet, h, par, lat = prof
    c = par * 24.0
    drain = fleet * h / c
    lo = DELETIONS_PER_DAY_MIN * fleet * h / c
    hi = DELETION_ARRIVAL_PER_DAY * (lat - drain)
    return lo, hi


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
    for R, f, dp, dc in itertools.product(REGIONS, DRAW_FRACTION, DECAY_POLICY,
                                          DECAY_RATE):
        c6[(R, f, dp, dc)] = float(np.mean([
            worst_over_forgetting(R, f, dp, dc, np.random.default_rng(SEED + i))
            for i in range(3)]))

    rows, feasible = [], []
    for prof, R, dr, so, po, f, b, dp, dc in itertools.product(
            PROFILES, REGIONS, DOMAIN_RANKS, SUBSPACE_OVERLAP, PROVENANCE_OVERLAP,
            DRAW_FRACTION, BATCH, DECAY_POLICY, DECAY_RATE):
        sub = R * dr / DIM
        free = c1[(R, dr, so)]
        over = c6[(R, f, dp, dc)]
        dpd = deletions_per_day(prof, po, b)
        lat = restore_latency_days(prof, b, po)
        probes = probe_cost(R)
        # recompile cost is bounded by the draw cap -- E0.1 A6 -- so C5 holds
        # whenever f < 1.0; at f = 1.0 the draw is the whole provenance.
        checks = {
            "C1 tail-safe free rank": free >= RANK_REQUEST,
            "C2 probe budget": probes <= PROBE_BUDGET,
            "C3 deletion throughput": dpd >= DELETIONS_PER_DAY_MIN,
            "C4 restoration latency": lat <= RESTORE_LATENCY_MAX_DAYS,
            "C5 recompile bounded": f < 1.0,
            "C6 over-forgetting": over <= OVER_FORGET_MAX,
        }
        row = {"profile": prof[0], "regions": R, "domain_rank": dr, "subscription": round(sub, 2),
               "subspace_overlap": so, "provenance_overlap": po,
               "draw_fraction": f, "batch": b, "decay_policy": dp,
               "decay_rate": dc,
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

    # F3 -- which constraint binds
    binds = {}
    for r in rows:
        for k in r["failed"]:
            binds[k] = binds.get(k, 0) + 1
    print("  F3  how often each constraint fails, across the whole sweep")
    print(f"      {'constraint':<26}{'fails':>8}{'of sweep':>11}")
    for k, v in sorted(binds.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<26}{v:>8}{100*v/n:>10.1f}%")

    # the correlated diagonal: subspace and provenance overlap move together
    diag = [r for r in rows
            if abs(r["subspace_overlap"] - (r["provenance_overlap"] - 0.0)) < 0.15]
    diag_ok = [r for r in diag if r["feasible"]]
    print(f"\n  F2  correlated diagonal (subspace overlap ~ provenance overlap):"
          f" {len(diag_ok)}/{len(diag)} feasible")

    print("\n  The C3-and-C4 window on batch size, per profile (saturated union).")
    print("  Fixing one profile is what produced EMPTY in the first two runs:")
    print(f"      {'profile':<20}{'need b >=':>11}{'need b <=':>11}{'window':>10}")
    windows = []
    for prof in PROFILES:
        lo, hi = batch_window(prof, 0.4)
        windows.append({"profile": prof[0], "b_min": round(lo, 2), "b_max": round(hi, 2),
                        "open": bool(hi >= lo)})
        print(f"      {prof[0]:<20}{lo:>11.2f}{hi:>11.2f}"
              f"{('open' if hi >= lo else 'EMPTY'):>10}")

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
        h = (f"      {'profile':<18}{'reg':>4}{'sub':>6}{'so':>5}{'po':>5}{'draw':>6}"
             f"{'batch':>7}{'rate':>6}{'free':>6}{'over':>7}{'del/day':>9}{'lat':>7}")
        print(h)
        for r in best:
            print(f"      {r['profile']:<18}{r['regions']:>4}{r['subscription']:>6.2f}"
                  f"{r['subspace_overlap']:>5.1f}{r['provenance_overlap']:>5.1f}"
                  f"{r['draw_fraction']:>6.2f}{r['batch']:>7}"
                  f"{r['decay_rate']:>6.2f}{r['free_rank']:>6}"
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
         "batch_windows": windows,
         "F1_non_empty": bool(f1), "F2_diagonal_non_empty": bool(f2),
         "feasible_use_based": sum(r["feasible"] for r in as_written),
         "feasible_stratified_decay": sum(r["feasible"] for r in compliant),
         "feasible": feasible[:60]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
