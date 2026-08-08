"""E2.3 -- Does the staged evaluation ladder buy affordability with invisible losses?

CLAIM UNDER TEST (Part I section 4, Part II section E):

    "Staged evaluation ladder. Cheap subset -> medium subset -> full evaluation,
     with full evaluation reserved for candidates beating the archive's
     second-highest score. This is what makes an archive affordable rather than
     a compute fire."

The last untested claim in the architecturally-consequential class. Its failure
mode is structurally invisible: a candidate rejected at the cheap rung is never
measured on the full one, so the losses do not appear in any metric the system
computes. The archive reports what it promoted, never what it discarded.

TWO WAYS THE CHEAP RUNG CAN DIFFER FROM THE FULL ONE, and they have opposite
consequences:

    SAMPLING NOISE     the cheap suite is a random subsample -- noisy but
                       UNBIASED. Losses are bad luck, spread evenly.
    COVERAGE BIAS      the cheap suite is built the natural way, by sampling
                       probes in proportion to traffic -- so it is BIASED. Losses
                       concentrate on candidates whose value sits in regions the
                       cheap suite barely covers.

The second is what the rest of this record predicts, because it is the weighting
rule at the Assay: promotion is a protection decision, and a traffic-weighted
cheap rung is exactly the forbidden weighting. If it holds, the staged ladder is
a NEW site in the enumeration, and the one where the loss is least recoverable,
since a rejected candidate leaves no trace to audit.

ARMS:
    full            evaluate every candidate on the full per-region suite. The
                    reference. Expensive by construction.
    ladder_random   cheap rung is an unbiased random probe subsample
    ladder_traffic  cheap rung is traffic-weighted -- the natural implementation

KILL CRITERIA (pre-registered):
    L1 fails if Spearman(cheap score, full score) < 0.7 -- the cheap rung does
       not preserve the ranking the expensive rung would give.
    L2 fails if the ladder's losses are CONCENTRATED in rare regions -- ratio of
       rare-specialist to frequent-specialist loss rate above 2.0 -- UNDER THE
       COVERAGE OBJECTIVE. The objective qualifier is forced, not decorative.
       Under the design's gate as written, value is the unweighted MEAN of
       per-region deltas, and a specialist worth 0.30 in one region of sixteen
       scores 0.019 against a generalist's 0.040. It is worth half a generalist
       and never enters the top decile, so dropping it is CORRECT and L2 as first
       written examined an empty population and could not find a defect. B14.
       Under I7's coverage objective -- how many regions a candidate lifts from
       below the competence floor to at or above it -- a rare specialist lifts
       one region, a frequent specialist lifts none (already above), and a
       generalist lifts the marginal one. All three populations are non-empty,
       which is what makes L2 scoreable at all. Both objectives are
       scored below, because WHICH ONE THE LADDER SHOULD SERVE is the actual
       question and the design never says.
    L3 reports the cost-versus-regret curve. Not pass/fail -- a ladder that saves
       80% of evaluations for 2% regret is a good trade and one that saves 20%
       for 30% regret is not, and the design states neither number.

Is there a world that produces the other verdict? For L2, yes, and it is the
control: `ladder_random` samples probes uniformly, so its losses should be
spread evenly across regions and its rare/frequent ratio should sit near 1.0. If
BOTH arms concentrate, the effect is the ladder itself rather than the
weighting, and the repair is different.

NOT PINNED, because pinned parameters have produced four of this programme's
rig bugs: cheap-rung size, probe noise and the archive's acceptance percentile
are all swept, and the headline is a curve.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_REGIONS = 16
N_CANDIDATES = 240
PROBES_PER_REGION = 40          # the FULL suite: equal per region, per the weighting rule
SEED = 20260806
N_SEEDS = 8

CHEAP_FRACTIONS = (0.05, 0.10, 0.25, 0.50)
NOISE_LEVELS = (0.02, 0.05, 0.10)
KEEP_FRACTION = 0.25            # cheap rung forwards the top quarter
RARE_CUTOFF = 0.5               # regions in the bottom half of traffic are "rare"

L1_RHO_MIN = 0.70
L2_CONCENTRATION_MAX = 2.0


COMPETENCE_FLOOR = 0.55         # I7: a region below this is under-covered


def make_world(rng):
    """Candidates with known per-region true value, and a known type."""
    rates = 1.0 / np.arange(1, N_REGIONS + 1, dtype=float)
    rates /= rates.sum()
    rare = np.argsort(rates)[: int(RARE_CUTOFF * N_REGIONS)]

    true = np.zeros((N_CANDIDATES, N_REGIONS))
    kind = []
    for c in range(N_CANDIDATES):
        roll = rng.random()
        if roll < 0.4:                                   # generalist
            true[c] = rng.normal(0.04, 0.01, size=N_REGIONS)
            kind.append("generalist")
        else:                                            # specialist
            r = int(rng.integers(0, N_REGIONS))
            true[c, r] = rng.normal(0.30, 0.05)
            kind.append("rare_spec" if r in rare else "freq_spec")
    # Current per-region competence, before any promotion. Frequent regions are
    # already well covered and rare ones are not -- which is what every finding
    # in this record predicts, and it is what makes coverage a live objective.
    current = 0.35 + 0.55 * (rates / rates.max())

    # OBJECTIVE 1 -- the gate as written: unweighted mean of per-region deltas.
    value_mean = true.mean(axis=1)

    # OBJECTIVE 2 -- I7 coverage: how many regions the candidate lifts from BELOW
    # the competence floor to at or above it.
    #
    # A first attempt used "gap closed below the floor, summed" and still favoured
    # generalists 0.56 to 0.166, because closing a little gap in fourteen regions
    # beats closing a lot in one -- so it discriminated no better than the mean
    # objective and L2 again had nothing to score. Counting regions LIFTED is the
    # quantity I7 actually asserts: a region below the floor is a defect, and what
    # matters is whether it is still one afterwards.
    below = current < COMPETENCE_FLOOR
    value_cov = ((below[None, :]) &
                 (current[None, :] + true >= COMPETENCE_FLOOR)).sum(axis=1).astype(float)

    return rates, np.array(kind), true, value_mean, value_cov


def observe(true_c, probes_per_region, rates, weighted, noise, rng):
    """Score a candidate on a suite of a given size and weighting."""
    if weighted:
        alloc = rng.multinomial(probes_per_region * N_REGIONS, rates)
    else:
        alloc = np.full(N_REGIONS, probes_per_region)
    seen = alloc > 0
    if not seen.any():
        return 0.0, 0
    est = np.where(seen, true_c + rng.normal(0, noise, N_REGIONS) /
                   np.sqrt(np.maximum(alloc, 1)), 0.0)
    # a suite reports the mean over the regions it actually covered
    return float(est[seen].mean()), int(alloc.sum())


def run(arm: str, cheap_frac: float, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rates, kind, true, value_mean, value_cov = make_world(rng)

    full_probes = PROBES_PER_REGION
    cheap_probes = max(int(cheap_frac * PROBES_PER_REGION), 1)
    cost = 0

    if arm == "full":
        scores = []
        for c in range(N_CANDIDATES):
            s, n = observe(true[c], full_probes, rates, False, noise, rng)
            scores.append(s); cost += n
        survivors = np.arange(N_CANDIDATES)
        final = np.array(scores)
        cheap_scores = final
    else:
        weighted = (arm == "ladder_traffic")
        cheap_scores = np.empty(N_CANDIDATES)
        for c in range(N_CANDIDATES):
            s, n = observe(true[c], cheap_probes, rates, weighted, noise, rng)
            cheap_scores[c] = s; cost += n
        keep = int(KEEP_FRACTION * N_CANDIDATES)
        survivors = np.argsort(-cheap_scores)[:keep]
        final = np.full(N_CANDIDATES, -np.inf)
        for c in survivors:
            s, n = observe(true[c], full_probes, rates, False, noise, rng)
            final[c] = s; cost += n

    picked = int(np.argmax(final))
    dropped = np.setdiff1d(np.arange(N_CANDIDATES), survivors)

    out = {"arm": arm, "cheap_fraction": cheap_frac, "noise": noise, "cost": cost,
           "rho_cheap_vs_true": float(spearmanr(cheap_scores, value_mean).statistic)}

    for label, value in (("mean", value_mean), ("cov", value_cov)):
        out[f"regret_{label}"] = float(value[int(np.argmax(value))] - value[picked])
        good = value >= np.quantile(value, 0.90)
        # The contrast is rare specialists against GENERALISTS, not against
        # frequent specialists. Under the coverage objective a frequent
        # specialist lifts zero regions -- its region is already above the floor
        # -- so it is never "good" and the rare-vs-frequent ratio compares
        # against an empty set. Generalists are good under both objectives, so
        # they are the only valid baseline. B15.
        lost = {}
        for t in ("rare_spec", "generalist"):
            sel = (kind == t) & good
            lost[t] = (float(np.isin(np.where(sel)[0], dropped).mean())
                       if sel.sum() >= 3 else float("nan"))
        out[f"lost_rare_{label}"] = lost["rare_spec"]
        out[f"lost_base_{label}"] = lost["generalist"]
        out[f"n_good_rare_{label}"] = int(((kind == "rare_spec") & good).sum())
        out[f"n_good_base_{label}"] = int(((kind == "generalist") & good).sum())
    return out


def agg(arm, cf, nz):
    rs = [run(arm, cf, nz, SEED + i) for i in range(N_SEEDS)]

    def m(k):
        return float(np.nanmean([r[k] for r in rs]))

    o = {"arm": arm, "cheap_fraction": cf, "noise": nz,
         "cost": round(m("cost")), "rho": round(m("rho_cheap_vs_true"), 3),
         "n_good_rare_cov": round(m("n_good_rare_cov"), 1)}
    for label in ("mean", "cov"):
        lr, lb = m(f"lost_rare_{label}"), m(f"lost_base_{label}")
        o[f"regret_{label}"] = round(m(f"regret_{label}"), 4)
        o[f"lost_rare_{label}"] = round(lr, 3)
        o[f"lost_base_{label}"] = round(lb, 3)
        o[f"conc_{label}"] = round(lr / max(lb, 1e-9), 2)
        o[f"n_good_base_{label}"] = round(m(f"n_good_base_{label}"), 1)
    return o


def main() -> int:
    print(f"\nE2.3  Does the staged ladder buy affordability with invisible losses?"
          f"\n      ({N_REGIONS} regions, {N_CANDIDATES} candidates,"
          f" {N_SEEDS} seeds)\n")

    base = agg("full", 1.0, 0.05)
    print(f"  reference: full evaluation on everything -- cost {base['cost']:,}\n")

    rows = [base]
    for arm in ("ladder_random", "ladder_traffic"):
        print(f"  {arm}")
        hdr = (f"    {'cheap':>7}{'noise':>7}{'saved':>8}{'rho':>8}"
               f"{'reg(mean)':>11}{'reg(cov)':>10}"
               f"{'lost rare':>11}{'lost base':>11}{'conc(cov)':>11}")
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        for cf in CHEAP_FRACTIONS:
            for nz in NOISE_LEVELS:
                r = agg(arm, cf, nz)
                rows.append(r)
                saved = 1.0 - r["cost"] / base["cost"]
                print(f"    {cf:>7.2f}{nz:>7.2f}{saved:>7.0%}{r['rho']:>8.3f}"
                      f"{r['regret_mean']:>11.4f}{r['regret_cov']:>10.4f}"
                      f"{r['lost_rare_cov']:>11.3f}{r['lost_base_cov']:>11.3f}"
                      f"{r['conc_cov']:>11.2f}")
        print()

    lad = [r for r in rows if r["arm"] != "full"]
    trf = [r for r in lad if r["arm"] == "ladder_traffic"]
    rnd = [r for r in lad if r["arm"] == "ladder_random"]

    # MANIPULATION CHECK -- the two arms must differ on the thing being varied,
    # and the population L2 examines must be non-empty. The first version failed
    # the second half silently: under the mean objective no rare specialist ever
    # entered the top decile, so L2 scored an empty set. B14.
    n_rare_cov = np.mean([r["n_good_rare_cov"] for r in lad])
    assert n_rare_cov >= 3, (
        f"manipulation check: only {n_rare_cov:.1f} rare specialists are 'good' "
        "under the coverage objective -- L2 has nothing to score")
    assert abs(np.mean([r["conc_cov"] for r in trf])
               - np.mean([r["conc_cov"] for r in rnd])) > 1e-9, \
        "manipulation did not take: weighted and unweighted cheap rungs identical"

    l1 = min(r["rho"] for r in lad) >= L1_RHO_MIN
    l2_trf = max(r["conc_cov"] for r in trf) <= L2_CONCENTRATION_MAX
    l2_rnd = max(r["conc_cov"] for r in rnd) <= L2_CONCENTRATION_MAX

    const_cov = len({r["regret_cov"] for r in lad}) <= 2
    if const_cov:
        print("  NOTE: regret under the coverage objective is near-constant across")
        print("  every ladder setting, because the FINAL pick is made on the")
        print("  full suite's MEAN score. That regret is a property of the GATE's")
        print("  objective, not of the ladder, and is reported here only so it is")
        print("  not mistaken for one.\n")

    print("  L3  the cost/regret trade, under BOTH objectives")
    print(f"      {'arm':<16}{'cheap':>7}{'saved':>8}{'reg(mean)':>11}{'reg(cov)':>10}")
    for r in sorted(lad, key=lambda x: x["cost"])[:2] + \
             sorted(lad, key=lambda x: -x["cost"])[:2]:
        print(f"      {r['arm']:<16}{r['cheap_fraction']:>7.2f}"
              f"{1-r['cost']/base['cost']:>7.0%}"
              f"{r['regret_mean']:>11.4f}{r['regret_cov']:>10.4f}")

    print(f"\n  L1 cheap rung preserves the ranking (rho >= {L1_RHO_MIN}):  "
          f"{'ok' if l1 else 'NO'}   worst rho {min(r['rho'] for r in lad):.3f}")
    print(f"\n  L2 losses not concentrated in rare regions, UNDER COVERAGE"
          f" (<= {L2_CONCENTRATION_MAX}x):")
    print(f"       ladder_random  (unbiased cheap rung -- the control): "
          f"{'ok' if l2_rnd else 'NO'}"
          f"   max {max(r['conc_cov'] for r in rnd):.2f}x")
    print(f"       ladder_traffic (traffic-weighted cheap rung):        "
          f"{'ok' if l2_trf else 'NO'}"
          f"   max {max(r['conc_cov'] for r in trf):.2f}x")
    print(f"\n     good-population sizes: rare specialists"
          f" {np.mean([r['n_good_rare_cov'] for r in lad]):.0f},"
          f" generalists {np.mean([r['n_good_base_cov'] for r in lad]):.0f}"
          f" -- both non-empty, which is what makes the ratio meaningful.")
    print(f"\n  VERDICT: {'PASS' if (l1 and l2_trf and l2_rnd) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e2_3_staged_ladder.json"
    out.write_text(json.dumps({"seed": SEED, "n_seeds": N_SEEDS,
                               "competence_floor": COMPETENCE_FLOOR, "rows": rows,
                               "L1_rank_preserved": bool(l1),
                               "L2_random_arm": bool(l2_rnd),
                               "L2_traffic_arm": bool(l2_trf)}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
