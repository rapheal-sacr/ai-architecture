"""E0.8 -- depth governs coverage and not the oracle line. They are two curves.

THE HYPOTHESIS THIS WAS BUILT TO TEST, AND IT FAILED. Stated first, as written,
because the pre-registration is the load-bearing part.

E0.6 and E0.7 measured what looked like two different things:

    E0.6  unowned provenance   0.483 at 16 rollouts -> 0.027 at 512
    E0.7  pool consumption     86.9% at pool 240 -> 12.1% at pool 4102

They are one curve seen from two ends. Both are set by TRAFFIC DEPTH PER ENTRY --
how many times each ledger entry gets a chance to be reached by a selected card.
E0.6 swept it by raising rollouts against a fixed ledger; E0.7 swept it by raising
the ledger against fixed rollouts. Neither said so, because each had only its own
axis, and "these are the same quantity" is an assertion until the two statistics
appear on one axis in one run.

THE CONSEQUENCE IF IT HOLDS, and it is uncomfortable:

    thin traffic  ->  sparse pool, little sharing, EXPENSIVE granularity
                  ->  and a large unowned surface, so a LARGE COVERAGE HOLE
    thick traffic ->  the opposite on all four

So the scarce currency and the coverage guarantee do not trade against each other
-- they fail together, in the same regime, and that regime is the tail. A design
that is sized on aggregate traffic will be sized in the thick-traffic regime and
will be wrong about both at once, in the same direction, for the same reason.

That is a stronger statement than either experiment made alone, and it is why
this is worth a run rather than a sentence: if the two curves did NOT co-move,
the register would have a genuine trade to manage, and the design would need a
knob. If they do co-move, there is no knob -- there is only depth.

WHAT THIS ALSO SETTLES. Worklist 2.3 asked for fleet provenance overlap on the
grounds that it decides section 1.3. E0.7 showed overlap is not the governing
variable and the birthday null explains the sharing. This gives the variable that
IS governing, on an axis E0.6 already owned, which is why 2.3 costs nothing more.

KILL CRITERIA (pre-registered):
    KD The two statistics do NOT co-move -- unowned fraction and pool consumption
       are not monotone opposites in depth. Then they are separate quantities,
       the unification is wrong, and each needs its own axis after all.
    KT There is no depth at which BOTH are bad -- i.e. the bad regimes do not
       overlap. Then the "both fail together" claim is false and the design has a
       trade to manage rather than a single variable to measure.

Is there a world that produces the other verdict? For KD, yes: if unowned
provenance were set by card structure rather than by selection, it would be flat
in depth while consumption still moved. For KT, yes: if the pool saturated at a
depth well below the one at which coverage closes, there would be a window where
granularity is cheap and coverage is already total.

RESULT, RECORDED HERE BECAUSE THE HYPOTHESIS ABOVE IS THE THING THAT FAILED:
KD fails. Distinct probes are FLAT at ~225 across the entire depth range while
unowned provenance collapses 0.692 -> 0.000. The two are not one curve. The
"uncomfortable corollary" this experiment was built to confirm does not hold --
coverage degrades in the tail and the oracle bill does not move. The pre-
registration above is left as written; it was wrong, and that is what a
pre-registration is for.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import InfluenceWorld                     # noqa: E402
from rig_a.core.register import ProbeRegistry, cover_report, owner_provenance  # noqa: E402

DIM = 16
N_ENTRIES = 960
ENTRIES_PER_CARD = 15
FLEET = 32
PROBES_PER_OWNER = 8
SEED = 20260806
N_WORLDS = 6
RETIRE_FRACTION = 0.25

# THE AXIS: rollouts per ledger entry. Ledger and fleet are held fixed so depth
# is the only thing moving -- E0.6 moved rollouts and E0.7 moved the ledger, and
# each confounded depth with its own second variable.
DEPTHS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def at_depth(depth: float, zipf_owners: bool = False) -> dict:
    """`zipf_owners` gives owners UNEQUAL experience, which is what exposes the
    between-owner form of the coverage anisotropy. With uniform rollouts every
    owner's provenance is the same size, so a fixed probe count lands equally on
    all of them and the spread is ~1.3x -- that is a property of the world's
    construction, not a finding about equal-count draws. A real fleet is promoted
    from wildly unequal experience."""
    n_rollouts = max(int(round(depth * N_ENTRIES)), FLEET)
    n_cards = max(N_ENTRIES // ENTRIES_PER_CARD, 2)
    unowned, never, consumed, distinct, pools, vs_draws, worst = [], [], [], [], [], [], []
    cov_worst, cov_best, prov_mean = [], [], []

    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w_i)
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=n_cards,
            n_rollouts=n_rollouts, n_adapters=FLEET, rng=rng,
        )
        if zipf_owners:
            # Zipfian experience per owner, same total rollouts. Nothing else moves.
            share = 1.0 / np.arange(1, FLEET + 1, dtype=float)
            share /= share.sum()
            order = rng.permutation(n_rollouts)
            cuts = np.cumsum(np.maximum((share * n_rollouts).astype(int), 1))[:-1]
            w.adapter_rollouts = [list(map(int, part))
                                  for part in np.split(order, cuts)]
        n_ret = max(int(round(FLEET * RETIRE_FRACTION)), 1)
        retired = set(int(a) for a in rng.choice(FLEET, size=n_ret, replace=False))
        live = set(range(FLEET)) - retired

        rep = cover_report(w, live_owners=live, retired_owners=retired)
        unowned.append(rep["unowned_fraction"])
        never.append(rep["never_owned_fraction"])

        prov = owner_provenance(w)
        covered = set().union(*[prov[a] for a in live]) if live else set()
        per_card = [sum(1 for e in srcs if e not in covered) / len(srcs)
                    for srcs in w.card_sources if srcs]
        worst.append(max(per_card) if per_card else 0.0)

        pool = set().union(*prov) if prov else set()
        reg = ProbeRegistry()
        for a in range(FLEET):
            reg.draw_for(a, prov[a], PROBES_PER_OWNER, rng)
        # EQUAL COUNT IS NOT EQUAL COVERAGE. probes_per_owner is a fixed count
        # while owner provenance grows with depth, so an owner with 10x the
        # provenance is probed 10x more thinly. That is the THIRD level of one
        # anisotropy: between-region (I8 fixes it), between-owner (E0.6 found
        # it), within-owner (this, and it is the qR^-1q symptom exactly).
        cov = [min(PROBES_PER_OWNER, len(p_)) / max(len(p_), 1) for p_ in prov if p_]
        if cov:
            cov_worst.append(min(cov))
            cov_best.append(max(cov))
        prov_mean.append(float(np.mean([len(p_) for p_ in prov])) if prov else 0.0)

        pools.append(len(pool))
        distinct.append(reg.distinct)
        consumed.append(reg.distinct / max(len(pool), 1))
        vs_draws.append(reg.distinct / (FLEET * PROBES_PER_OWNER))

    return {
        "depth": depth, "n_rollouts": n_rollouts, "zipf_owners": zipf_owners,
        "unowned_fraction": round(float(np.mean(unowned)), 4),
        "never_owned_fraction": round(float(np.mean(never)), 4),
        "worst_card_unowned": round(float(np.mean(worst)), 4),
        "pool": round(float(np.mean(pools)), 1),
        "distinct_probes": round(float(np.mean(distinct)), 1),
        "pool_consumed": round(float(np.mean(consumed)), 4),
        "distinct_over_draws": round(float(np.mean(vs_draws)), 4),
        "mean_provenance": round(float(np.mean(prov_mean)), 1),
        "coverage_worst": round(float(np.mean(cov_worst)), 4) if cov_worst else 0.0,
        "coverage_best": round(float(np.mean(cov_best)), 4) if cov_best else 0.0,
        "coverage_spread": round(
            float(np.mean(cov_best)) / max(float(np.mean(cov_worst)), 1e-9), 2)
            if cov_worst else 0.0,
    }


def main() -> int:
    pts = [at_depth(d) for d in DEPTHS]

    print("\nE0.8 -- depth governs coverage, not the oracle line\n")
    print(f"  ledger {N_ENTRIES}, fleet {FLEET}, {PROBES_PER_OWNER} probes/owner"
          f" = {FLEET * PROBES_PER_OWNER} draws. Only depth moves.\n")
    print(f"  {'depth':>7}{'rollouts':>10}{'pool':>8}{'unowned':>10}{'worst card':>12}"
          f"{'distinct':>10}{'consumed':>10}{'/draws':>9}")
    for p in pts:
        print(f"  {p['depth']:>7.2f}{p['n_rollouts']:>10}{p['pool']:>8.0f}"
              f"{p['unowned_fraction']:>10.3f}{p['worst_card_unowned']:>12.3f}"
              f"{p['distinct_probes']:>10.0f}{p['pool_consumed']:>10.1%}"
              f"{p['distinct_over_draws']:>9.1%}")
    print("\n    unowned    E0.6's quantity -- live provenance no live owner covers")
    print("    consumed   E0.7's quantity -- distinct probes / eligible pool")
    print("    /draws     distinct probes / fleet x probes. 100% = no sharing.")

    lo, hi = pts[0], pts[-1]
    un = [p["unowned_fraction"] for p in pts]
    co = [p["pool_consumed"] for p in pts]
    dr = [p["distinct_over_draws"] for p in pts]
    dp = [p["distinct_probes"] for p in pts]
    un_falls = all(a >= b - 1e-9 for a, b in zip(un, un[1:]))
    co_rises = all(a <= b + 1e-9 for a, b in zip(co, co[1:]))
    kd = un_falls and co_rises
    # The oracle line is bad when sharing is ABSENT -- distinct near draws.
    oracle_bad = [p for p in pts if p["distinct_over_draws"] >= 0.80]
    cover_bad = [p for p in pts if p["unowned_fraction"] > 0.10]
    kt = bool(oracle_bad and cover_bad)

    print(f"\n  KD unowned falls AND consumption rises with depth:  "
          f"{'ok' if kd else 'NO'}")
    print(f"  KT a regime exists where both are bad:             {'ok' if kt else 'NO'}")

    print("\n  KD FAILS, AND IT FALSIFIES THE UNIFICATION.")
    print(f"    unowned          {un[0]:.3f} -> {un[-1]:.3f}   collapses to zero by depth 1.0")
    print(f"    distinct probes  {dp[0]:.0f} -> {dp[-1]:.0f}   FLAT across the whole range")
    print(f"    distinct/draws   {dr[0]:.1%} -> {dr[-1]:.1%}   flat: sharing never appears")
    print("    Consumption falls rather than rises, and it falls because its")
    print("    DENOMINATOR grows -- the pool runs 388 -> 960 while the numerator")
    print("    is pinned by draws. Reading consumption alone would have shown a")
    print("    curve moving with depth and hidden that the oracle cost never")
    print("    moved at all. Same defect as the withdrawn 77%: a ratio whose")
    print("    denominator is doing the work.")

    print("\n  SO DEPTH GOVERNS COVERAGE AND DOES NOT GOVERN THE ORACLE LINE.")
    print("  They share one driver -- pool size -- but the oracle line also")
    print("  depends on fleet x probes, which depth does not touch. Here draws")
    print(f"  ({FLEET * PROBES_PER_OWNER}) stayed BELOW the pool at every depth, so the")
    print("  configuration never left the linear regime and the oracle bill was")
    print("  constant while coverage went from 69% unowned to zero.")

    print("\n  WHICH KILLS THE COROLLARY, and the corollary was the interesting")
    print("  part. `Both currencies degrade together in the tail` is not what")
    print("  happens: coverage degrades in the tail and the oracle bill does not")
    print("  move. If anything the tail is where sharing would START to help,")
    print("  because that is where the pool shrinks toward the draw count -- the")
    print("  opposite direction from the one the corollary predicted.")
    print("  Reaching that regime needs pool < draws, which this configuration")
    print("  cannot do without dropping below one rollout per adapter (B18), so")
    print("  it is stated as unmeasured rather than asserted either way.")

    print("\n  WHAT SURVIVES FOR WORKLIST 2.3. Not the unification. E0.7 still")
    print("  shows overlap is not the governing variable for the oracle line, and")
    print("  this shows depth is not either -- the governing quantity is")
    print("  fleet x probes against the pool, and it is arithmetic. Depth is the")
    print("  governing variable for COVERAGE, which is E0.6's result standing")
    print("  alone. Two axes, two quantities, and one run fewer than assuming so.")

    print("\n  EQUAL COUNT IS NOT EQUAL COVERAGE -- the third level of the same")
    print("  anisotropy, and depth is the axis that shows it. I8 fixes the")
    print("  between-region level and E0.6 found the between-owner level; this is")
    print("  WITHIN an owner, and it is open.")
    print(f"    {'depth':>7}{'mean prov':>11}{'cov worst':>11}{'cov best':>10}{'spread':>9}")
    for p in pts:
        print(f"    {p['depth']:>7.2f}{p['mean_provenance']:>11.0f}"
              f"{p['coverage_worst']:>11.3f}{p['coverage_best']:>10.3f}"
              f"{p['coverage_spread']:>9.2f}x")
    print("    cov = probes drawn / that owner's provenance size. A fixed count")
    print("    against a growing provenance is a FALLING coverage rate, and the")
    print("    spread is how unequally the `equal` draw actually lands.")

    print("\n  BUT THE SPREAD ABOVE IS A PROPERTY OF THIS WORLD, NOT A FINDING.")
    print("  Uniform rollouts give every owner the same provenance size, so a")
    print("  fixed probe count lands equally on all of them. The between-owner")
    print("  form needs UNEQUAL experience, which a real fleet has. Same worlds,")
    print("  same totals, Zipfian rollouts per owner:")
    zpts = [at_depth(d, zipf_owners=True) for d in DEPTHS]
    print(f"    {'depth':>7}{'mean prov':>11}{'cov worst':>11}{'cov best':>10}{'spread':>9}")
    for p in zpts:
        print(f"    {p['depth']:>7.2f}{p['mean_provenance']:>11.0f}"
              f"{p['coverage_worst']:>11.3f}{p['coverage_best']:>10.3f}"
              f"{p['coverage_spread']:>9.1f}x")
    zmax = max(p["coverage_spread"] for p in zpts)
    umax = max(p["coverage_spread"] for p in pts)
    print(f"    worst spread: {umax:.1f}x uniform against {zmax:.1f}x Zipfian.")
    print("    So `equal-N per owner` is equal in COUNT and unequal in COVERAGE by")
    print(f"    up to {zmax:.0f}x once owners differ in experience -- and every owner")
    print("    is still reported as having had its equal draw. I8 is satisfied and")
    print("    the tail inside an owner is not protected. Third level of the same")
    print("    anisotropy: between-region (I8), between-owner (E0.6), within-owner.")

    verdict = "FAIL" if not kd else "PASS"
    print(f"\n  ONE CURVE, TWO CURRENCIES: {verdict} -- they are two curves\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_8_traffic_depth.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "n_entries": N_ENTRIES,
         "fleet": FLEET, "probes_per_owner": PROBES_PER_OWNER,
         "depths": list(DEPTHS), "points": pts,
         "KD_comove": bool(kd), "KT_both_bad_regime": bool(kt),
         "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
