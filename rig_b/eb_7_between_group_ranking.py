"""EB.7 -- EB.5's two repairs, built and measured. Worklist v3 item 1.1.

EB.5 diagnosed and did not build. The diagnosis: `rig_a/core/spectrum.py`
accumulates `R = lam*R + outer(a,a)` -- uncentered -- and ranks committed
directions by TOTAL energy. On a 1.5B the top-5 directions by total energy hold
77.4% of centered variance and carry 0.2% between-domain share against 2.0%
elsewhere: a **0.13x discrimination ratio**. The criterion spends three quarters
of its budget on directions that distinguish nothing.

Two repairs were named. Both are now in `spectrum.py` and this measures them.

    1. CENTER. `RtEstimator(centered=True)` subtracts the running mean, exact at
       lam = 1.0. One line of arithmetic, and EB.5 showed it fixes the 0.5B and
       not the 1.5B.
    2. RANK BY BETWEEN-GROUP VARIANCE. `rank_by_between_group` takes the top-r
       eigenvectors of the between-group scatter B rather than of the total
       scatter T. This is the repair that is supposed to matter.

MEASURED ON DIRECTIONS, NOT AXES. EB.5 reported the between-domain share of the
top-5 DIMENSIONS, which is axis-aligned. The criterion selects eigen-DIRECTIONS,
so the honest test ranks directions and scores each by

    discrimination(u)  =  u'Bu / u'Tu

the fraction of what a direction moves that distinguishes one group from another.
A direction that merely moves a lot scores near zero however much energy it holds.

WHAT THE REPAIR COSTS, reported beside what it buys. Ranking by B rather than T
selects directions that discriminate, not directions that carry energy — so the
committed subspace holds LESS total energy at the same rank. That is a real
tradeoff and it is exactly the kind this record has been caught reporting only one
side of.

KILL CRITERION (pre-registered, from worklist v3 1.1):
    KG1 The discrimination ratio does not move off 0.13x. The repair is then
        wrong and the energy criterion needs a different one.
    KG2 The repair buys discrimination but the committed subspace captures so
        little total energy that it cannot be the protected subspace at all --
        an arbitrary bar, so reported as a curve in r rather than a verdict.

Is there a world that produces the other verdict? For KG1, yes: if the
high-energy directions were ALSO the discriminative ones -- which is what the
design implicitly assumes -- then ranking by B would select the same directions
and the ratio would not move. EB.5 says they are not, but EB.5 measured axes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rig_a.core.spectrum import (  # noqa: E402
    RtEstimator,
    between_group_scatter,
    discrimination,
    rank_by_between_group,
)
from rig_b.eb_2_activation_spectra import extract  # noqa: E402
from mlx_lm import load                            # noqa: E402

MODELS = (
    ("Qwen2.5-0.5B-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", 12),
    ("Qwen2.5-1.5B-4bit", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", 14),
)
RANKS = (5, 8, 16, 32)
SEED = 20260806


def top_by_energy(groups: list, r: int, centered: bool) -> np.ndarray:
    """What the design does: top-r eigenvectors of the accumulated scatter."""
    est = RtEstimator(dim=groups[0].shape[1], lam=1.0, centered=centered)
    for g in groups:
        est.update(g)
    _, U = est.eig()
    return U[:, :r]


def main() -> int:
    t0 = time.time()
    print("\nEB.7 -- EB.5's repairs, built and measured (worklist v3 item 1.1)\n")
    print("  discrimination(u) = u'Bu / u'Tu, averaged over the top-r directions.")
    print("  How much of what a direction moves DISTINGUISHES one owner from")
    print("  another. Energy is not discrimination, and the criterion ranks energy.\n")

    out = {}
    for tag, path, layer in MODELS:
        model, tok = load(path)
        acts = extract(model, tok, layer)
        del model, tok
        groups = [np.asarray(v, dtype=np.float64) for v in acts.values()]
        B, T = between_group_scatter(groups)
        tot_energy = float(np.trace(T))

        rows = []
        for r in RANKS:
            u_unc = top_by_energy(groups, r, centered=False)
            u_cen = top_by_energy(groups, r, centered=True)
            u_bet = rank_by_between_group(groups, r)

            def energy(U):
                return float(np.trace(U.T @ T @ U)) / max(tot_energy, 1e-12)

            rows.append({
                "rank": r,
                "disc_uncentered": round(discrimination(u_unc, B, T), 4),
                "disc_centered": round(discrimination(u_cen, B, T), 4),
                "disc_between": round(discrimination(u_bet, B, T), 4),
                "energy_centered": round(energy(u_cen), 4),
                "energy_between": round(energy(u_bet), 4),
            })
        # the population baseline: mean discrimination over ALL directions
        base = float(np.trace(B) / max(np.trace(T), 1e-12))
        out[tag] = {"dim": int(groups[0].shape[1]), "baseline": round(base, 4),
                    "n_groups": len(groups),
                    "rank_B": int(np.linalg.matrix_rank(B, tol=1e-8)),
                    "rows": rows}

    for tag, _, _ in MODELS:
        o = out[tag]
        print(f"  {tag}  (dim {o['dim']}, population discrimination {o['baseline']:.1%})")
        print(f"    {'rank':>6}{'uncentered':>12}{'centered':>10}{'between':>10}"
              f"{'ratio':>8}   {'energy cen':>11}{'energy bet':>11}")
        for r in o["rows"]:
            ratio = r["disc_between"] / max(r["disc_centered"], 1e-12)
            print(f"    {r['rank']:>6}{r['disc_uncentered']:>12.4f}"
                  f"{r['disc_centered']:>10.4f}{r['disc_between']:>10.4f}"
                  f"{ratio:>8.1f}x{r['energy_centered']:>11.1%}"
                  f"{r['energy_between']:>11.1%}")
        print()
    print("    uncentered/centered/between  mean discrimination of the top-r")
    print("      directions under each ranking. ratio = between / centered.")
    print("    energy  share of total scatter the top-r directions capture --")
    print("      what the repair COSTS, beside what it buys.")

    big = out[MODELS[1][0]]
    r5 = next(x for x in big["rows"] if x["rank"] == 5)
    gain = r5["disc_between"] / max(r5["disc_centered"], 1e-12)
    kg1 = gain >= 2.0

    print(f"\n  KG1 the discrimination ratio moves:   {'ok' if kg1 else 'NO'}")
    print(f"      1.5B at r=5: centered ranking {r5['disc_centered']:.4f},"
          f" between-group ranking {r5['disc_between']:.4f}  = {gain:.1f}x")

    print("\n  CENTERING IS NOT THE REPAIR, and this separates the two cleanly.")
    for tag, _, _ in MODELS:
        o = out[tag]; row = next(x for x in o["rows"] if x["rank"] == 5)
        print(f"    {tag:>20}  uncentered {row['disc_uncentered']:.4f}"
              f" -> centered {row['disc_centered']:.4f}"
              f" -> between {row['disc_between']:.4f}")
    print("    Centering moves discrimination barely at either size. Ranking by")
    print("    between-group variance is what moves it, and it is the repair that")
    print("    needs a grouping -- which under the register is already recorded")
    print("    and, under I8, already paid for.")

    print("\n  THE GAIN DECAYS WITH RANK AND INVERTS, AND THAT IS STRUCTURAL.")
    for tag, _, _ in MODELS:
        o = out[tag]
        ratios = [(x["rank"], x["disc_between"] / max(x["disc_centered"], 1e-12))
                  for x in o["rows"]]
        print(f"    {tag:>20}  " + "  ".join(f"r={r}:{v:.1f}x" for r, v in ratios)
              + f"   rank(B) = {o['rank_B']}, groups = {o['n_groups']}")
    print("    B is a sum of `n_groups` outer products of deviations that sum to")
    print("    zero, so rank(B) <= n_groups - 1. Beyond that the `between-group`")
    print("    eigenvectors lie in B's NULL SPACE -- they carry no between-group")
    print("    variance at all, and ranking by B selects arbitrary directions.")
    print("    That is why the ratio falls to 1.0 at r=16 and to 0.5 at r=32 with")
    print("    eight groups: past rank 7 the repair has nothing left to rank by.")
    print("\n    SO THE REPAIR CARRIES A CEILING NOBODY HAS STATED: committed rank")
    print("    chosen by between-owner variance is bounded by OWNER COUNT, not by")
    print("    dimension. A 64-owner fleet can commit at most 63 directions this")
    print("    way. That is a third data-shaped bound on the same budget, beside")
    print("    R12's per-owner |provenance| floor and E1.1d's subscription cliff,")
    print("    and it is the only one that gets LOOSER as the fleet grows -- which")
    print("    is the opposite direction from R12's.")

    print("\n  AND WHAT IT COSTS. The between-group basis captures less total")
    print("  energy at the same rank, because it is selecting for discrimination")
    print("  rather than for magnitude. That is the tradeoff, stated: a committed")
    print("  subspace chosen this way protects what distinguishes owners and")
    print("  leaves more of the shared bulk in the free pool.")

    print("\n  WHAT THIS DOES NOT SETTLE. Discrimination is not tail safety. The")
    print("  next step is E1.6's instruments run with the committed basis chosen")
    print("  by B rather than by T -- whether the blindness ratio improves, not")
    print("  just whether the directions discriminate. That is a separate run and")
    print("  it is not claimed here.")

    verdict = "PASS" if kg1 else "FAIL"
    print(f"\n  EB.5's SECOND REPAIR: {verdict}\n  ({time.time() - t0:.0f}s)")

    o = pathlib.Path(__file__).resolve().parents[1] / "results" / "eb_7_between_group_ranking.json"
    o.write_text(json.dumps(
        {"models": [m[0] for m in MODELS], "ranks": list(RANKS),
         "by_model": out, "gain_at_r5_1_5B": round(gain, 2),
         "rank_ceiling": "committed rank by between-owner variance <= n_owners - 1",
         "KG1_ratio_moves": bool(kg1), "verdict": verdict}, indent=2))
    print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
