"""E3.3 -- Is "maximize off-diagonal mass" a well-posed partition objective?

CLAIM UNDER TEST (Part III section 3):

    "Split/merge stops being a heuristic and becomes an optimization with an
     objective: **choose the partition that maximizes off-diagonal mass in the
     aggregated T.** A partition that makes T diagonal is, by construction, a
     partition that hides transfer that is actually there."

The diagnosis behind this is right -- a partition so coarse that all transfer
falls inside regions does hide it. But the objective built from that diagnosis
has no opposing force. Off-diagonal mass is what falls *between* regions, so
splitting any region can only move mass off the diagonal, never onto it. The
maximum is therefore the finest partition available: one region per task
signature, where every measurement is off-diagonal and the off-diagonal
fraction is exactly 1.0.

That is not an optimization with a solution; it is a drive toward atomisation,
and it is self-defeating for T's actual purpose. One region per signature means
one measurement per cell and no statistical strength anywhere.

Part III's own section 3 names the failure it is trying to avoid -- "too-fine
signatures make T diagonal and starve transfer" -- and then adopts an objective
that is monotone in fineness.

WHAT THE OBJECTIVE SHOULD BE. T exists to predict transfer: given a promotion
in region r, what happens in r'. So score a partition by how well the T it
induces predicts *held-out* off-target deltas. That has a real optimum. Too
coarse averages away structure; too fine has no data per cell. The bias and
the variance pull against each other and the true granularity sits between.

KILL CRITERIA (pre-registered):
    D1 fails if off-diagonal mass is monotone increasing in granularity, i.e.
       maximised by the finest partition rather than by the true one.
    D2 fails if the objective's argmax does not recover the planted structure.
    The proposed replacement passes only if its argmax lands on the true
    granularity.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_SIGNATURES = 192
TRUE_CLUSTERS = 4          # the planted structure the objective should recover
N_TAU = 30000              # observation-grain transfer measurements
WITHIN_MEAN = 0.06         # transfer is strong inside a true cluster
BETWEEN_MEAN = 0.005       # and weak across clusters
NOISE = 0.03
SEED = 20260806

GRANULARITIES = [1, 2, 4, 8, 16, 32, 64, 96, 192]


def true_cluster(sig: int) -> int:
    return sig * TRUE_CLUSTERS // N_SIGNATURES


def partition(sig: np.ndarray, k: int) -> np.ndarray:
    """Assign signatures to k regions, nested with the planted structure.

    Aligned so that k = TRUE_CLUSTERS recovers the truth exactly, k < that is a
    coarsening and k > that is a refinement. This is the friendliest possible
    setup for the objective under test -- the correct answer is in the search
    space and every candidate is nested with it.
    """
    return sig * k // N_SIGNATURES


def make_tau(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per measured off-target delta, at observation grain (Part III section 3)."""
    src = rng.integers(0, N_SIGNATURES, size=N_TAU)
    probe = rng.integers(0, N_SIGNATURES, size=N_TAU)
    same = np.array([true_cluster(s) == true_cluster(p) for s, p in zip(src, probe)])
    mean = np.where(same, WITHIN_MEAN, BETWEEN_MEAN)
    delta = rng.normal(mean, NOISE)
    return src, probe, delta


def aggregate_T(src, probe, delta, k) -> tuple[np.ndarray, np.ndarray]:
    """T[r, r'] computed on read against the current ontology version."""
    r_src, r_probe = partition(src, k), partition(probe, k)
    T = np.zeros((k, k))
    counts = np.zeros((k, k))
    np.add.at(T, (r_src, r_probe), delta)
    np.add.at(counts, (r_src, r_probe), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(counts > 0, T / np.maximum(counts, 1), np.nan)
    return T, counts


def offdiagonal_mass(T: np.ndarray, counts: np.ndarray) -> float:
    """The objective as Part III states it: fraction of transfer mass off the diagonal."""
    mass = np.nan_to_num(np.abs(T) * counts)
    if mass.sum() <= 0:
        return 0.0
    off = mass.sum() - np.trace(mass)
    return float(off / mass.sum())


def predictive_error(src, probe, delta, k, rng) -> float:
    """The proposed replacement: held-out mean squared error of T's predictions.

    Build T on a training half, predict each held-out delta by its cell, and
    back off to the global mean where a cell has no training data. A partition
    is good exactly insofar as the T it induces generalises.
    """
    n = len(delta)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]

    T, counts = aggregate_T(src[tr], probe[tr], delta[tr], k)
    global_mean = float(delta[tr].mean())

    r_src, r_probe = partition(src[te], k), partition(probe[te], k)
    pred = T[r_src, r_probe]
    seen = counts[r_src, r_probe] > 0
    pred = np.where(seen & ~np.isnan(pred), pred, global_mean)
    return float(np.mean((pred - delta[te]) ** 2))


def main() -> int:
    rng = np.random.default_rng(SEED)
    src, probe, delta = make_tau(rng)

    rows = []
    for k in GRANULARITIES:
        T, counts = aggregate_T(src, probe, delta, k)
        rows.append(
            {
                "regions": k,
                "offdiag_mass": round(offdiagonal_mass(T, counts), 4),
                "predictive_mse": round(
                    predictive_error(src, probe, delta, k, np.random.default_rng(SEED)), 6
                ),
                "mean_rows_per_cell": round(float(N_TAU / (k * k)), 1),
            }
        )

    best_offdiag = max(rows, key=lambda r: r["offdiag_mass"])["regions"]
    best_pred = min(rows, key=lambda r: r["predictive_mse"])["regions"]

    masses = [r["offdiag_mass"] for r in rows]
    monotone = all(b >= a - 1e-9 for a, b in zip(masses, masses[1:]))

    d1 = not monotone
    d2 = best_offdiag == TRUE_CLUSTERS
    fix_ok = best_pred == TRUE_CLUSTERS

    hdr = f"{'regions':>9}{'offdiag mass':>15}{'predictive MSE':>17}{'rows/cell':>12}"
    print(
        f"\nE3.3  Is 'maximize off-diagonal mass' well posed?"
        f"   ({N_SIGNATURES} signatures, {TRUE_CLUSTERS} planted clusters)\n"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = ""
        if r["regions"] == TRUE_CLUSTERS:
            mark = "   <- true structure"
        if r["regions"] == best_offdiag:
            mark += "   <- objective's argmax"
        print(
            f"{r['regions']:>9}{r['offdiag_mass']:>15.4f}"
            f"{r['predictive_mse']:>17.6f}{r['mean_rows_per_cell']:>12.1f}{mark}"
        )

    print(
        f"\n  off-diagonal mass is monotone in fineness: {monotone}"
        f"\n  argmax of the stated objective:  {best_offdiag} regions"
        f"\n  argmin of held-out MSE (proposed replacement): {best_pred} regions"
        f"\n  planted truth: {TRUE_CLUSTERS} regions"
    )
    print(
        f"\n  D1 objective not monotone in fineness: {'ok' if d1 else 'NO'}"
        f"\n  D2 objective recovers planted structure: {'ok' if d2 else 'NO'}"
        f"\n  replacement recovers planted structure:  {'ok' if fix_ok else 'NO'}"
        f"\n\n  VERDICT: {'PASS' if (d1 and d2) else 'FAIL'}\n"
    )

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e3_3_offdiagonal_degeneracy.json"
    out.write_text(
        json.dumps(
            {
                "seed": SEED,
                "true_clusters": TRUE_CLUSTERS,
                "rows": rows,
                "objective_argmax": best_offdiag,
                "replacement_argmin": best_pred,
                "monotone_in_fineness": bool(monotone),
                "D1_not_monotone": bool(d1),
                "D2_recovers_truth": bool(d2),
                "replacement_recovers_truth": bool(fix_ok),
                "verdict": "PASS" if (d1 and d2) else "FAIL",
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
