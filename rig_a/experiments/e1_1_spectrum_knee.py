"""E1.1 -- Is `occupied rank = #{sigma_k > eps}` a well-posed budget?

CLAIM UNDER TEST (Part II section A, and the whole L7 subspace budget):

    "Occupied rank = #{sigma_k > eps}. The free basis is the eigenvectors with
     sigma_k <= eps. That is where an adapter may write without touching
     committed competence."

    "'Cap active count' becomes a computed constraint... The cap stops being a
     guessed integer and becomes a constraint you can check."

Both sentences require a threshold eps that is a property of the *system*
rather than a knob. Two things have to hold:

    (K1) THRESHOLD STABILITY. There is a range of eps, at least one decade
         wide, over which the occupied-rank count barely moves. Without a
         plateau, "occupied rank" is not a measurement of the system; it is a
         restatement of whatever eps someone picked, and the "computed
         constraint" is a guessed integer wearing a hat.

    (K2) FREE MEANS FREE. Held-out real queries must carry negligible energy
         in the sub-eps subspace. Writing into the free basis is trivially
         harmless to vectors lying exactly in the committed span -- that is
         linear algebra, not evidence. The real question is whether live
         traffic stays in that span.

KILL CRITERIA (pre-registered):
    K1 fails if no decade-wide window has mean elasticity < 0.10
       (i.e. rank moves more than ~26% per decade of eps everywhere).
    K2 fails if held-out leakage > 5%, or if a free-basis adapter write
       perturbs held-out outputs by more than 5% relative.

If either fails on the realistic stream, the subspace budget does not carry
the weight Part II puts on it, and two of its downstream claims -- computed
interference caps and computed reclamation sets -- lose their foundation.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.spectrum import (  # noqa: E402
    RtEstimator,
    free_subspace_leakage,
    interference_from_free_write,
    rank_elasticity,
    rank_for_energy,
    stream_bimodal,
    DomainMixture,
    stream_exponential,
    stream_power_law,
)

DIM = 128
N_OBS = 4000
N_HELDOUT = 500
SEED = 20260806

PLATEAU_ELASTICITY = 0.10   # K1 threshold
LEAKAGE_LIMIT = 0.05        # K2 threshold
INTERFERENCE_LIMIT = 0.05   # K2 threshold


def widest_plateau(
    eps_grid: np.ndarray, elasticity: np.ndarray, ranks: np.ndarray, dim: int
) -> float:
    """Width, in decades, of the widest contiguous *interior* run of flat elasticity.

    Saturated regions must be excluded. At very small eps every eigenvalue
    clears the bar and rank pins at `dim`; at very large eps none do and rank
    pins at 1. Both regions have zero elasticity and neither is a knee -- a
    power-law spectrum with no knee at all still shows multi-decade flatness at
    both ends. Counting those was the bug in the first run of this experiment.
    """
    interior = (ranks > 1) & (ranks < dim)
    flat = (elasticity < PLATEAU_ELASTICITY) & interior
    best = run_start = 0.0
    in_run = False
    log_eps = np.log10(eps_grid)
    for i, ok in enumerate(flat):
        if ok and not in_run:
            in_run, run_start = True, log_eps[i]
        elif not ok and in_run:
            in_run = False
            best = max(best, log_eps[i - 1] - run_start)
    if in_run:
        best = max(best, log_eps[-1] - run_start)
    return float(best)


def evaluate(name: str, feats: np.ndarray, heldout: np.ndarray, rng) -> dict:
    est = RtEstimator(dim=DIM, lam=1.0)
    est.update(feats)
    sigma, _ = est.eig()

    eps_grid = np.logspace(np.log10(sigma.max()) - 6, np.log10(sigma.max()), 240)
    ranks = np.array([np.sum(sigma > e) for e in eps_grid])
    elasticity = rank_elasticity(est, eps_grid)
    plateau = widest_plateau(eps_grid, elasticity, ranks, DIM)

    # Probe K2 at the eps a practitioner would actually choose: the point of
    # maximum spectral curvature -- the "knee", if one exists at all.
    log_sigma = np.log10(np.maximum(sigma, sigma.max() * 1e-12))
    curvature = np.abs(np.gradient(np.gradient(log_sigma)))
    knee_idx = int(np.argmax(curvature[2:-2])) + 2
    eps_knee = float(sigma[knee_idx])

    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)
    leakage = free_subspace_leakage(est, eps_knee, heldout)
    interference = interference_from_free_write(
        est, eps_knee, heldout, readout, rank_request=8, rng=rng
    )

    # The eps-free view of the same question: how much rank must stay committed
    # to hold 95% of held-out traffic energy, and what is left over to allocate.
    r95 = rank_for_energy(est, heldout, 0.95)

    rank_hi = int(np.sum(sigma > eps_knee * 10))
    rank_lo = int(np.sum(sigma > eps_knee / 10))
    swing = rank_lo / max(rank_hi, 1)

    k1 = plateau >= 1.0
    k2 = leakage <= LEAKAGE_LIMIT and interference <= INTERFERENCE_LIMIT

    return {
        "stream": name,
        "plateau_decades": round(plateau, 3),
        "min_elasticity": round(float(elasticity.min()), 4),
        "eps_knee_rel": round(eps_knee / float(sigma.max()), 8),
        "rank_at_knee": int(np.sum(sigma > eps_knee)),
        "rank_at_eps_div10": rank_lo,
        "rank_at_eps_x10": rank_hi,
        "rank_swing_per_2_decades": round(float(swing), 2),
        "r95_committed": int(r95),
        "free_rank_at_95pct": int(DIM - r95),
        "heldout_leakage": round(leakage, 4),
        "free_write_interference": round(interference, 4),
        "K1_threshold_stable": bool(k1),
        "K2_free_means_free": bool(k2),
        "verdict": "PASS" if (k1 and k2) else "FAIL",
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []

    f = stream_bimodal(N_OBS, DIM, n_committed=24, floor=0.02, rng=rng)
    h = stream_bimodal(N_HELDOUT, DIM, n_committed=24, floor=0.02, rng=rng)
    rows.append(evaluate("bimodal (shape the design assumes)", f, h, rng))

    f = stream_exponential(N_OBS, DIM, rate=0.12, rng=rng)
    h = stream_exponential(N_HELDOUT, DIM, rate=0.12, rng=rng)
    rows.append(evaluate("exponential (soft knee)", f, h, rng))

    f = stream_power_law(N_OBS, DIM, alpha=1.0, rng=rng)
    h = stream_power_law(N_HELDOUT, DIM, alpha=1.0, rng=rng)
    rows.append(evaluate("power law alpha=1.0", f, h, rng))

    world = DomainMixture(DIM, n_domains=12, domain_rank=10, alpha=1.0, rng=rng)
    f, _ = world.sample(N_OBS, rng)
    h, _ = world.sample(N_HELDOUT, rng)   # same world, genuinely held out
    rows.append(evaluate("domain mixture (realistic WAM traffic)", f, h, rng))

    hdr = (
        f"{'stream':<40}{'plateau':>8}{'swing':>7}{'r95':>6}{'free':>6}"
        f"{'leak':>8}{'interf':>8}{'K1':>5}{'K2':>5}  verdict"
    )
    print(f"\nE1.1  Is occupied rank a well-posed budget?   (dim={DIM})\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['stream']:<40}{r['plateau_decades']:>8.2f}"
            f"{r['rank_swing_per_2_decades']:>7.1f}"
            f"{r['r95_committed']:>6}{r['free_rank_at_95pct']:>6}"
            f"{r['heldout_leakage']:>8.3f}{r['free_write_interference']:>8.3f}"
            f"{'ok' if r['K1_threshold_stable'] else 'no':>5}"
            f"{'ok' if r['K2_free_means_free'] else 'no':>5}"
            f"  {r['verdict']}"
        )
    print(
        f"\n  plateau  widest interior decade-range where rank is insensitive to eps"
        f" (K1 needs >= 1.00)"
        f"\n  swing    rank(eps/10) / rank(eps*10) -- how far the 'computed' budget"
        f" moves for +/-1 decade of eps"
        f"\n  r95      committed rank needed to hold 95% of held-out query energy"
        f"\n  free     dim - r95: the rank actually available to allocate"
        f"\n  leak     held-out query energy sitting in the sub-eps 'free' subspace"
        f" (K2 needs <= {LEAKAGE_LIMIT})"
        f"\n  interf   held-out output shift from a 10%-magnitude free-basis write"
        f" (K2 needs <= {INTERFERENCE_LIMIT})\n"
    )

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_1_spectrum_knee.json"
    out.write_text(json.dumps({"seed": SEED, "dim": DIM, "n_obs": N_OBS, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
