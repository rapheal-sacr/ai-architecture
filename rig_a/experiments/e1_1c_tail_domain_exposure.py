"""E1.1c -- Is E1.1b's PASS an artifact of traffic-weighted aggregation?

E1.1b reversed E1.1 by replacing an eigenvalue threshold with GPM's energy
criterion, and reported the budget sound: all four streams usable and safe, free
rank ample, empty only at 5x over-subscription.

Both halves of that verdict are traffic-weighted. `rank_for_energy` sums energy
over held-out queries drawn at DomainMixture's Zipfian visit rates, and
`leakage_by_rank` / `interference_by_rank` return a MEAN over those same queries.
So the criterion chooses r to protect frequent traffic, and the safety metric is
dominated by frequent traffic. A rare domain contributes ~1/(N H_N) of the mean
it is averaged into.

That is not a rounding concern, it is the definition of "energy". Protection is
allocated in proportion to frequency, so a rare domain's subspace lands in the
tail, below the cut, and is classified FREE -- and the metric that should catch
it cannot, because it is averaged over the distribution that made the domain
rare in the first place.

WHY THIS MATTERS MORE HERE THAN ELSEWHERE. L7's own admitted failure mode is
coverage bias, and the rare-domain tail is exactly the long-tail personal
knowledge the ledger exists to preserve. A budget that is safe on average and
unsafe in the tail is unsafe for WAM's stated purpose specifically.

Same G1 bar as E1.1b (interference <= 0.05 on held-out traffic). Asked per
domain, on probe sets of equal size for every domain, instead of on the traffic
average.

KILL CRITERIA (pre-registered):
    P1 fails if, at E1.1b's reported operating point rho = 0.95, any domain
       exceeds the interference limit. E1.1b's PASS is then conditional on
       aggregation, not on the mechanism.
    P2 fails if per-domain exposure is monotone in rarity (Spearman(rate,
       leakage) <= -0.5). Monotone means this is structural, not a tail event --
       a single unlucky domain would not correlate.
    P3 (the number that matters) the tail-safe retention: smallest rho at which
       EVERY domain clears the bar, and the free rank remaining there. If that
       free rank is far below E1.1b's, the budget survives but its headroom does
       not, and E1.2's predicted reclamation deadlock gets a real number.

FALSIFIER FOR MY OWN CLAIM. Random domain bases in dim 128 are near-orthogonal,
which is the pessimistic case for free rank. Real domains share structure, so
Panel C sweeps a shared-subspace fraction. If overlap rescues it, the finding is
about this generator and not about the criterion -- report either way.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.spectrum import (  # noqa: E402
    DomainMixture,
    RtEstimator,
    interference_by_rank,
    leakage_by_rank,
    rank_for_energy,
)

DIM = 128
N_OBS = 4000
N_CALIB = 500
N_TEST = 500
N_PROBE = 400
RANK_REQUEST = 8
INTERFERENCE_LIMIT = 0.05
SEED = 20260806
RETENTIONS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9995)
CONFIGS = ((8, 8), (16, 8), (32, 4))


def _probes(world, n_domains, domain_rank, rng):
    """Equal-size probe set per domain. Uniform over domains, not over traffic."""
    return {
        d: (rng.normal(size=(N_PROBE, domain_rank)) * world.within) @ world.bases[d].T
        for d in range(n_domains)
    }


def panel_a_b(n_domains: int, domain_rank: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    world = DomainMixture(DIM, n_domains, domain_rank, 1.0, rng)
    feats, labels = world.sample(N_OBS, rng)
    calib, calib_lab = world.sample(N_CALIB, rng)
    test, _ = world.sample(N_TEST, rng)
    est = RtEstimator(dim=DIM, lam=1.0)
    est.update(feats)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)
    probe = _probes(world, n_domains, domain_rank, rng)

    curve, tail_safe = [], None
    for rho in RETENTIONS:
        r = rank_for_energy(est, calib, rho)
        interfs = np.array([
            interference_by_rank(est, r, probe[d], readout, RANK_REQUEST, rng)
            for d in range(n_domains)
        ])
        leaks = np.array([leakage_by_rank(est, r, probe[d]) for d in range(n_domains)])
        row = {
            "retention": rho,
            "committed_rank": int(r),
            "free_rank": int(DIM - r),
            "interf_traffic": round(interference_by_rank(
                est, r, test, readout, RANK_REQUEST, rng), 4),
            "leak_traffic": round(leakage_by_rank(est, r, test), 4),
            "interf_worst": round(float(interfs.max()), 4),
            "leak_worst": round(float(leaks.max()), 4),
            "n_domains_failing": int(np.sum(interfs > INTERFERENCE_LIMIT)),
        }
        curve.append(row)
        if tail_safe is None and row["n_domains_failing"] == 0:
            tail_safe = row

    r95 = rank_for_energy(est, calib, 0.95)
    exposure = np.array([leakage_by_rank(est, r95, probe[d]) for d in range(n_domains)])
    rho_s = float(spearmanr(world.rates, exposure).statistic)

    # repair arms, both at rho = 0.95
    def arm(tag, e, r):
        i = np.array([interference_by_rank(e, r, probe[d], readout, RANK_REQUEST, rng)
                      for d in range(n_domains)])
        return {"arm": tag, "committed_rank": int(r), "free_rank": int(DIM - r),
                "interf_worst": round(float(i.max()), 4),
                "n_domains_failing": int(np.sum(i > INTERFERENCE_LIMIT))}

    repairs = [arm("baseline", est, r95)]
    repairs.append(arm("R-a worst-case rank", est,
                       max(rank_for_energy(est, probe[d], 0.95) for d in range(n_domains))))
    w = 1.0 / world.rates[labels]; w /= w.mean()
    est_b = RtEstimator(dim=DIM, lam=1.0); est_b.update(feats * np.sqrt(w)[:, None])
    wc = 1.0 / world.rates[calib_lab]; wc /= wc.mean()
    repairs.append(arm("R-b freq-balanced R_t", est_b,
                       rank_for_energy(est_b, calib * np.sqrt(wc)[:, None], 0.95)))

    at95 = next(c for c in curve if c["retention"] == 0.95)
    return {
        "n_domains": n_domains, "domain_rank": domain_rank,
        "rate_ratio": round(float(world.rates[0] / world.rates[-1]), 1),
        "curve": curve, "tail_safe": tail_safe,
        "spearman_rate_vs_leakage": round(rho_s, 3),
        "exposure_by_rate": [round(float(v), 4) for v in exposure],
        "repairs_at_095": repairs,
        "P1_pass": bool(at95["n_domains_failing"] == 0),
        "P2_pass": bool(rho_s > -0.5),
    }


def panel_c(n_domains: int, domain_rank: int, share: float, seed: int) -> dict:
    """Falsifier: does domain-subspace overlap rescue the criterion?"""
    rng = np.random.default_rng(seed)
    common = np.linalg.qr(rng.normal(size=(DIM, domain_rank)))[0]
    bases = []
    for _ in range(n_domains):
        own = np.linalg.qr(rng.normal(size=(DIM, domain_rank)))[0]
        bases.append(np.linalg.qr(share * common + (1 - share) * own)[0])
    within = np.arange(1, domain_rank + 1, dtype=float) ** -0.5
    rates = 1.0 / np.arange(1, n_domains + 1, dtype=float); rates /= rates.sum()

    def draw(n):
        lab = rng.choice(n_domains, size=n, p=rates)
        f = np.empty((n, DIM))
        for i, d in enumerate(lab):
            f[i] = bases[d] @ (rng.normal(size=domain_rank) * within)
        return f

    est = RtEstimator(dim=DIM, lam=1.0); est.update(draw(N_OBS))
    calib = draw(N_CALIB)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)
    probe = {d: (rng.normal(size=(N_PROBE, domain_rank)) * within) @ bases[d].T
             for d in range(n_domains)}
    overlap = float(np.mean([np.linalg.norm(bases[i].T @ bases[j]) ** 2 / domain_rank
                             for i in range(n_domains) for j in range(i + 1, n_domains)]))
    for rho in RETENTIONS:
        r = rank_for_energy(est, calib, rho)
        worst = max(interference_by_rank(est, r, probe[d], readout, RANK_REQUEST, rng)
                    for d in range(n_domains))
        if worst <= INTERFERENCE_LIMIT:
            return {"n_domains": n_domains, "share": share,
                    "mean_subspace_overlap": round(overlap, 3),
                    "tail_safe_retention": rho, "committed_rank": int(r),
                    "free_rank": int(DIM - r)}
    return {"n_domains": n_domains, "share": share,
            "mean_subspace_overlap": round(overlap, 3),
            "tail_safe_retention": None, "committed_rank": None, "free_rank": 0}


def main() -> int:
    print("\nE1.1c  Per-domain exposure under the energy criterion")
    print("       (E1.1b's G1 bar, asked per domain instead of on the traffic mean)\n")

    panels = [panel_a_b(nd, dr, SEED) for nd, dr in CONFIGS]
    for p in panels:
        print(f"--- {p['n_domains']} domains, rank {p['domain_rank']}, "
              f"rate ratio {p['rate_ratio']}x ---")
        h = (f"{'rho':>8}{'r':>5}{'free':>6}{'int_traffic':>13}{'int_worst':>11}"
             f"{'leak_worst':>12}{'failing':>10}")
        print(h); print("-" * len(h))
        for c in p["curve"]:
            print(f"{c['retention']:>8.4f}{c['committed_rank']:>5d}{c['free_rank']:>6d}"
                  f"{c['interf_traffic']:>13.4f}{c['interf_worst']:>11.4f}"
                  f"{c['leak_worst']:>12.4f}"
                  f"{c['n_domains_failing']:>7d}/{p['n_domains']}")
        print(f"  Spearman(traffic rate, domain leakage) at rho=0.95: "
              f"{p['spearman_rate_vs_leakage']:+.3f}")
        print(f"  P1 {'ok' if p['P1_pass'] else 'FAIL'}   "
              f"P2 {'ok' if p['P2_pass'] else 'FAIL'}")
        print("  repairs at rho=0.95:")
        for a in p["repairs_at_095"]:
            print(f"    {a['arm']:<24} r={a['committed_rank']:>4} free={a['free_rank']:>4} "
                  f"worst={a['interf_worst']:.4f}  "
                  f"{a['n_domains_failing']}/{p['n_domains']} failing")
        print()

    print("--- Panel C: does subspace overlap rescue it? ---")
    h = (f"{'domains':>8}{'share':>7}{'overlap':>9}{'tail-safe rho':>15}"
         f"{'r':>5}{'free':>6}{'fits rank-8':>13}")
    print(h); print("-" * len(h))
    cpanels = []
    for nd, dr in ((16, 8), (32, 4)):
        for share in (0.0, 0.4, 0.7):
            c = panel_c(nd, dr, share, SEED)
            cpanels.append(c)
            rs = str(c["tail_safe_retention"]) if c["tail_safe_retention"] else ">0.9995"
            print(f"{c['n_domains']:>8}{c['share']:>7.1f}{c['mean_subspace_overlap']:>9.3f}"
                  f"{rs:>15}{str(c['committed_rank']):>5}{c['free_rank']:>6}"
                  f"{('yes' if c['free_rank'] >= RANK_REQUEST else 'NO'):>13}")

    verdict = "FAIL" if any(not p["P1_pass"] for p in panels) else "PASS"
    print(f"\nP1 (no domain fails at rho=0.95): {verdict}")
    print("P3 tail-safe operating points:")
    for p in panels:
        ts = p["tail_safe"]
        if ts:
            print(f"  {p['n_domains']:>3} domains: rho={ts['retention']}, "
                  f"free rank {ts['free_rank']}/{DIM} "
                  f"({100*ts['free_rank']/DIM:.0f}% of dimension)")

    out = pathlib.Path(__file__).resolve().parents[2] / "results"
    out.mkdir(exist_ok=True)
    (out / "e1_1c_tail_domain_exposure.json").write_text(json.dumps(
        {"seed": SEED, "dim": DIM, "rank_request": RANK_REQUEST,
         "interference_limit": INTERFERENCE_LIMIT,
         "panels": panels, "overlap_sweep": cpanels, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
