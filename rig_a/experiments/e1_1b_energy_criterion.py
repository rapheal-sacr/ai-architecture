"""E1.1b -- Does the subspace budget survive its STRONGEST faithful reading?

E1.1 tested the mechanism as Part II section A literally states it:

    "occupied rank = #{sigma_k > eps}"

and found no stable eps. But that is not the strongest reading available. The
design points at GPM directly -- "truncated SVD over a small activation buffer
-- what GPM actually does" -- and GPM does not threshold eigenvalues. It keeps
the top-r directions capturing a chosen fraction of feature energy. That
reparameterisation replaces an absolute threshold on eigenvalues with a
relative one on cumulative energy, and it is strictly better behaved: it has no
scale, it cannot be knocked off by a uniformly mis-scaled spectrum, and it is
the standard practice this design is drawing on.

`rank_for_energy` implementing exactly this criterion was already in
spectrum.py during E1.1, documented as "the parameter-free version of the
budget question", computed, and reported as a diagnostic -- while the verdict
was decided entirely at the curvature knee. That is the wrong way round. E1.1's
FAIL is therefore a verdict on the weaker reading, and this experiment decides
whether the failure survives the stronger one.

WHY LEAKAGE ALONE CANNOT SETTLE IT. Choosing r to capture rho of a set's energy
guarantees (1 - rho) leakage on that same set. Measuring leakage there would be
circular. So r is chosen on a CALIBRATION sample and everything is measured on
a fresh TEST sample. The calibration/test gap is itself the finding about
whether the spectrum estimate generalises.

THE QUESTION THAT ACTUALLY MATTERS. Under this criterion the budget is a curve,
not a number: raise rho and free rank shrinks. So the real question is whether
any point on that curve is simultaneously usable and safe:

    usable  free rank >= RANK_REQUEST, or there is nothing to allocate
    safe    interference on fresh traffic <= INTERFERENCE_LIMIT

KILL CRITERIA (pre-registered):
    G1 fails if NO retention level rho yields free rank >= RANK_REQUEST with
       interference <= 5% on held-out traffic. That is the budget being
       unusable under its best reading.
    G2 fails if test leakage exceeds calibration leakage by more than 2x at
       rho = 0.95 -- the spectrum estimate not generalising off its own sample.

Is there a world that produces the other verdict? Yes, and E1.1 already
contains it: the bimodal stream has 105 free directions with a noise floor
4 orders of magnitude down. If the budget is workable anywhere, it is workable
there, and a stream-by-stream split is the expected outcome rather than a
uniform verdict.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.spectrum import (  # noqa: E402
    RtEstimator,
    interference_by_rank,
    leakage_by_rank,
    rank_for_energy,
    stream_bimodal,
    DomainMixture,
    stream_exponential,
    stream_power_law,
)

DIM = 128
N_OBS = 4000
N_CALIB = 500
N_TEST = 500
SEED = 20260806

RETENTIONS = (0.90, 0.95, 0.99, 0.999)
RANK_REQUEST = 8            # a rank-8 adapter, the same size E1.1 probed
INTERFERENCE_LIMIT = 0.05   # G1
LEAKAGE_BLOWUP = 2.0        # G2


def evaluate(name: str, feats, calib, test, rng) -> dict:
    est = RtEstimator(dim=DIM, lam=1.0)
    est.update(feats)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)

    curve = []
    for rho in RETENTIONS:
        r = rank_for_energy(est, calib, rho)          # chosen on calibration
        free = DIM - r
        leak_cal = leakage_by_rank(est, r, calib)     # circular by construction
        leak_test = leakage_by_rank(est, r, test)     # the honest number
        interf = interference_by_rank(
            est, r, test, readout, rank_request=RANK_REQUEST, rng=rng
        )
        curve.append(
            {
                "retention": rho,
                "committed_rank": int(r),
                "free_rank": int(free),
                "leak_calib": round(leak_cal, 4),
                "leak_test": round(leak_test, 4),
                "interference": round(interf, 4),
                "usable": bool(free >= RANK_REQUEST),
                "safe": bool(interf <= INTERFERENCE_LIMIT),
            }
        )

    workable = [c for c in curve if c["usable"] and c["safe"]]
    at95 = next(c for c in curve if c["retention"] == 0.95)
    blowup = at95["leak_test"] / max(at95["leak_calib"], 1e-9)

    g1 = len(workable) > 0
    g2 = blowup <= LEAKAGE_BLOWUP

    return {
        "stream": name,
        "curve": curve,
        "workable_retentions": [c["retention"] for c in workable],
        "leak_blowup_at_95": round(float(blowup), 2),
        "G1_budget_workable": bool(g1),
        "G2_estimate_generalises": bool(g2),
        "verdict": "PASS" if (g1 and g2) else "FAIL",
    }


def capacity_sweep(rng) -> list[dict]:
    """At what traffic-to-dimension ratio does the budget actually go empty?

    The domain-mixture result above is at DIM=128 with 12 domains of rank 10 --
    120 effective directions in 128, a 94% fill by construction. Reporting that
    as "the budget fails on realistic traffic" without this sweep would be
    reporting an artifact of the chosen dimension: at d=4096 the same 12 domains
    would leave enormous free rank.

    So the finding is not binary, it is a ratio. This locates the boundary.
    """
    out = []
    for dim in (128, 256, 512):
        for n_domains in (4, 8, 16, 32, 64):
            world = DomainMixture(dim, n_domains=n_domains, domain_rank=10,
                                  alpha=1.0, rng=rng)
            feats, _ = world.sample(4000, rng)
            calib, _ = world.sample(500, rng)
            test, _ = world.sample(500, rng)

            est = RtEstimator(dim=dim, lam=1.0)
            est.update(feats)
            readout = rng.normal(size=(dim, 16)) / np.sqrt(dim)

            best_free = 0
            for rho in RETENTIONS:
                r = rank_for_energy(est, calib, rho)
                free = dim - r
                interf = interference_by_rank(
                    est, r, test, readout, rank_request=RANK_REQUEST, rng=rng)
                if free >= RANK_REQUEST and interf <= INTERFERENCE_LIMIT:
                    best_free = max(best_free, free)

            out.append({
                "dim": dim,
                "n_domains": n_domains,
                "traffic_rank": n_domains * 10,
                "fill_ratio": round(n_domains * 10 / dim, 3),
                "best_safe_free_rank": int(best_free),
                "workable": bool(best_free >= RANK_REQUEST),
            })
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    specs = [
        ("bimodal (shape the design assumes)",
         lambda n: stream_bimodal(n, DIM, n_committed=24, floor=0.02, rng=rng)),
        ("exponential (soft knee)",
         lambda n: stream_exponential(n, DIM, rate=0.12, rng=rng)),
        ("power law alpha=1.0",
         lambda n: stream_power_law(n, DIM, alpha=1.0, rng=rng)),
        ("domain mixture (realistic WAM traffic)",
         (lambda w: (lambda n: w.sample(n, rng)[0]))(
             DomainMixture(DIM, n_domains=12, domain_rank=10, alpha=1.0, rng=rng))),
    ]

    rows = [evaluate(name, gen(N_OBS), gen(N_CALIB), gen(N_TEST), rng)
            for name, gen in specs]

    print(f"\nE1.1b  Subspace budget under the energy (GPM) criterion   (dim={DIM},"
          f" rank-{RANK_REQUEST} adapter)\n")
    for r in rows:
        print(f"  {r['stream']}")
        print(f"    {'retention':>10}{'committed':>11}{'free':>7}"
              f"{'leak cal':>10}{'leak test':>11}{'interf':>9}   usable  safe")
        for c in r["curve"]:
            print(
                f"    {c['retention']:>10.3f}{c['committed_rank']:>11}{c['free_rank']:>7}"
                f"{c['leak_calib']:>10.4f}{c['leak_test']:>11.4f}{c['interference']:>9.4f}"
                f"{'   yes' if c['usable'] else '    no':>9}"
                f"{'  yes' if c['safe'] else '   no':>6}"
            )
        print(f"    -> workable retentions: {r['workable_retentions'] or 'NONE'}"
              f"   | leak blowup at 0.95: {r['leak_blowup_at_95']}x"
              f"   | {r['verdict']}\n")

    print("  G1 some retention gives free rank >= "
          f"{RANK_REQUEST} at interference <= {INTERFERENCE_LIMIT}")
    print(f"  G2 test leakage <= {LEAKAGE_BLOWUP}x calibration leakage at rho=0.95\n")

    sweep = capacity_sweep(np.random.default_rng(SEED))
    print("  Capacity sweep - where does the budget actually go empty?")
    print("  (domain mixture, rank-10 domains; 'fill' = traffic rank / dim)\n")
    print(f"    {'dim':>6}{'domains':>9}{'fill':>8}{'safe free rank':>16}  workable")
    print("    " + "-" * 47)
    for s in sweep:
        print(f"    {s['dim']:>6}{s['n_domains']:>9}{s['fill_ratio']:>8.2f}"
              f"{s['best_safe_free_rank']:>16}"
              f"{'   yes' if s['workable'] else '    no'}")
    boundary = [s for s in sweep if not s["workable"]]
    if boundary:
        print(f"\n    budget empty from fill ratio ~{min(s['fill_ratio'] for s in boundary):.2f}"
              f" upward\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_1b_energy_criterion.json"
    out.write_text(json.dumps({"seed": SEED, "dim": DIM, "rank_request": RANK_REQUEST,
                               "rows": rows, "capacity_sweep": sweep}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
