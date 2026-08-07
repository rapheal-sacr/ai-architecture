"""E1.1d -- Tail-safe free rank as a function of subscription ratio.

THE MISSING CURVE, AND IT BLOCKS JOINT FEASIBILITY.

Every free-rank number in this programme is measured at one point. E1.1c's
CONFIGS are (8,8), (16,8), (32,4) -- 64, 128 and 128 directions of traffic
against dim 128, so one config at 0.5x subscription and two at exactly 1.0x.
E1.2's world is 16 domains x 8 directions = 128 = dim, also exactly 1.0x. So
"17-18 free at tail-safe" and "6-15 free under the release rules" are both
readings at Sum(r_d)/d = 1, and neither says what happens on either side of it.

Joint feasibility takes free rank as its primary axis. Running it against a
single subscription point would produce a feasible region at one slice of the
parameter that most needs varying, which is the same error as reporting a level
where only a ratio is robust.

Two axes, because E1.1c Panel C already showed the second one is decisive:

    subscription   Sum over domains of domain_rank, divided by dim
    overlap        shared-subspace fraction between domains; Panel C found
                   overlap 0.73 relaxed tail-safe retention from 0.999 to 0.995
                   and tripled free rank

E1.2 sits at overlap 0, the pessimistic end. Real domains share structure, so
where the boundary actually falls is a Rig B measurement -- this locates the
boundary as a function of both, so that measurement has something to index into.

KILL CRITERIA (pre-registered):
    S1 the deliverable, not a pass/fail: the subscription ratio at which
       tail-safe free rank falls below RANK_REQUEST, per overlap level. That is
       the feasibility boundary joint feasibility needs.
    S2 fails if the boundary is insensitive to overlap -- if so, E1.1c Panel C's
       rescue does not generalise past its two configs and the Rig B overlap
       measurement is not worth taking.
    S3 fails if tail-safe free rank is below RANK_REQUEST even at 0.25x
       subscription with high overlap. That would mean no realistic operating
       point exists at all, rather than a boundary to stay under.

Is there a world that produces the other verdict? For S3, yes and it is the
expected one: at 0.25x subscription the traffic spans 32 of 128 directions, so
generous free rank is the default expectation and finding otherwise would be the
surprise. For S2, yes: if the tail-safe retention is set by the rarest domain's
isolation rather than by aggregate crowding, overlap would not move it.
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
    rank_for_energy,
)

DIM = 128
DOMAIN_RANK = 8               # held fixed so subscription isolates domain COUNT
N_OBS = 4000
N_CALIB = 500
N_PROBE = 300
RANK_REQUEST = 8
INTERFERENCE_LIMIT = 0.05
SEED = 20260806
RETENTIONS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9995)
N_DOMAINS_GRID = (4, 8, 16, 24, 32, 48)     # -> subscription 0.25 .. 3.0
OVERLAPS = (0.0, 0.4, 0.7)


def build(n_domains: int, share: float, rng):
    """Domains on partially shared subspaces, as E1.1c Panel C constructs them."""
    common = np.linalg.qr(rng.normal(size=(DIM, DOMAIN_RANK)))[0]
    bases = []
    for _ in range(n_domains):
        own = np.linalg.qr(rng.normal(size=(DIM, DOMAIN_RANK)))[0]
        bases.append(np.linalg.qr(share * common + (1 - share) * own)[0])
    within = np.arange(1, DOMAIN_RANK + 1, dtype=float) ** -0.5
    rates = 1.0 / np.arange(1, n_domains + 1, dtype=float)
    rates /= rates.sum()

    def draw(n):
        lab = rng.choice(n_domains, size=n, p=rates)
        f = np.empty((n, DIM))
        for i, d in enumerate(lab):
            f[i] = bases[d] @ (rng.normal(size=DOMAIN_RANK) * within)
        return f

    probes = {d: (rng.normal(size=(N_PROBE, DOMAIN_RANK)) * within) @ bases[d].T
              for d in range(n_domains)}
    measured_overlap = float(np.mean([
        np.linalg.norm(bases[i].T @ bases[j]) ** 2 / DOMAIN_RANK
        for i in range(n_domains) for j in range(i + 1, n_domains)
    ])) if n_domains > 1 else 0.0
    return draw, probes, measured_overlap


def tail_safe(n_domains: int, share: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draw, probes, overlap = build(n_domains, share, rng)
    est = RtEstimator(dim=DIM, lam=1.0)
    est.update(draw(N_OBS))
    calib = draw(N_CALIB)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)

    for rho in RETENTIONS:
        r = rank_for_energy(est, calib, rho)
        worst = max(
            interference_by_rank(est, r, probes[d], readout, RANK_REQUEST, rng)
            for d in range(n_domains)
        )
        if worst <= INTERFERENCE_LIMIT:
            return {"n_domains": n_domains, "share": share,
                    "subscription": round(n_domains * DOMAIN_RANK / DIM, 2),
                    "measured_overlap": round(overlap, 3),
                    "tail_safe_retention": rho, "committed_rank": int(r),
                    "free_rank": int(DIM - r),
                    "usable": bool(DIM - r >= RANK_REQUEST)}
    return {"n_domains": n_domains, "share": share,
            "subscription": round(n_domains * DOMAIN_RANK / DIM, 2),
            "measured_overlap": round(overlap, 3),
            "tail_safe_retention": None, "committed_rank": None,
            "free_rank": 0, "usable": False}


def main() -> int:
    print(f"\nE1.1d  Tail-safe free rank vs subscription ratio"
          f"   (dim={DIM}, domain rank {DOMAIN_RANK}, rank-{RANK_REQUEST} request)\n")

    rows = []
    for share in OVERLAPS:
        print(f"  overlap share {share:.1f}")
        hdr = (f"    {'subscr':>8}{'domains':>9}{'measured ov':>13}"
               f"{'tail-safe rho':>15}{'committed':>11}{'free':>7}{'usable':>8}")
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        for nd in N_DOMAINS_GRID:
            r = tail_safe(nd, share, SEED)
            rows.append(r)
            rs = f"{r['tail_safe_retention']}" if r["tail_safe_retention"] else ">0.9995"
            cr = str(r["committed_rank"]) if r["committed_rank"] is not None else "-"
            print(f"    {r['subscription']:>8.2f}{r['n_domains']:>9}"
                  f"{r['measured_overlap']:>13.3f}{rs:>15}{cr:>11}"
                  f"{r['free_rank']:>7}{('yes' if r['usable'] else 'NO'):>8}")
        print()

    # S1 -- the boundary, per overlap level
    print("  S1  feasibility boundary: highest subscription still usable")
    print(f"      {'overlap':>9}{'last usable subscr':>21}{'free there':>13}"
          f"{'first unusable':>16}")
    boundaries = []
    for share in OVERLAPS:
        sub = [r for r in rows if r["share"] == share]
        ok = [r for r in sub if r["usable"]]
        bad = [r for r in sub if not r["usable"]]
        last = max((r["subscription"] for r in ok), default=None)
        first_bad = min((r["subscription"] for r in bad), default=None)
        freeat = next((r["free_rank"] for r in ok if r["subscription"] == last), 0)
        boundaries.append({"share": share, "last_usable_subscription": last,
                           "free_rank_there": freeat,
                           "first_unusable_subscription": first_bad})
        print(f"      {share:>9.1f}{str(last):>21}{freeat:>13}{str(first_bad):>16}")

    lasts = [b["last_usable_subscription"] for b in boundaries]
    s2 = len({x for x in lasts if x is not None}) > 1
    low_high = next((r for r in rows
                     if r["share"] == max(OVERLAPS) and r["subscription"] <= 0.25), None)
    s3 = bool(low_high and low_high["usable"])

    print(f"\n  S2 boundary moves with overlap:              {'ok' if s2 else 'NO'}")
    print(f"  S3 a usable point exists at low subscription: {'ok' if s3 else 'NO'}")
    print("\n  S1 is the deliverable, not a verdict: joint feasibility must take")
    print("  subscription AND region count as variables, not constants. Note region")
    print("  count also sets per-region probe cost under the weighting rule, so it")
    print("  enters the feasible region twice -- once as crowding, once as budget.\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_1d_subscription_curve.json"
    out.write_text(json.dumps(
        {"seed": SEED, "dim": DIM, "domain_rank": DOMAIN_RANK,
         "rank_request": RANK_REQUEST, "interference_limit": INTERFERENCE_LIMIT,
         "rows": rows, "boundaries": boundaries,
         "S2_boundary_moves_with_overlap": bool(s2),
         "S3_usable_at_low_subscription": bool(s3)}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
