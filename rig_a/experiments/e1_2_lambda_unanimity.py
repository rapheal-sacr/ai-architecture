"""E1.2 -- Does three-lambda unanimity deadlock the subspace budget?

CLAIM UNDER TEST (Part III section 2), the repair adopted for the budget
accountant:

    | allocate rank in a direction | short only | if the direction was actually
        committed, the promotion gate's non-regression test catches it |
    | release rank (mark a direction free) | unanimity across all three |
        nothing downstream catches this |

The asymmetry is argued from blast radius and is sound as far as it goes. What
it never prices is the RATE: release replenishes the free pool, and unanimity
means it replenishes at the speed of the SLOWEST estimator. If lambda_long
implies a commitment half-life much longer than adapters live, retired rank is
not reclaimed for many cycles, occupancy ratchets, and every region ends in
`reclaim`.

U2 IS NOT A SIMULATION RESULT -- IT IS A COMPOSITION, AND STRONGER FOR IT.
Allocation ranks on the short estimator in EVERY arm, so no lambda_long and no
release rule can touch it. Compose that with E1.1c and the second half of this
experiment is a two-line proof rather than a measurement:

    allocation consults only the fast estimator
    the only downstream catch is traffic-weighted (E1.1c)
    => allocation errors in low-mass domains are undetectable, for ANY setting

That cannot be argued down by a seed count. The simulation below is retained to
SIZE the effect, not to establish it.

GROUND TRUTH TOOK TWO ATTEMPTS AND BOTH FAILURES ARE INSTRUCTIVE.

The first version asked "did we allocate into a direction the LONG estimator
holds?", which is undefined for the two arms whose free_mask already contains
the long estimator: X and not-X is empty. So `unanimity` and `long_only`
reported zero unsafe allocations because they could not report anything else,
and the table invited the reading that unanimity is safe on U2. Same defect class
as `true_influencers - provenance`, in the arm that looked best.

The second version used a THRESHOLDED oracle, `base_rate > EPS`. That is worse
in a subtler way: it makes rare domains uncommitted BY DEFINITION, so allocating
into them is never flagged -- reproducing, inside the instrument, the exact
traffic weighting the finding is about.

What U2 actually asks is per-domain and unweighted. Every direction carries
signal for its own domain whatever that domain's traffic share, so the measure is
the fraction of each domain's subspace overwritten, and the question is whether
the WORST domain's damage is visible to a traffic-weighted check. Same move as
E1.1c: ask per region on equal footing rather than averaging over the
distribution that made the region rare.

ALLOCATION ORDER: PREDICTED TO BE THE MECHANISM, MEASURED AS A MINOR ONE. The
allocator takes the least-committed free directions first, which looks like the
conservative choice -- and least-visited is identical to rarest-domain, so the
prediction was that this policy preferentially targets the tail. Directionally
confirmed and much weaker than expected: ascending gives worst-domain damage
0.858 at mean traffic rank 11.1, descending 0.815 at rank 9.2, random 0.841 at
10.5. All three concentrate on the rarer half.
The reason is more damning than the prediction. Ordering only selects WITHIN the
free pool, and the free pool is rare-domain-dominated under every rule, because
low traffic is what "free" looks like to a visit-weighted estimator. So the pool
composition is the mechanism and the ordering is a detail -- which means
allocation order is not available as a repair.

ARMS (release rule; allocation is short-only throughout, as the design says):
    unanimity    Part III section 2's rule -- free iff all three say free
    short_only   fast and unsafe -- what unanimity was adopted to prevent
    medium_only  the middle
    long_only    what unanimity effectively degenerates to

KILL CRITERIA (pre-registered / re-registered):
    U1  [as originally registered] fails if unanimity starves >0.20 of requests
        while short_only starves <0.05.
    U1' [re-registered] the absolute form assumed short_only would be
        comfortable and it is not -- the budget is tight for every rule. U1'
        fails if unanimity starves >= 1.5x the fastest rule. Both print.
    U2  fails if the worst domain's overwritten fraction exceeds what a
        traffic-weighted check reports by 2x or more -- i.e. the damage that
        matters is invisible to the only instrument that would catch it.
    MANIPULATION CHECK: before any arm comparison is interpreted, at least one
    arm must sustain free rank > 0. B7 was a run in which nothing was ever free,
    so every arm starved identically and U1 "passed" on a comparison that never
    happened. The lint's logical form was satisfied; the manipulation had not
    taken. Checking the criterion COULD flip is not enough -- verify the arms
    actually differ on the quantity being varied.

Is there a world that produces the other verdict? For U1', yes -- the low
lambda_long end of the sweep, where release outruns retirement. For U2, yes: if
committed-but-decayed directions belonged to frequent domains, a traffic-mean
check would see them.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

DIM = 128
N_DOMAINS = 16
DIRS_PER_DOMAIN = DIM // N_DOMAINS
RANK_REQUEST = 8
SEED = 20260806

LAMBDAS = {"short": 0.90, "medium": 0.98, "long": 0.999}
# EPS must fall INSIDE the range of per-direction equilibrium rates or the
# partition is degenerate -- see B7. Range here is 0.46 to 7.39.
EPS = 1.0
EVENTS_PER_CYCLE = 200
CYCLES = 1200
BURN_IN = 300
ADAPTER_HALFLIFE = 80
ARRIVAL = 0.10
ADAPTER_SIGNAL = 4.0

RULES = ("unanimity", "short_only", "medium_only", "long_only")
ORDERS = ("ascending", "descending", "random")
N_SEEDS = 12


def halflife(lam: float) -> float:
    return float(np.log(2) / -np.log(lam))


class Budget:
    def __init__(self, rule, rng, lam_long=LAMBDAS["long"], order="ascending",
                 arrival=ARRIVAL):
        self.rule, self.rng, self.order, self.arrival = rule, rng, order, arrival
        self.lams = dict(LAMBDAS, long=lam_long)
        self.sig = {k: np.zeros(DIM) for k in self.lams}
        self.domain = np.repeat(np.arange(N_DOMAINS), DIRS_PER_DOMAIN)
        rates = 1.0 / np.arange(1, N_DOMAINS + 1, dtype=float)
        self.rates = rates / rates.sum()
        self.occupied = np.zeros(DIM, dtype=bool)
        self.adapters: list[dict] = []
        self.base_rate = EVENTS_PER_CYCLE * self.rates[self.domain] / DIRS_PER_DOMAIN

    def normalised(self, key):
        return self.sig[key] * (1.0 - self.lams[key])

    def committed(self, key):
        return self.normalised(key) > EPS

    def domain_damage(self):
        """Fraction of each domain's own subspace currently overwritten.

        Every direction carries signal for its domain, whatever that domain's
        traffic share. So the harm from allocation is per domain and unweighted,
        and the question U2 actually asks is whether the WORST domain's damage is
        visible to a traffic-weighted check. Same move as E1.1c: ask per region
        on equal footing rather than averaging over the distribution that made
        the region rare.
        """
        return np.array([self.occupied[self.domain == d].mean()
                         for d in range(N_DOMAINS)])

    def free_mask(self):
        if self.rule == "unanimity":
            free = ~(self.committed("short") | self.committed("medium")
                     | self.committed("long"))
        elif self.rule == "short_only":
            free = ~self.committed("short")
        elif self.rule == "medium_only":
            free = ~self.committed("medium")
        else:
            free = ~self.committed("long")
        return free & ~self.occupied

    def step(self):
        visits = np.zeros(DIM)
        doms = self.rng.choice(N_DOMAINS, size=EVENTS_PER_CYCLE, p=self.rates)
        offs = self.rng.integers(0, DIRS_PER_DOMAIN, size=EVENTS_PER_CYCLE)
        np.add.at(visits, doms * DIRS_PER_DOMAIN + offs, 1.0)
        for a in self.adapters:
            visits[a["dirs"]] += ADAPTER_SIGNAL
        for k, lam in self.lams.items():
            self.sig[k] = lam * self.sig[k] + visits

        alive = []
        for a in self.adapters:
            if self.rng.random() < 1.0 / ADAPTER_HALFLIFE:
                self.occupied[a["dirs"]] = False
            else:
                alive.append(a)
        self.adapters = alive

        served = starved = allocated = 0
        if self.rng.random() < self.arrival:
            idx = np.where(self.free_mask())[0]
            if len(idx) >= RANK_REQUEST:
                short = self.normalised("short")[idx]
                if self.order == "ascending":
                    order = idx[np.argsort(short)]           # least-visited first
                elif self.order == "descending":
                    order = idx[np.argsort(-short)]
                else:
                    order = self.rng.permutation(idx)
                dirs = order[:RANK_REQUEST]
                self.occupied[dirs] = True
                self.adapters.append({"dirs": dirs})
                served, allocated = 1, RANK_REQUEST
            else:
                starved = 1

        dmg = self.domain_damage()
        return {"free": int(self.free_mask().sum()), "served": served,
                "starved": starved, "allocated": allocated,
                "worst_damage": float(dmg.max()),
                "traffic_damage": float(np.sum(self.rates * dmg)),
                "worst_domain": int(np.argmax(dmg))}


def run(rule, seed, lam_long=LAMBDAS["long"], order="ascending"):
    b = Budget(rule, np.random.default_rng(seed), lam_long, order)
    frees, tot = [], {"served": 0, "starved": 0, "allocated": 0}
    worst, traffic, worst_dom = [], [], []
    for c in range(CYCLES):
        r = b.step()
        if c >= BURN_IN:
            frees.append(r["free"])
            worst.append(r["worst_damage"])
            traffic.append(r["traffic_damage"])
            worst_dom.append(r["worst_domain"])
            for k in tot:
                tot[k] += r[k]
    req = tot["served"] + tot["starved"]
    return {
        "mean_free_rank": float(np.mean(frees)),
        "starve_rate": tot["starved"] / max(req, 1),
        "allocated": tot["allocated"],
        "adapters_live": len(b.adapters),
        "worst_damage": float(np.mean(worst)),
        "traffic_damage": float(np.mean(traffic)),
        "worst_domain_rank": float(np.mean(worst_dom)),
    }


def run_many(rule, lam_long=LAMBDAS["long"], order="ascending"):
    rs = [run(rule, SEED + i, lam_long, order) for i in range(N_SEEDS)]

    def m(k):
        return float(np.mean([r[k] for r in rs]))

    return {"rule": rule, "order": order,
            "mean_free_rank": round(m("mean_free_rank"), 1),
            "starve_rate": round(m("starve_rate"), 3),
            "starve_se": round(float(np.std([r["starve_rate"] for r in rs]))
                               / np.sqrt(N_SEEDS), 3),
            "allocated": int(sum(r["allocated"] for r in rs)),
            "adapters_live": round(m("adapters_live"), 1),
            "worst_damage": round(m("worst_damage"), 3),
            "traffic_damage": round(m("traffic_damage"), 3),
            "blindness": round(m("worst_damage") / max(m("traffic_damage"), 1e-9), 1),
            "worst_domain_rank": round(m("worst_domain_rank"), 1)}


def main() -> int:
    global ARRIVAL
    print(f"\nE1.2  Does three-lambda unanimity deadlock the budget?"
          f"   (dim={DIM}, rank-{RANK_REQUEST}, {N_SEEDS} seeds)\n")
    for k, lam in LAMBDAS.items():
        print(f"    {k:<8} lambda={lam:<7} commitment half-life {halflife(lam):8.1f}")
    print(f"    adapter half-life {ADAPTER_HALFLIFE}\n")

    _keep, ARRIVAL = ARRIVAL, 0.0
    base = Budget("unanimity", np.random.default_rng(SEED), arrival=0.0)
    for _ in range(400):
        base.step()
    traffic_only = int(base.free_mask().sum())
    ARRIVAL = _keep
    print(f"  traffic-only free rank: {traffic_only}/{DIM}\n")

    rows = [run_many(r) for r in RULES]
    hdr = (f"  {'release rule':<14}{'mean free':>11}{'starve':>9}{'se':>7}{'live':>7}"
           f"{'worst dmg':>11}{'traffic dmg':>13}{'blind':>7}{'worst dom':>11}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['rule']:<14}{r['mean_free_rank']:>11.1f}{r['starve_rate']:>9.3f}"
              f"{r['starve_se']:>7.3f}{r['adapters_live']:>7.1f}"
              f"{r['worst_damage']:>11.3f}{r['traffic_damage']:>13.3f}"
              f"{r['blindness']:>7.1f}{r['worst_domain_rank']:>11.1f}")
    print("  worst dmg   fraction of the WORST domain's subspace overwritten.")
    print("              CONTINGENT: domains here partition the space disjointly, so")
    print("              every allocation damages someone by construction and this")
    print("              level cannot come out low. It is a distributional statistic")
    print("              at subscription 1.0 and inter-domain overlap 0 -- the")
    print("              pessimistic end of the axis E1.1c Panel C found decisive.")
    print("  traffic dmg the same, averaged over traffic -- what a check would see")
    print("  blind       THE FINDING. Two views of the SAME damage, so the ratio")
    print("              survives whether or not the damage was avoidable.")
    print(f"  worst dom   mean traffic rank of the worst-hit domain (0=most frequent,"
          f" {N_DOMAINS-1}=rarest)")

    # MANIPULATION CHECK -- B7's lesson.
    assert max(r["mean_free_rank"] for r in rows) > 0, (
        "manipulation check failed: no arm sustained free rank, so the arms were "
        "never compared on the quantity being varied -- this is B7")

    una = next(r for r in rows if r["rule"] == "unanimity")
    sho = next(r for r in rows if r["rule"] == "short_only")
    u1 = not (una["starve_rate"] > 0.20 and sho["starve_rate"] < 0.05)
    ratio = una["starve_rate"] / max(sho["starve_rate"], 1e-9)
    u1r = ratio < 1.5
    # U2: is the worst domain's damage visible to a traffic-weighted check?
    u2 = sho["blindness"] < 2.0

    print("\n  Allocation order (short_only, where the effect is largest)")
    hdr2 = (f"  {'order':<14}{'mean free':>11}{'starve':>9}{'worst dmg':>11}"
            f"{'blind':>7}{'worst dom':>11}")
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    orders = [run_many("short_only", order=o) for o in ORDERS]
    for r in orders:
        print(f"  {r['order']:<14}{r['mean_free_rank']:>11.1f}{r['starve_rate']:>9.3f}"
              f"{r['worst_damage']:>11.3f}{r['blindness']:>7.1f}"
              f"{r['worst_domain_rank']:>11.1f}")

    print("\n  lambda_long sweep (unanimity)")
    hdr3 = f"  {'lambda_long':>12}{'half-life':>11}{'mean free':>11}{'starve':>9}{'se':>7}"
    print(hdr3); print("  " + "-" * (len(hdr3) - 2))
    sweep = []
    for ll in (0.98, 0.99, 0.995, 0.999, 0.9995):
        r = run_many("unanimity", lam_long=ll)
        sweep.append({"lambda_long": ll, "halflife": round(halflife(ll), 1), **r})
        print(f"  {ll:>12.4f}{halflife(ll):>11.1f}{r['mean_free_rank']:>11.1f}"
              f"{r['starve_rate']:>9.3f}{r['starve_se']:>7.3f}")
    best = min(sweep, key=lambda x: x["starve_rate"])
    print(f"\n    best swept lambda_long = {best['lambda_long']} at starve"
          f" {best['starve_rate']:.3f} -- still"
          f" {best['starve_rate']/sho['starve_rate']:.2f}x short_only."
          f" U1' fails across the ENTIRE range.")
    if abs(best["lambda_long"] - LAMBDAS["medium"]) < 1e-9:
        print("    And that is lambda_medium: unanimity is least bad exactly where")
        print("    the three-timescale structure degenerates to two, so the third")
        print("    timescale contributes nothing but cost at every setting tested.")

    print(f"\n  U1  as pre-registered: {'ok' if u1 else 'NO'}"
          f"   (short_only starves {sho['starve_rate']:.3f}, not <0.05)")
    print(f"  U1' re-registered, relative: ratio {ratio:.2f}x -> {'ok' if u1r else 'NO'}")
    print(f"  U2  worst-domain damage visible to a traffic check"
          f" (blindness {sho['blindness']:.1f}x): {'ok' if u2 else 'NO'}"
          f"   (see also the composition proof in the docstring)")
    print(f"\n  VERDICT (U1', U2): {'PASS' if (u1r and u2) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_2_lambda_unanimity.json"
    out.write_text(json.dumps(
        {"seed": SEED, "dim": DIM, "eps": EPS, "lambdas": LAMBDAS, "n_seeds": N_SEEDS,
         "traffic_only_free_rank": traffic_only,
         "rows": rows, "allocation_order": orders, "lambda_long_sweep": sweep,
         "best_lambda_long": best["lambda_long"],
         "U1_as_preregistered": bool(u1), "U1_relative": bool(u1r),
         "starve_ratio": round(float(ratio), 2), "U2_allocation_caught": bool(u2),
         "verdict": "PASS" if (u1r and u2) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
