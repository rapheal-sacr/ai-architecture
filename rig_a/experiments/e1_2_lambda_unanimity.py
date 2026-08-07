"""E1.2 -- Does three-lambda unanimity deadlock the subspace budget?

CLAIM UNDER TEST (Part III section 2), the repair adopted for the budget
accountant:

    | allocate rank in a direction | short only | if the direction was actually
        committed, the promotion gate's non-regression test catches it |
    | release rank (mark a direction free) | unanimity across all three |
        nothing downstream catches this. You overwrite committed competence and
        the only record that it was committed has already decayed |

The asymmetry is argued from blast radius and it is sound as far as it goes.
What it does not price is the RATE. Release is what replenishes the free pool,
and requiring unanimity means the pool replenishes at the speed of the SLOWEST
estimator. Part III section 2's own dashboard item warns about lambda_long
implying a commitment half-life shorter than the measured adapter half-life. The
opposite gap is the one that bites here: if lambda_long implies a commitment
half-life much LONGER than adapters live, then retired adapters' rank is not
reclaimed for many cycles after they die, occupancy ratchets, and every region
ends in `reclaim`.

WHY THIS IS NOW WORTH RUNNING AT PRIORITY. E1.2's predicted deadlock originally
compounded with a number that was retracted (B4's "6 free directions"). It now
compounds with a live one: E1.1c measured tail-safe free rank at 17-18 of 128
against a RANK_REQUEST of 8. Room for roughly two adapters. A release rule that
replenishes slowly against a pool that small is the strongest remaining
candidate for a genuine architectural failure in this programme.

NORMALISATION, because it decides whether the arms are comparable at all. A
direction visited at rate v with decay lambda equilibrates at sigma* = v/(1-lam),
so the three estimators sit on wildly different scales and a single eps would
mean three different things. Each sigma is therefore normalised by (1-lambda),
so all three estimate the same underlying visit rate and one eps is meaningful
across them. Decay after visits stop is lambda^t regardless, giving commitment
half-lives of 6.6, 34.3 and 693 cycles.

ARMS (release rule; allocation is short-only throughout, as the design says):
    unanimity    Part III section 2's rule -- free iff all three say free
    short_only   fast and unsafe -- the thing unanimity was adopted to prevent
    medium_only  the middle
    long_only    what unanimity effectively degenerates to

KILL CRITERIA (pre-registered):
    U1 fails if unanimity spends more than 20% of cycles unable to serve an
       allocation request while short_only spends less than 5%. That is the
       safety fix creating the failure it was adopted to prevent.
    U2 fails if allocations made on the short estimator alone land in directions
       the long estimator still considers committed AND the resulting
       interference is invisible to a traffic-weighted non-regression check.
       That is the stated justification for short-only allocation being false.

Is there a world that produces the other verdict? For U1, yes: if lambda_long's
implied commitment half-life is shorter than the adapter half-life, release
outruns retirement and no ratchet forms -- that is the low-lambda_long end of the
sweep. For U2, yes: if committed-but-decayed directions belong to frequent
domains, a traffic-mean check sees them and the justification holds.
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
# EPS must fall INSIDE the range of per-direction equilibrium rates, or the
# partition is degenerate. With Zipfian domains the range here is 0.46 (rarest)
# to 7.39 (most frequent); EPS = 1.0 commits the top seven domains and leaves 72
# of 128 directions free at steady state with no adapters. The first run of this
# experiment used 0.25, below every direction's rate, so nothing was ever free,
# every arm starved 100% of the time, and U1 "passed" because both arms failed
# identically. Rig bug B7.
EPS = 1.0                     # on the normalised visit-rate scale
TRAFFIC = 200.0               # visit mass per cycle, split Zipfian over domains
EVENTS_PER_CYCLE = 200        # sparse arrivals, so rare domains genuinely go quiet
CYCLES = 1200
BURN_IN = 300
ADAPTER_HALFLIFE = 80         # cycles; the design's own "adapter half-life"
ARRIVAL = 0.10                # adapter compile requests per cycle
ADAPTER_SIGNAL = 4.0          # clears EPS, so a live adapter commits its own dirs

RULES = ("unanimity", "short_only", "medium_only", "long_only")
N_SEEDS = 12          # single runs are far too noisy at this arrival rate


def halflife(lam: float) -> float:
    return float(np.log(2) / -np.log(lam))


class Budget:
    def __init__(self, rule: str, rng, lam_long: float = LAMBDAS["long"]):
        self.rule = rule
        self.rng = rng
        self.lams = dict(LAMBDAS, long=lam_long)
        # raw second-moment accumulators, one per estimator
        self.sig = {k: np.zeros(DIM) for k in self.lams}
        self.domain = np.repeat(np.arange(N_DOMAINS), DIRS_PER_DOMAIN)
        rates = 1.0 / np.arange(1, N_DOMAINS + 1, dtype=float)
        self.rates = rates / rates.sum()
        self.occupied = np.zeros(DIM, dtype=bool)
        self.adapters: list[dict] = []
        self.next_id = 0

    def normalised(self, key: str) -> np.ndarray:
        """sigma * (1 - lambda): all three then estimate the same visit rate."""
        return self.sig[key] * (1.0 - self.lams[key])

    def committed(self, key: str) -> np.ndarray:
        return self.normalised(key) > EPS

    def free_mask(self) -> np.ndarray:
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

    def step(self) -> dict:
        # Sparse stochastic traffic: draw events, each picking a domain Zipfian
        # then a direction within it. Same expectation as spreading the mass
        # evenly, but rare domains now go quiet for stretches -- which is what
        # makes the three decay rates behave differently at all.
        visits = np.zeros(DIM)
        doms = self.rng.choice(N_DOMAINS, size=EVENTS_PER_CYCLE, p=self.rates)
        offs = self.rng.integers(0, DIRS_PER_DOMAIN, size=EVENTS_PER_CYCLE)
        np.add.at(visits, doms * DIRS_PER_DOMAIN + offs, 1.0)
        # live adapters keep their own directions carrying signal
        for a in self.adapters:
            visits[a["dirs"]] += ADAPTER_SIGNAL

        for k, lam in self.lams.items():
            self.sig[k] = lam * self.sig[k] + visits

        # retire
        alive = []
        for a in self.adapters:
            a["age"] += 1
            if self.rng.random() < 1.0 / ADAPTER_HALFLIFE:
                self.occupied[a["dirs"]] = False
            else:
                alive.append(a)
        self.adapters = alive

        # allocation demand -- always on the SHORT estimator, per the design
        served, starved, unsafe_alloc, unsafe_rare = 0, 0, 0, 0
        if self.rng.random() < ARRIVAL:
            free = self.free_mask()
            idx = np.where(free)[0]
            if len(idx) >= RANK_REQUEST:
                # allocation ranks by the short estimator only
                order = idx[np.argsort(self.normalised("short")[idx])]
                dirs = order[:RANK_REQUEST]
                self.occupied[dirs] = True
                self.adapters.append({"id": self.next_id, "dirs": dirs, "age": 0})
                self.next_id += 1
                served = 1
                # U2: did we write into something the LONG estimator still holds?
                still = self.committed("long")[dirs]
                unsafe_alloc = int(still.sum())
                if unsafe_alloc:
                    # would a traffic-weighted non-regression check see it? Only
                    # if those directions belong to domains with real traffic mass.
                    mass = self.rates[self.domain[dirs[still]]].sum()
                    if mass < 0.05:
                        unsafe_rare = int(still.sum())
            else:
                starved = 1

        return {"free": int(self.free_mask().sum()), "served": served,
                "starved": starved, "adapters": len(self.adapters),
                "unsafe_alloc": unsafe_alloc, "unsafe_rare": unsafe_rare}


def run_many(rule: str, lam_long: float = LAMBDAS["long"]) -> dict:
    """Average over seeds. A single run of this simulation is dominated by the
    arrival draw -- the first version reported a lambda_long sweep that was not
    monotone (0.461, 0.589, 0.779, 0.657, 0.461), which was variance, not shape."""
    rs = [run(rule, SEED + i, lam_long) for i in range(N_SEEDS)]
    return {
        "rule": rule,
        "mean_free_rank": round(float(np.mean([r["mean_free_rank"] for r in rs])), 1),
        "starve_rate": round(float(np.mean([r["starve_rate"] for r in rs])), 3),
        "starve_sd": round(float(np.std([r["starve_rate"] for r in rs])), 3),
        "adapters_live": round(float(np.mean([r["adapters_live"] for r in rs])), 1),
        "unsafe_alloc_dirs": int(np.sum([r["unsafe_alloc_dirs"] for r in rs])),
        "unsafe_rare_dirs": int(np.sum([r["unsafe_rare_dirs"] for r in rs])),
    }


def run(rule: str, seed: int, lam_long: float = LAMBDAS["long"]) -> dict:
    b = Budget(rule, np.random.default_rng(seed), lam_long)
    frees, served, starved, unsafe, unsafe_rare = [], 0, 0, 0, 0
    for c in range(CYCLES):
        r = b.step()
        if c >= BURN_IN:
            frees.append(r["free"])
            served += r["served"]
            starved += r["starved"]
            unsafe += r["unsafe_alloc"]
            unsafe_rare += r["unsafe_rare"]
    req = served + starved
    return {
        "rule": rule,
        "mean_free_rank": round(float(np.mean(frees)), 1),
        "final_free_rank": int(frees[-1]),
        "starve_rate": round(starved / max(req, 1), 3),
        "adapters_live": len(b.adapters),
        "unsafe_alloc_dirs": unsafe,
        "unsafe_rare_dirs": unsafe_rare,
    }


def main() -> int:
    print(f"\nE1.2  Does three-lambda unanimity deadlock the budget?"
          f"   (dim={DIM}, rank-{RANK_REQUEST} requests)\n")
    print("  commitment half-lives implied by each lambda:")
    for k, lam in LAMBDAS.items():
        print(f"    {k:<8} lambda={lam:<7} half-life {halflife(lam):8.1f} cycles")
    print(f"    adapter half-life {ADAPTER_HALFLIFE} cycles"
          f"  <- release must outrun this or occupancy ratchets\n")

    # Genuinely adapter-free: the allocation loop must be off, or this reports
    # the occupied state rather than the traffic-only baseline.
    global ARRIVAL
    _keep, ARRIVAL = ARRIVAL, 0.0
    base = Budget("unanimity", np.random.default_rng(SEED))
    for _ in range(400):
        base.step()
    traffic_only = int(base.free_mask().sum())
    ARRIVAL = _keep
    print(f"  traffic-only free rank, no adapters: {traffic_only}/{DIM}"
          f"  (EPS={EPS}) -- the budget every rule starts from\n")

    rows = [run_many(r) for r in RULES]
    hdr = (f"  {'release rule':<14}{'mean free':>11}{'starve rate':>13}{'sd':>7}"
           f"{'live':>7}{'unsafe dirs':>13}{'of which rare':>15}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['rule']:<14}{r['mean_free_rank']:>11.1f}{r['starve_rate']:>13.3f}"
              f"{r['starve_sd']:>7.3f}{r['adapters_live']:>7.1f}"
              f"{r['unsafe_alloc_dirs']:>13}{r['unsafe_rare_dirs']:>15}")
    print(f"  (mean of {N_SEEDS} seeds)")

    una = next(r for r in rows if r["rule"] == "unanimity")
    sho = next(r for r in rows if r["rule"] == "short_only")
    # U1 as pre-registered. The absolute thresholds assumed short_only would be
    # comfortable; it is not -- the budget is tight for every rule -- so the
    # criterion does not fire even where the effect is large. Reported as
    # registered, with the relative number beside it and a re-registration below.
    u1 = not (una["starve_rate"] > 0.20 and sho["starve_rate"] < 0.05)
    ratio = una["starve_rate"] / max(sho["starve_rate"], 1e-9)
    u1r = ratio < 1.5
    u2 = not (una["unsafe_rare_dirs"] > 0 or sho["unsafe_rare_dirs"] > 0)

    print(f"\n  Sweep: lambda_long against a {ADAPTER_HALFLIFE}-cycle adapter half-life")
    hdr2 = (f"  {'lambda_long':>12}{'half-life':>11}{'mean free':>11}"
            f"{'starve rate':>13}")
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    sweep = []
    for ll in (0.98, 0.99, 0.995, 0.999, 0.9995):
        r = run_many("unanimity", lam_long=ll)
        sweep.append({"lambda_long": ll, "halflife": round(halflife(ll), 1), **r})
        print(f"  {ll:>12.4f}{halflife(ll):>11.1f}{r['mean_free_rank']:>11.1f}"
              f"{r['starve_rate']:>13.3f}")

    print(f"\n  U1  as pre-registered (unanimity >0.20 AND short_only <0.05): "
          f"{'ok' if u1 else 'NO'}")
    print(f"      RE-REGISTERED as relative -- the absolute form assumed short_only")
    print(f"      would be comfortable, and it starves at {sho['starve_rate']:.3f}."
          f" Corrected criterion:")
    print(f"      U1' fails if unanimity starves >=1.5x the fastest rule."
          f"  ratio {ratio:.2f}x -> {'ok' if u1r else 'NO'}")
    print(f"  U2 short-only allocation is caught downstream as claimed:      "
          f"{'ok' if u2 else 'NO'}")
    print(f"\n  VERDICT (on U1' and U2): {'PASS' if (u1r and u2) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_2_lambda_unanimity.json"
    out.write_text(json.dumps(
        {"seed": SEED, "dim": DIM, "eps": EPS, "lambdas": LAMBDAS,
         "traffic_only_free_rank": traffic_only,
         "adapter_halflife": ADAPTER_HALFLIFE, "rows": rows, "lambda_long_sweep": sweep,
         "U1_as_preregistered": bool(u1), "U1_relative_reregistered": bool(u1r),
         "unanimity_vs_short_starve_ratio": round(float(ratio), 2),
         "U2_allocation_caught": bool(u2), "n_seeds": N_SEEDS,
         "verdict": "PASS" if (u1r and u2) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
