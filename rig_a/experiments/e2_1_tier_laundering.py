"""E2.1 -- Does probe harvesting widen the verifiable surface, or only deepen it?

CLAIM UNDER TEST (Part III section 5):

    "Every real interaction that resolves with a verified outcome is a candidate
     probe, and I1 guarantees it has a readable preimage. Sample them at
     resolution, seal them, hide them. That is a T1 artifact promoted to T2 by
     the acts of freezing and concealment -- no authoring, no annotation."

Part III section 0 argues that verification coverage is the binding constraint
on the whole architecture, and section 5 is the mechanism offered against it. So
this is the cheapest measurement with the largest consequence in the plan.

Two problems, of different kinds.

TIER LAUNDERING. "Resolves with a verified outcome" does not name a tier. An
outcome adjudicated by a T3 learned judge, frozen and hidden, becomes a T2
probe by the act of concealment -- and T3 is the tier the design explicitly
forbids from promoting anything. The suite then carries judge opinion with the
authority of a sealed probe. Whether this happens at all is a definitional
question; how much of the suite it contaminates is empirical, and depends on
the traffic mix.

COVERAGE BIAS, AND WHY THE OBVIOUS VERSION IS A TAUTOLOGY. The obvious follow-up
is "filter to T0/T1 only, then check whether any surviving probe covers a
T3-only domain." That question answers itself: a T0/T1 probe cannot come from a
domain with no T0/T1 verifier, by definition. It is not a measurement.

The real question is about PARTIAL verifiability. Domains are not uniformly one
tier. A domain typically has some slice that is cheaply checkable -- short,
closed-form, deterministic -- and a remainder that is not, and the two are not
a random split. If checkability correlates with a task being easy, then a
harvested suite is a biased sample of each domain's easy corner, reported as
coverage of the domain.

That correlation strength is the parameter everything turns on, and it is NOT
measured here -- it is swept. What this experiment establishes is the shape of
the damage as a function of it, and which regime you would have to be in for
harvesting to work. Locating where real traffic sits on that curve is Rig B
work.

KILL CRITERIA (pre-registered):
    H1 fails if an unfiltered harvest puts any T3-adjudicated outcome into the
       sealed suite. Absolute -- one laundered probe is a tier violation.
    H2 fails if a strict T0/T1 filter yields less than 20% of the GAP SET --
       not of traffic. Traffic-weighted yield is the flattering axis and cannot
       really fail: it is dominated by high-volume domains, which are exactly
       the ones already checkable. What a sealed suite has to cover is what the
       system is about to practise, and gap-set membership is anti-correlated
       with checkability by construction -- gaps are where the system fails,
       failures are disproportionately hard, hard is disproportionately
       un-checkable. An earlier version of this experiment reported 94-97%
       "coverage" measured as traffic volume, which is compatible with covering
       almost none of the gap set, and made H2 and E2.5 the same question
       answered twice with the metric that could not fail.
    H3 fails if the harvested slice's mean difficulty differs from its domain's
       true mean by more than 0.10 at correlation >= 0.5 -- the suite testing a
       corner while reporting the domain.

Is there a world that produces the other verdict? For H1, yes: a traffic mix
with no T3-only resolutions launders nothing. For H3, yes: at correlation 0 the
checkable slice is a random sample and the bias vanishes. Both endpoints are in
the sweep below, which is what makes the middle informative.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_INTERACTIONS = 40000
N_DOMAINS = 12
SEED = 20260806

YIELD_MIN = 0.20            # H2
BIAS_LIMIT = 0.10           # H3
CORRELATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Per-domain probability that an interaction is checkable by a sound verifier
# (T0/T1) at all. Code-like domains are mostly checkable; open-ended ones
# barely. This profile is an assumption about the world, stated here rather
# than buried, and it is the main thing a Rig B replication would replace.
DOMAIN_CHECKABILITY = np.array(
    [0.95, 0.90, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.12, 0.06, 0.03, 0.01]
)


def simulate(correlation: float, rng: np.random.Generator) -> dict:
    """One traffic sample. `correlation` couples checkability to easiness."""
    # Zipfian traffic across domains.
    rates = 1.0 / np.arange(1, N_DOMAINS + 1, dtype=float)
    rates /= rates.sum()
    domain = rng.choice(N_DOMAINS, size=N_INTERACTIONS, p=rates)

    difficulty = rng.uniform(0.0, 1.0, size=N_INTERACTIONS)

    # Gap-set weight. A gap is a region the system fails in, so weight rises
    # with difficulty. This is the distribution L8 will actually draw practice
    # from, and it is the distribution a sealed suite has to cover.
    gap_weight = difficulty**2

    # An interaction is checkable with probability set by its domain, tilted by
    # difficulty in proportion to `correlation`. At correlation 0 the tilt is
    # absent and checkability is a coin flip independent of difficulty.
    base = DOMAIN_CHECKABILITY[domain]
    tilt = 1.0 - correlation * difficulty
    p_checkable = np.clip(base * tilt, 0.0, 1.0)
    checkable = rng.random(N_INTERACTIONS) < p_checkable

    # Everything else that resolves at all resolves under a learned judge.
    resolved = checkable | (rng.random(N_INTERACTIONS) < 0.85)
    tier = np.where(checkable, 1, 3)      # 1 = T0/T1 sound, 3 = learned judge

    # -- harvest policy A: Part III section 5 as written, no tier filter -----
    harvested_all = resolved
    n_all = int(harvested_all.sum())
    laundered = int((harvested_all & (tier == 3)).sum())

    # -- harvest policy B: strict T0/T1 only --------------------------------
    harvested_strict = resolved & checkable
    n_strict = int(harvested_strict.sum())
    yield_strict = n_strict / N_INTERACTIONS
    # The number that actually decides section 5: what fraction of the gap set,
    # not of the traffic, survives a sound-tier filter.
    yield_gap = float(gap_weight[harvested_strict].sum() / gap_weight.sum())

    # -- H3: is the strict suite representative of the domains it claims? ----
    biases, covered_mass, blind_mass = [], 0.0, 0.0
    for d in range(N_DOMAINS):
        in_d = domain == d
        in_d_h = in_d & harvested_strict
        mass = float(in_d.mean())
        if in_d_h.sum() >= 30:
            bias = float(difficulty[in_d_h].mean() - difficulty[in_d].mean())
            biases.append(abs(bias))
            covered_mass += mass
        else:
            blind_mass += mass       # too few probes to test this domain at all

    mean_bias = float(np.mean(biases)) if biases else 0.0
    worst_bias = float(np.max(biases)) if biases else 0.0

    return {
        "correlation": correlation,
        "unfiltered_suite_size": n_all,
        "laundered_probes": laundered,
        "laundering_rate": round(laundered / max(n_all, 1), 4),
        "strict_yield_traffic": round(yield_strict, 4),
        "strict_yield_gapset": round(yield_gap, 4),
        "mean_difficulty_bias": round(mean_bias, 4),
        "worst_difficulty_bias": round(worst_bias, 4),
        "traffic_mass_covered": round(covered_mass, 4),
        "traffic_mass_blind": round(blind_mass, 4),
    }


def main() -> int:
    rows = [simulate(c, np.random.default_rng(SEED + i))
            for i, c in enumerate(CORRELATIONS)]

    h1 = all(r["laundered_probes"] == 0 for r in rows)
    h2 = all(r["strict_yield_gapset"] >= YIELD_MIN for r in rows)
    h2_traffic = all(r["strict_yield_traffic"] >= YIELD_MIN for r in rows)
    h3 = all(r["mean_difficulty_bias"] <= BIAS_LIMIT
             for r in rows if r["correlation"] >= 0.5)

    hdr = (f"{'corr':>6}{'suite':>8}{'laundered':>11}{'rate':>8}"
           f"{'yield:traffic':>15}{'yield:gapset':>14}{'mean bias':>11}"
           f"{'worst':>8}{'blind':>8}")
    print(f"\nE2.1  Does probe harvesting widen the verifiable surface?"
          f"   ({N_INTERACTIONS} interactions, {N_DOMAINS} domains)\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['correlation']:>6.2f}{r['unfiltered_suite_size']:>8}"
            f"{r['laundered_probes']:>11}{r['laundering_rate']:>8.3f}"
            f"{r['strict_yield_traffic']:>15.3f}{r['strict_yield_gapset']:>14.3f}"
            f"{r['mean_difficulty_bias']:>11.4f}"
            f"{r['worst_difficulty_bias']:>8.4f}{r['traffic_mass_blind']:>8.3f}"
        )
    print(
        f"\n  corr          how strongly checkability tracks easiness (swept, not measured)"
        f"\n  laundered     T3-adjudicated outcomes sitting in an unfiltered 'T2' suite"
        f"\n  yield:traffic fraction of TRAFFIC harvestable under a T0/T1-only filter"
        f"\n  yield:gapset  fraction of the GAP SET so harvestable -- the number that"
        f" decides section 5"
        f"\n  mean/worst    difficulty gap between a domain's harvested slice and the domain"
        f"\n  covered/blind traffic mass in domains with enough probes to test / with too few"
    )
    print(
        f"\n  H1 no laundering under the policy as written:      {'ok' if h1 else 'NO'}"
        f"\n  H2 gap-set yield >= {YIELD_MIN} at every correlation:     {'ok' if h2 else 'NO'}"
        f"\n     (traffic-weighted, the old flattering axis:          {'ok' if h2_traffic else 'NO'})"
        f"\n  H3 mean bias <= {BIAS_LIMIT} where corr >= 0.5:           {'ok' if h3 else 'NO'}"
        f"\n\n  VERDICT: {'PASS' if (h1 and h2 and h3) else 'FAIL'}\n"
    )

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e2_1_tier_laundering.json"
    out.write_text(json.dumps(
        {"seed": SEED, "domain_checkability": DOMAIN_CHECKABILITY.tolist(),
         "rows": rows, "H1_no_laundering": h1, "H2_yield_sufficient": h2,
         "H3_representative": h3, "H2_traffic_weighted": h2_traffic,
         "verdict": "PASS" if (h1 and h2 and h3) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
