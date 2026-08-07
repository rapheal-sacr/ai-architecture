"""E3.1d -- Is E3.1's compositional-only acquisition figure a tiebreak artifact?

E3.1 reported, and this plan published in PLAN.md, README.md and claims.yaml:

    "transfer acquires compositional-only skills at 55% while learning 72% of
     skills overall, so it still under-acquires exactly the skills whose value
     is invisible to a first-order measure. It merely stops patches crowding
     them out."

The 55% was offered as evidence for the MECHANISM claim -- that net transfer
removes the reward for narrowness rather than detecting generality. This
experiment checks whether that number measures anything.

THE STRUCTURAL REASON TO DOUBT IT. A compositional-only skill has `true_deltas`
identically zero in every region, by construction. So under net transfer it
scores exactly 0.000 -- and so does a patch (its only gain is at the target,
which net transfer excludes), and so does an already-learned skill. Direct check
in the clean regime:

    regular skill (>=2 regions)   target= 0.060   transfer= 0.060
    comp-only skill               target= 0.000   transfer= 0.000
    patch                         target= 0.140   transfer= 0.000
    already-learned skill         target= 0.000   transfer= 0.000

Three candidate classes tie at zero, and E3.1's selection loop uses
`if score > best_score` -- strict -- so the tie is broken by whichever candidate
the pool happened to emit first. No first-order statistic over per-region deltas
can distinguish a candidate whose per-region deltas are all zero from one that
has already been applied. Any acquisition of compositional-only skills is luck.

ARMS -- the same transfer ranking, four tiebreak rules:
    first         current behaviour: earliest in pool emission order wins
    random        uniform among the tied set
    prefer_patch  ties resolved toward patches
    prefer_skill  ties resolved toward skills

KILL CRITERION (pre-registered):
    N4 fails -- i.e. the number is an artifact -- if `comp_only` spread across
    tiebreak rules exceeds 0.10. A quantity that moves more than ten points
    under a rule that should be irrelevant is not measuring the ranking.

Is there a world that produces the other verdict? Yes: if net transfer genuinely
discriminated compositional-only skills, all four rules would agree, because the
tie would not exist. That is exactly what a real detection mechanism would look
like, and it is what the published claim implied.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "e3_1", ROOT / "rig_a" / "experiments" / "e3_1_transfer_ranking.py")
e31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e31)

SEED = 20260806
N_TRIALS = 40
TIEBREAKS = ("first", "random", "prefer_patch", "prefer_skill")
SPREAD_LIMIT = 0.10


def run(tiebreak: str, seed: int, noise: float, spurious: float) -> dict:
    rng = np.random.default_rng(seed)
    w = e31.World(rng, noise, spurious)

    for _ in range(e31.N_PROMOTIONS):
        pool = []
        for _ in range(e31.CANDIDATES_PER_ROUND):
            if rng.random() < 0.5:
                pool.append(("skill", int(rng.integers(0, e31.N_SKILLS))))
            else:
                pool.append(("patch", int(rng.integers(0, e31.N_REGIONS))))

        scored = []
        for cand in pool:
            d = w.measured_deltas(cand)
            ti = int(np.argmax(d))
            scored.append((float(d.sum() - d[ti]), cand))

        best = max(s for s, _ in scored)
        tied = [c for s, c in scored if s >= best - 1e-12]

        if tiebreak == "first" or len(tied) == 1:
            pick = tied[0]
        elif tiebreak == "random":
            pick = tied[int(rng.integers(0, len(tied)))]
        elif tiebreak == "prefer_patch":
            patches = [c for c in tied if c[0] == "patch"]
            pick = patches[0] if patches else tied[0]
        else:
            skills = [c for c in tied if c[0] == "skill"]
            pick = skills[0] if skills else tied[0]
        w.apply(pick)

    return {
        "compositional": w.compositional_score(),
        "comp_only": w.comp_only_acquired(),
        "skills": len(w.learned),
    }


def panel(noise: float, spurious: float, label: str) -> dict:
    out = {"regime": label, "noise": noise, "spurious": spurious, "rules": {}}
    for tb in TIEBREAKS:
        trials = [run(tb, SEED + t, noise, spurious) for t in range(N_TRIALS)]
        out["rules"][tb] = {
            "comp_only": round(float(np.mean([t["comp_only"] for t in trials])), 3),
            "compositional": round(float(np.mean([t["compositional"] for t in trials])), 3),
            "skills": round(float(np.mean([t["skills"] for t in trials])), 2),
        }
    vals = [out["rules"][tb]["comp_only"] for tb in TIEBREAKS]
    out["comp_only_spread"] = round(max(vals) - min(vals), 3)
    return out


def main() -> int:
    panels = [
        panel(0.0, 0.0, "clean"),
        panel(0.03, 0.03, "noisy + spurious"),
    ]

    print(f"\nE3.1b  Is the compositional-only figure a tiebreak artifact?"
          f"   ({N_TRIALS} trials)\n")
    for pn in panels:
        print(f"  {pn['regime']}")
        print(f"    {'tiebreak':<16}{'comp_only':>12}{'compositional':>16}{'skills':>9}")
        for tb in TIEBREAKS:
            r = pn["rules"][tb]
            print(f"    {tb:<16}{r['comp_only']:>12.3f}"
                  f"{r['compositional']:>16.3f}{r['skills']:>9.2f}")
        print(f"    spread in comp_only: {pn['comp_only_spread']:.3f}\n")

    worst = max(pn["comp_only_spread"] for pn in panels)
    artifact = worst > SPREAD_LIMIT

    print(f"  worst spread across tiebreak rules: {worst:.3f}"
          f"   (N4 threshold {SPREAD_LIMIT})")
    print(f"  N4 comp_only is a tiebreak artifact: {'YES' if artifact else 'no'}")
    if artifact:
        print("\n  => The 55% / 72% pair is withdrawn. It measures pool emission")
        print("     order, not the ranking rule. The analytic claim survives on")
        print("     stronger grounds: a first-order statistic over per-region")
        print("     deltas CANNOT see a candidate whose per-region deltas are all")
        print("     zero. That needs no number.\n")

    out = ROOT / "results" / "e3_1d_tiebreak_audit.json"
    out.write_text(json.dumps(
        {"seed": SEED, "panels": panels, "worst_spread": worst,
         "N4_tiebreak_artifact": bool(artifact)}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
