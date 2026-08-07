"""E3.1b -- Is E3.1's T4 inversion caused by spillover, or by ASYMMETRIC spillover?

E3.1's fourth regime is the one that decides the architecture's central
generalization claim. T1-T3 pass; T4 -- "patches with REAL spillover" -- inverts
the margin to -0.008, and PLAN.md concludes:

    "Part II section B is conditionally true, and the condition is unstated:
     narrow patches must have near-zero *real* off-target effect ...
     Measuring real patch spillover decides whether the architecture's central
     generalization claim holds in practice."

But look at where T4 applies the spillover:

    if kind == "patch":
        if self.spillover > 0:
            off = np.full(N_REGIONS, self.spillover)

Only patches. Skill adapters get none.

The justification offered for the regime is "a low-rank update fit to one region
is not region-confined." That argument does not distinguish patches from skills
-- a skill adapter is also a low-rank update, fit to a *broader* distribution,
so if anything it should spill more. T4 therefore models a specific and much
stronger condition than the one it is defended by: patches leak and skills do
not.

And the arithmetic matters. Net transfer is a SUM over off-target regions. Adding
the same constant to every candidate's off-target vector shifts every score by
the same amount and leaves the ranking untouched. So a symmetric spillover
should be nearly free, and only an asymmetric one can invert the margin. If that
is what is happening, T4 is not a test of "real adapters spill over" -- it is a
test of a differential that has no stated reason to exist.

KILL CRITERIA (pre-registered):
    S1 fails (T4 stands as a general finding) if SYMMETRIC spillover also
       inverts the margin. Then spillover per se defeats net-transfer ranking
       and PLAN.md's reading is right.
    S2 fails (T4 is an asymmetry artifact) if symmetric spillover PRESERVES the
       margin while patch-only inverts it at the same magnitude. Then the
       condition Part II section B needs is not "patches have near-zero
       spillover" but the far weaker "patches do not spill over MORE than
       skills", and the Rig B measurement changes from a level to a difference.
    S3 reports the differential threshold: how much excess patch spillover, over
       and above skill spillover, the ranking tolerates before inverting. That
       number is the actual condition the design owes a statement of.

Everything else is E3.1 as it ships -- same World, same arms, same seeds, same
trial count. Only the spillover application site moves.
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
    "e3_1_base", ROOT / "rig_a" / "experiments" / "e3_1_transfer_ranking.py")
E = importlib.util.module_from_spec(_spec)
sys.modules["e3_1_base"] = E
_spec.loader.exec_module(E)

MODES = ("patch_only", "symmetric")


class World(E.World):
    """E3.1's world with the spillover application site made a parameter."""

    def __init__(self, rng, noise, spurious, spill_patch=0.0, spill_skill=0.0):
        super().__init__(rng, noise, spurious, spill_patch)
        self.spill_patch = spill_patch
        self.spill_skill = spill_skill

    def measured_deltas(self, cand):
        d = self.true_deltas(cand).copy()
        kind, ident = cand
        amount = self.spill_patch if kind == "patch" else self.spill_skill
        if amount > 0:
            off = np.full(E.N_REGIONS, amount)
            if kind == "patch":
                off[ident] = 0.0
            else:
                t = self.true_deltas(cand)
                if t.max() > 0:
                    off[int(np.argmax(t))] = 0.0   # its own first-order region
            d = d + off
        if kind == "patch" and self.spurious > 0:
            s = self.rng.normal(0.0, self.spurious, size=E.N_REGIONS)
            s[ident] = 0.0
            d = d + s
        if self.noise > 0:
            d = d + self.rng.normal(0.0, self.noise, size=E.N_REGIONS)
        return d


def run_arm(arm, seed, noise, spurious, spill_patch, spill_skill):
    rng = np.random.default_rng(seed)
    w = World(rng, noise, spurious, spill_patch, spill_skill)
    for _ in range(E.N_PROMOTIONS):
        pool = []
        for _ in range(E.CANDIDATES_PER_ROUND):
            if rng.random() < 0.5:
                pool.append(("skill", int(rng.integers(0, E.N_SKILLS))))
            else:
                pool.append(("patch", int(rng.integers(0, E.N_REGIONS))))
        best, best_score = None, -np.inf
        for cand in pool:
            d = w.measured_deltas(cand)
            ti = int(np.argmax(d))
            score = {"target": d[ti],
                     "transfer": float(d.sum() - d[ti]),
                     "hybrid": d[ti] + float(d.sum() - d[ti])}[arm]
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            w.apply(best)
    return (w.compositional_score(), len(w.learned), w.comp_only_acquired())


def regime(name, noise, spurious, sp, ss):
    res = {}
    for arm in ("target", "transfer"):
        t = [run_arm(arm, E.SEED + i, noise, spurious, sp, ss) for i in range(E.N_TRIALS)]
        res[arm] = {
            "compositional": round(float(np.mean([x[0] for x in t])), 4),
            "skills": round(float(np.mean([x[1] for x in t])), 2),
            "comp_only": round(float(np.mean([x[2] for x in t])), 3),
        }
    return {"regime": name, "spill_patch": sp, "spill_skill": ss,
            "arms": res,
            "margin": round(res["transfer"]["compositional"]
                            - res["target"]["compositional"], 4)}


def main() -> int:
    print("\nE3.1b  Does symmetric spillover invert the margin, or only asymmetric?\n")
    rows = [
        regime("clean", 0.0, 0.0, 0.0, 0.0),
        regime("noisy + spurious", 0.02, 0.03, 0.0, 0.0),
        regime("spill 0.02  patch only  (E3.1 T4)", 0.02, 0.03, 0.02, 0.0),
        regime("spill 0.02  symmetric", 0.02, 0.03, 0.02, 0.02),
        regime("spill 0.05  patch only", 0.02, 0.03, 0.05, 0.0),
        regime("spill 0.05  symmetric", 0.02, 0.03, 0.05, 0.05),
    ]
    h = (f"{'regime':<36}{'target':>9}{'transfer':>10}{'margin':>9}"
         f"{'sk(t)':>7}{'sk(x)':>7}{'comp_only(x)':>14}")
    print(h); print("-" * len(h))
    for r in rows:
        a, b = r["arms"]["target"], r["arms"]["transfer"]
        print(f"{r['regime']:<36}{a['compositional']:>9.3f}{b['compositional']:>10.3f}"
              f"{r['margin']:>+9.3f}{a['skills']:>7.1f}{b['skills']:>7.1f}"
              f"{b['comp_only']:>14.3f}")

    # S3: how much EXCESS patch spillover does the ranking tolerate?
    print("\nS3  differential sweep -- skill spillover fixed at 0.02")
    h2 = f"{'patch spill':>12}{'excess':>9}{'target':>9}{'transfer':>10}{'margin':>9}"
    print(h2); print("-" * len(h2))
    sweep = []
    for sp in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        r = regime(f"sp={sp}", 0.02, 0.03, sp, 0.02)
        sweep.append({"spill_patch": sp, "excess": round(sp - 0.02, 3), **r})
        print(f"{sp:>12.3f}{sp-0.02:>9.3f}"
              f"{r['arms']['target']['compositional']:>9.3f}"
              f"{r['arms']['transfer']['compositional']:>10.3f}{r['margin']:>+9.3f}")

    po = next(r for r in rows if "patch only  (E3.1 T4)" in r["regime"])
    sy = next(r for r in rows if r["regime"] == "spill 0.02  symmetric")
    s1_fails = sy["margin"] <= 0
    s2_fails = (po["margin"] <= 0) and (sy["margin"] > 0)
    first_inv = next((s["excess"] for s in sweep if s["margin"] <= 0), None)

    print(f"\nS1 (symmetric also inverts): {'FAIL' if s1_fails else 'ok -- it does not'}")
    print(f"S2 (T4 is an asymmetry artifact): "
          f"{'CONFIRMED' if s2_fails else 'not confirmed'}")
    print(f"S3 excess patch spillover tolerated before inversion: "
          f"{'none of those tested' if first_inv is None else f'{first_inv:.3f}'}")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "e3_1b_spillover_symmetry.json").write_text(json.dumps(
        {"seed": E.SEED, "trials": E.N_TRIALS, "regimes": rows,
         "differential_sweep": sweep,
         "S1_symmetric_also_inverts": bool(s1_fails),
         "S2_asymmetry_artifact": bool(s2_fails),
         "S3_excess_tolerated": first_inv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
