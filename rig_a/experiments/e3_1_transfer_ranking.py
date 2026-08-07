"""E3.1 -- Does ranking by net transfer accumulate abstractions, or just patches?

CLAIM UNDER TEST (Part II section B) -- the central generalization claim of the
whole architecture:

    "among candidates that pass, rank by net transfer, not by target gain. A
     candidate gaining 8 points on target and +1 across five neighbours beats
     one gaining 12 on target and 0 elsewhere. That single reordering is the
     difference between accumulating patches and accumulating abstractions."

    "Run that gate for a year and you get a hundred adapters, each correct, none
     transferable, subspace exhausted -- a system that learns fast and
     generalizes badly."

This is the first experiment in the programme whose failure would have no local
repair. Everything tested so far has been a local mechanism question -- is this
formula right, is this list complete -- and local errors are locally fixable by
construction. If net-transfer ranking does not in fact accumulate abstractions,
Part II section B is false and there is no rewording that saves it: the
architecture becomes a very well-audited patch accumulator.

WHY THE OBVIOUS VERSION WOULD BE A TAUTOLOGY. Build a world where general skills
have positive off-target delta and patches have zero, and net-transfer ranking
prefers general skills by construction. That tests arithmetic, not the design.

The non-trivial question is whether the rule survives measurement conditions
that are actually present:

    NOISE          off-target deltas are estimated from finite probes.
    SPURIOUS       a patch can show incidental positive off-target delta.
    LATENCY        and this is the sharp one -- a genuinely general abstraction
                   may show NEAR-ZERO immediate off-target delta, because its
                   value only appears in COMPOSITION with skills not yet
                   acquired. Net transfer is a first-order measurement. If the
                   abstractions that matter most are second-order, the rule
                   systematically misses exactly what it was built to find.

ARMS:
    target      rank by target gain          (the gate Part II section B rejects)
    transfer    rank by net transfer         (the gate it proposes)
    hybrid      rank by target + net         (a middle the design does not use)

MEASURE: held-out COMPOSITIONAL tasks, which require several skills at once and
which no patch can help with. That is the operational meaning of "accumulating
abstractions" and it is not what either gate optimises.

KILL CRITERIA (pre-registered):
    T1 fails if `transfer` does not beat `target` on compositional held-out
       performance in the clean regime. That is Part II section B's claim in its
       most favourable setting; failing here would falsify it outright.
    T2 fails if `transfer`'s advantage does not survive realistic measurement --
       specifically if it drops below half its clean-regime margin once probe
       noise and spurious off-target correlation are present.
    T3 fails if `transfer` does not beat `target` at acquiring
       COMPOSITIONAL-ONLY skills, the ones whose payoff is invisible to a
       first-order transfer measurement.

Is there a world that produces the other verdict? For T1, yes: if patches
carried real off-target benefit, the two rankings would agree and neither would
dominate. For T3, no world in this family makes compositional-only skills
visible to a first-order measure -- which is the point, and why T3 is expected
to fail while T1 passes. A split verdict is the informative outcome.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_SKILLS = 14
N_COMPOSITIONAL_ONLY = 5      # skills whose payoff is invisible to first-order delta
N_REGIONS = 10
N_PROMOTIONS = 24
CANDIDATES_PER_ROUND = 8
N_COMPOSITIONAL_TASKS = 200
SEED = 20260806
N_TRIALS = 24

GAIN_SKILL = 0.06             # a general skill helps each region that needs it
GAIN_PATCH = 0.14             # a patch helps its one region MORE -- why the naive gate likes it
ARMS = ("target", "transfer", "hybrid")


class World:
    def __init__(self, rng, noise: float, spurious: float, spillover: float = 0.0):
        self.rng = rng
        self.noise = noise
        self.spurious = spurious
        self.spillover = spillover
        # Which skills each region needs.
        self.region_skills = [
            set(rng.choice(N_SKILLS, size=rng.integers(2, 5), replace=False).tolist())
            for _ in range(N_REGIONS)
        ]
        # Compositional-only skills pay nothing measurable on their own.
        self.comp_only = set(range(N_SKILLS - N_COMPOSITIONAL_ONLY, N_SKILLS))
        # Held-out tasks, each needing several skills at once. No patch helps.
        self.tasks = [
            set(rng.choice(N_SKILLS, size=3, replace=False).tolist())
            for _ in range(N_COMPOSITIONAL_TASKS)
        ]
        self.learned: set[int] = set()
        self.patched: set[int] = set()

    # -- what a candidate would actually do ---------------------------------

    def true_deltas(self, cand) -> np.ndarray:
        d = np.zeros(N_REGIONS)
        kind, ident = cand
        if kind == "skill":
            if ident in self.learned:
                return d
            for r in range(N_REGIONS):
                if ident in self.region_skills[r]:
                    # A compositional-only skill produces no first-order gain.
                    d[r] = 0.0 if ident in self.comp_only else GAIN_SKILL
        else:
            if ident not in self.patched:
                d[ident] = GAIN_PATCH
        return d

    def measured_deltas(self, cand) -> np.ndarray:
        """What the gate sees: finite probes, plus incidental off-target movement."""
        d = self.true_deltas(cand).copy()
        kind, ident = cand
        if kind == "patch":
            if self.spillover > 0:
                # Systematically POSITIVE off-target benefit, not zero-mean.
                # This is the world built to defeat the rule: net transfer now
                # rewards patches directly, so the statistic no longer separates
                # narrow from general.
                off = np.full(N_REGIONS, self.spillover)
                off[ident] = 0.0
                d = d + off
            if self.spurious > 0:
                off = self.rng.normal(0.0, self.spurious, size=N_REGIONS)
                off[ident] = 0.0
                d = d + off
        if self.noise > 0:
            d = d + self.rng.normal(0.0, self.noise, size=N_REGIONS)
        return d

    def apply(self, cand) -> None:
        kind, ident = cand
        if kind == "skill":
            self.learned.add(ident)
        else:
            self.patched.add(ident)

    def compositional_score(self) -> float:
        """Fraction of held-out compositional tasks whose every skill is learned."""
        return float(np.mean([t <= self.learned for t in self.tasks]))

    def comp_only_acquired(self) -> float:
        return len(self.learned & self.comp_only) / max(len(self.comp_only), 1)


def run_arm(arm: str, seed: int, noise: float, spurious: float,
            spillover: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    w = World(rng, noise, spurious, spillover)

    for _ in range(N_PROMOTIONS):
        pool = []
        for _ in range(CANDIDATES_PER_ROUND):
            if rng.random() < 0.5:
                pool.append(("skill", int(rng.integers(0, N_SKILLS))))
            else:
                pool.append(("patch", int(rng.integers(0, N_REGIONS))))

        best, best_score = None, -np.inf
        for cand in pool:
            d = w.measured_deltas(cand)
            target_idx = int(np.argmax(d))
            target_gain = d[target_idx]
            net = float(d.sum() - d[target_idx])
            score = {"target": target_gain,
                     "transfer": net,
                     "hybrid": target_gain + net}[arm]
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            w.apply(best)

    return {
        "arm": arm,
        "compositional_score": w.compositional_score(),
        "skills_learned": len(w.learned),
        "comp_only_acquired": w.comp_only_acquired(),
        "patches": len(w.patched),
    }


def regime(name: str, noise: float, spurious: float, spillover: float = 0.0) -> dict:
    out = {"regime": name, "noise": noise, "spurious": spurious,
           "spillover": spillover, "arms": {}}
    for arm in ARMS:
        trials = [run_arm(arm, SEED + t, noise, spurious, spillover)
                  for t in range(N_TRIALS)]
        out["arms"][arm] = {
            "compositional": round(float(np.mean([t["compositional_score"] for t in trials])), 4),
            "compositional_sd": round(float(np.std([t["compositional_score"] for t in trials])), 4),
            "skills": round(float(np.mean([t["skills_learned"] for t in trials])), 2),
            "comp_only": round(float(np.mean([t["comp_only_acquired"] for t in trials])), 3),
            "patches": round(float(np.mean([t["patches"] for t in trials])), 2),
        }
    return out


def main() -> int:
    regimes = [
        regime("clean", 0.0, 0.0),
        regime("noisy probes", 0.03, 0.0),
        regime("noisy + spurious", 0.03, 0.03),
        # Adversarial: patches carry REAL positive spillover, so net transfer
        # rewards them directly. Built to defeat the rule, not to confirm it.
        regime("patches with real spillover", 0.03, 0.03, spillover=0.02),
    ]

    print(f"\nE3.1  Does net-transfer ranking accumulate abstractions?"
          f"   ({N_TRIALS} trials, {N_PROMOTIONS} promotions)\n")
    for rg in regimes:
        print(f"  {rg['regime']}  (noise {rg['noise']}, spurious {rg['spurious']},"
              f" spillover {rg.get('spillover', 0.0)})")
        print(f"    {'arm':<12}{'compositional':>15}{'sd':>8}{'skills':>9}"
              f"{'comp-only':>12}{'patches':>10}")
        for arm in ARMS:
            a = rg["arms"][arm]
            print(f"    {arm:<12}{a['compositional']:>15.3f}{a['compositional_sd']:>8.3f}"
                  f"{a['skills']:>9.2f}{a['comp_only']:>12.3f}{a['patches']:>10.2f}")
        print()

    clean = regimes[0]["arms"]
    hard = regimes[2]["arms"]
    adv = regimes[3]["arms"]
    margin_clean = clean["transfer"]["compositional"] - clean["target"]["compositional"]
    margin_hard = hard["transfer"]["compositional"] - hard["target"]["compositional"]

    t1 = margin_clean > 0
    t2 = margin_hard >= 0.5 * margin_clean if margin_clean > 0 else False
    t3 = hard["transfer"]["comp_only"] > hard["target"]["comp_only"]

    print(f"  clean margin (transfer - target): {margin_clean:+.3f}")
    print(f"  hard  margin:                     {margin_hard:+.3f}"
          f"   ({margin_hard / margin_clean:.0%} of clean)" if margin_clean > 0 else "")
    adv_margin = adv["transfer"]["compositional"] - adv["target"]["compositional"]
    t4 = adv_margin > 0
    print(f"  adversarial margin (real patch spillover): {adv_margin:+.3f}")
    print(f"\n  T1 transfer beats target, clean regime:        {'ok' if t1 else 'NO'}")
    print(f"  T2 advantage survives noise + spurious:        {'ok' if t2 else 'NO'}")
    print(f"  T3 transfer acquires compositional-only skills:{'ok' if t3 else 'NO'}")
    print(f"  T4 survives patches with real spillover:       {'ok' if t4 else 'NO'}")
    print(f"\n  VERDICT: {'PASS' if (t1 and t2 and t3 and t4) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e3_1_transfer_ranking.json"
    out.write_text(json.dumps(
        {"seed": SEED, "regimes": regimes,
         "margin_clean": round(margin_clean, 4), "margin_hard": round(margin_hard, 4),
         "T1_beats_target_clean": bool(t1), "T2_survives_measurement": bool(t2),
         "T3_acquires_compositional_only": bool(t3),
         "T4_survives_real_spillover": bool(t4),
         "margin_adversarial": round(adv_margin, 4),
         "verdict": "PASS" if (t1 and t2 and t3 and t4) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
