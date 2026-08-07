"""E3.1c -- Does the transfer MATRIX earn anything over a plain breadth penalty?

E3.1's conclusion is that net transfer does not detect generality, it removes the
reward for narrowness. If that is the mechanism, then the comparison that decides
whether Root 3 is justified is not target-vs-transfer. It is net transfer against
the cheapest thing that also removes the reward for narrowness:

    arm 4    score = target_gain + beta * breadth(measured_deltas)

where `breadth` is scale-free -- a COUNT of regions moved above threshold, or the
entropy of the delta profile. Critically it aggregates no per-region magnitudes,
so it needs no per-cell history, no shrinkage, no signature partition, and no
ontology version to stamp artifacts against.

WHAT IS AND IS NOT AT STAKE. Arm 4 still requires measuring per-region deltas on
probe regions -- the same measurement tau rows are built from. So a match does not
remove the evaluation cost, which is Root 2 and binding. What it removes is the
MATRIX: E3.2's censored-tau estimation, E3.3's partition objective, E3.4's
observation-grain storage, and the L3 compiled signature-ontology view Part III
section 3 introduced to make re-partition safe. It also leaves T's three other
uses untested -- L8 curriculum prior, L7 merge prior, and diagonal-T as a
memorisation diagnostic -- which would then have to justify the matrix on their
own, having never been the stated reason for it.

PANEL B -- the unswept number. spillover = 0.02 appears once in E3.1, chosen. This
sweeps it and reports eps*, the value at which the patch-only margin crosses zero,
so the Rig B question becomes "does a rank-r adapter fit to one region produce
systematic off-target deltas above eps*" rather than "is the condition satisfied".

PANEL C -- is the comp-only number tie-breaking? `true_deltas` returns exactly
zero in every region for a comp_only skill, and already-learned skills also return
all-zeros. So late in a run most candidates score 0 and `if score > best_score`
awards the tie to whichever the pool happened to emit first. If comp-only
acquisition is tie-breaking rather than signal, randomising the tie-break should
move it and preferring patches on ties should collapse it.

KILL CRITERIA (pre-registered):
    N1 fails (the matrix is not earning its keep) if the best breadth arm comes
       within 0.02 compositional score of net transfer in the clean regime AND in
       both noisy regimes. 0.02 is ~1/4 of E3.1's clean margin.
    N2 fails (breadth is strictly better) if any breadth arm BEATS net transfer
       by more than 0.02 in any regime -- which would say the matrix is not
       merely redundant but worse than the one-line version.
    N3 the eps* sweep must locate a crossing inside [0, 0.10]; if the margin never
       crosses, E3.1's T4 was outside the informative range entirely.
    N4 comp-only acquisition must be insensitive to tie-break order. If it is
       not, the 55% is a property of pool emission order and cannot be reported
       as under-acquisition.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location(
    "e3_1_base", ROOT / "rig_a" / "experiments" / "e3_1_transfer_ranking.py")
E = importlib.util.module_from_spec(_s)
sys.modules["e3_1_base"] = E
_s.loader.exec_module(E)

NR = E.N_REGIONS
THETA = 0.02          # "moved" threshold for the scope count
BETAS = (0.02, 0.05, 0.10, 0.20)


def breadth_scope(d: np.ndarray, theta: float = THETA) -> float:
    """How many regions moved. A count -- no magnitudes aggregated."""
    return float(np.sum(d > theta))


def breadth_entropy(d: np.ndarray) -> float:
    """Entropy of the positive-delta profile. Scale-free by normalisation."""
    p = np.clip(d, 0.0, None)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def score_of(arm: str, d: np.ndarray, beta: float) -> float:
    ti = int(np.argmax(d))
    tgt = float(d[ti])
    net = float(d.sum() - d[ti])
    if arm == "target":
        return tgt
    if arm == "transfer":
        return net
    if arm == "hybrid":
        return tgt + net
    if arm == "scope":
        return tgt + beta * breadth_scope(d)
    if arm == "entropy":
        return tgt + beta * breadth_entropy(d)
    raise ValueError(arm)


def run(arm, seed, noise, spurious, spillover, beta=0.0, tie="first"):
    rng = np.random.default_rng(seed)
    w = E.World(rng, noise, spurious, spillover)
    for _ in range(E.N_PROMOTIONS):
        pool = []
        for _ in range(E.CANDIDATES_PER_ROUND):
            if rng.random() < 0.5:
                pool.append(("skill", int(rng.integers(0, E.N_SKILLS))))
            else:
                pool.append(("patch", int(rng.integers(0, E.N_REGIONS))))
        scored = [(score_of(arm, w.measured_deltas(c), beta), c) for c in pool]
        best = max(s for s, _ in scored)
        ties = [c for s, c in scored if s >= best - 1e-12]
        if tie == "first":
            pick = ties[0]
        elif tie == "random":
            pick = ties[int(rng.integers(0, len(ties)))]
        elif tie == "prefer_patch":
            p = [c for c in ties if c[0] == "patch"]
            pick = p[0] if p else ties[0]
        else:
            raise ValueError(tie)
        w.apply(pick)
    return w.compositional_score(), len(w.learned), w.comp_only_acquired()


def agg(arm, noise, spurious, spillover, beta=0.0, tie="first"):
    t = [run(arm, E.SEED + i, noise, spurious, spillover, beta, tie)
         for i in range(E.N_TRIALS)]
    return {"compositional": round(float(np.mean([x[0] for x in t])), 4),
            "sd": round(float(np.std([x[0] for x in t])), 4),
            "skills": round(float(np.mean([x[1] for x in t])), 2),
            "comp_only": round(float(np.mean([x[2] for x in t])), 3)}


REGIMES = (("clean", 0.0, 0.0, 0.0),
           ("noisy probes", 0.02, 0.0, 0.0),
           ("noisy + spurious", 0.02, 0.03, 0.0))


def main() -> int:
    print("\nE3.1c  Does the transfer matrix beat a one-line breadth penalty?\n")

    # -- Panel A: arm 4 against net transfer ------------------------------
    results = {}
    for name, nz, sp, sv in REGIMES:
        row = {"transfer": agg("transfer", nz, sp, sv),
               "target": agg("target", nz, sp, sv)}
        for arm in ("scope", "entropy"):
            best, bb = None, None
            for b in BETAS:
                r = agg(arm, nz, sp, sv, beta=b)
                if best is None or r["compositional"] > best["compositional"]:
                    best, bb = r, b
            row[arm] = {**best, "beta": bb}
        results[name] = row

    h = (f"{'regime':<20}{'target':>9}{'transfer':>10}{'scope':>9}{'beta':>6}"
         f"{'entropy':>9}{'beta':>6}{'gap: best-tr':>14}")
    print(h); print("-" * len(h))
    n1_gaps = []
    for name, _, _, _ in REGIMES:
        r = results[name]
        bestb = max(r["scope"]["compositional"], r["entropy"]["compositional"])
        gap = bestb - r["transfer"]["compositional"]
        n1_gaps.append(gap)
        print(f"{name:<20}{r['target']['compositional']:>9.3f}"
              f"{r['transfer']['compositional']:>10.3f}"
              f"{r['scope']['compositional']:>9.3f}{r['scope']['beta']:>6.2f}"
              f"{r['entropy']['compositional']:>9.3f}{r['entropy']['beta']:>6.2f}"
              f"{gap:>+14.3f}")
    n1_fail = all(g >= -0.02 for g in n1_gaps)
    n2_fail = any(g > 0.02 for g in n1_gaps)
    print(f"\nN1 (breadth within 0.02 of transfer in ALL regimes): "
          f"{'FAIL -- matrix not earning its keep' if n1_fail else 'ok -- transfer wins somewhere'}")
    print(f"N2 (breadth BEATS transfer by >0.02 anywhere): "
          f"{'FAIL -- breadth strictly better' if n2_fail else 'no'}")

    # -- Panel B: the unswept spillover, absolute --------------------------
    print("\nPanel B  patch-only spillover sweep (eps*)")
    hb = f"{'spillover':>11}{'target':>9}{'transfer':>10}{'margin':>9}{'sk(x)':>7}"
    print(hb); print("-" * len(hb))
    sweep, eps_star = [], None
    for sv in (0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100):
        a = agg("target", 0.02, 0.03, sv)
        b = agg("transfer", 0.02, 0.03, sv)
        m = round(b["compositional"] - a["compositional"], 4)
        sweep.append({"spillover": sv, "margin": m,
                      "target": a["compositional"], "transfer": b["compositional"],
                      "skills_transfer": b["skills"]})
        if eps_star is None and m <= 0:
            eps_star = sv
        print(f"{sv:>11.3f}{a['compositional']:>9.3f}{b['compositional']:>10.3f}"
              f"{m:>+9.3f}{b['skills']:>7.1f}")
    print(f"\nN3 eps* (margin crosses zero at patch-only spillover): "
          f"{'not in [0,0.10]' if eps_star is None else f'{eps_star:.3f}'}")

    # -- Panel C: is comp-only tie-breaking? -------------------------------
    print("\nPanel C  comp-only acquisition vs tie-break rule (transfer arm)")
    hc = f"{'tie-break':<16}{'clean':>9}{'comp_only':>11}{'skills':>8}"
    print(hc); print("-" * len(hc))
    ties = {}
    for tie in ("first", "random", "prefer_patch"):
        r = agg("transfer", 0.0, 0.0, 0.0, tie=tie)
        ties[tie] = r
        print(f"{tie:<16}{r['compositional']:>9.3f}{r['comp_only']:>11.3f}"
              f"{r['skills']:>8.1f}")
    spread = max(v["comp_only"] for v in ties.values()) - \
        min(v["comp_only"] for v in ties.values())
    print(f"\nN4 comp-only spread across tie-break rules: {spread:.3f}  "
          f"{'FAIL -- it is pool emission order' if spread > 0.10 else 'ok -- insensitive'}")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "e3_1c_narrowness_baseline.json").write_text(json.dumps(
        {"seed": E.SEED, "trials": E.N_TRIALS, "theta": THETA, "betas": list(BETAS),
         "panel_a": results, "panel_b_sweep": sweep, "eps_star": eps_star,
         "panel_c_ties": ties, "comp_only_spread": round(spread, 4),
         "N1_matrix_redundant": bool(n1_fail), "N2_breadth_better": bool(n2_fail),
         "N4_tiebreak_artifact": bool(spread > 0.10)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
