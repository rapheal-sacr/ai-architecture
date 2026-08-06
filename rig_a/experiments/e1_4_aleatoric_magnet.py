"""E1.4 -- Can the gap set tell "hard" from "random"?

CLAIM UNDER TEST (Part I section 3, Part II section A reading 2):

    "high posterior variance turns | L4 | the RLS error covariance P is
     already an epistemic gap detector"

    "Integrate posterior variance per domain signature over turns. High
     integrated variance in a region *is* the gap set. No new estimator,
     which is the claim Part I made and this is why it holds."

The word doing the work is *epistemic*. Posterior variance is not epistemic
variance; it is epistemic + aleatoric. The design never separates them, and
the two readings the architecture needs cannot both come from one quantity:

    the GATE (L4) must use PREDICTIVE variance. It answers "how wrong am I
        likely to be on this query", and a region where outcomes are genuinely
        random is a region where the weights should not be trusted.

    the GAP SET (L8) must use EPISTEMIC variance. It answers "would practice
        help here", and a region where outcomes are genuinely random is a
        region where practice is worthless.

Those are different numbers in exactly the places that matter. Worse, the
frontier-shaping reward actively selects for the confusion: 4*p(1-p) is
maximised at p = 0.5, and a coin flip sits at p = 0.5 permanently. A pure
noise region earns the highest possible proposer reward forever.

ARMS:
    predictive  the reading the gate needs, used as the gap set
    epistemic   the reading the gap set needs, used as the gate
    reducible   candidate fix: gap = variance *decline* under practice, so a
                region that does not improve when practised stops asking

KILL CRITERIA (pre-registered):
    A1 fails if noise regions capture > 3x their fair share of practice budget.
    A2 fails if gate confidence is uncorrelated with realized error
       (Spearman rho < 0.5 between predicted variance and true error rate).
    The fix passes only if it clears A1 *and* A2 at once, and does not cost
    more than 2 competence points on the learnable regions.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.world import Challenger, make_world  # noqa: E402

N_LEARNABLE = 11
N_NOISE = 1
CYCLES = 400
BUDGET_PER_CYCLE = 200
SEED = 20260806

MAGNET_LIMIT = 3.0      # A1: multiples of fair share
CALIBRATION_MIN = 0.5   # A2: Spearman rho


def run_arm(reading: str, per_source_cap: float | None, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    regions = make_world(N_LEARNABLE, N_NOISE, rng)
    ch = Challenger(reading=reading, per_source_cap=per_source_cap)
    ch.observe(regions)

    spend = np.zeros(len(regions))
    late_spend = np.zeros(len(regions))

    for cycle in range(CYCLES):
        alloc = ch.allocate(regions, BUDGET_PER_CYCLE)
        for r, n in zip(regions, alloc):
            if n > 0:
                r.practice(int(n), rng)
        ch.observe(regions)
        spend += alloc
        if cycle >= CYCLES // 2:
            late_spend += alloc

    noise_idx = [i for i, r in enumerate(regions) if r.is_noise]
    fair_share = len(noise_idx) / len(regions)

    total = max(spend.sum(), 1.0)
    late_total = max(late_spend.sum(), 1.0)
    noise_share = spend[noise_idx].sum() / total
    noise_share_late = late_spend[noise_idx].sum() / late_total

    learnable = [r for r in regions if not r.is_noise]
    mean_comp = float(np.mean([r.competence for r in learnable]))
    mean_ceiling = float(np.mean([r.ceiling for r in learnable]))

    # A2: does the signal this arm would hand the gate track real error?
    pred = np.array([r.predictive_var() if reading != "epistemic" else r.epistemic_var()
                     for r in regions])
    err = np.array([r.realized_error() for r in regions])
    rho = float(spearmanr(pred, err).statistic)

    a1 = (noise_share_late / fair_share) <= MAGNET_LIMIT
    a2 = rho >= CALIBRATION_MIN

    return {
        "arm": reading + ("+cap" if per_source_cap else ""),
        "noise_share_overall": round(float(noise_share), 4),
        "noise_share_late": round(float(noise_share_late), 4),
        "fair_share": round(fair_share, 4),
        "magnet_multiple": round(float(noise_share_late / fair_share), 2),
        "learnable_competence": round(mean_comp, 4),
        "learnable_ceiling": round(mean_ceiling, 4),
        "competence_shortfall": round(mean_ceiling - mean_comp, 4),
        "gate_calibration_rho": round(rho, 3),
        "A1_not_a_noise_magnet": bool(a1),
        "A2_gate_calibrated": bool(a2),
        "verdict": "PASS" if (a1 and a2) else "FAIL",
    }


def budget_sweep(seed: int) -> list[dict]:
    """Does the wasted budget actually cost capability, or only compute?

    At a generous budget every learnable region saturates even when half the
    practice is thrown away, so the magnet looks free. It is not free -- it is
    prepaid. This sweep tightens the budget until the waste starts showing up
    as lost competence, which is the regime any real deployment runs in.
    """
    out = []
    for cycles in (10, 25, 50, 100, 200):
        base = run_arm("predictive", None, seed)
        fixed = run_arm("reducible", None, seed)
        # rerun both at the shortened horizon
        global CYCLES
        saved, CYCLES = CYCLES, cycles
        try:
            base = run_arm("predictive", None, seed)
            fixed = run_arm("reducible", None, seed)
        finally:
            CYCLES = saved
        out.append(
            {
                "cycles": cycles,
                "predictive_competence": base["learnable_competence"],
                "reducible_competence": fixed["learnable_competence"],
                "competence_lost_to_magnet": round(
                    fixed["learnable_competence"] - base["learnable_competence"], 4
                ),
                "predictive_noise_share": base["noise_share_late"],
            }
        )
    return out


def main() -> int:
    arms = [
        ("predictive", None),          # gap set = what the gate needs
        ("predictive", 0.25),          # + Part III section 6.3 per-source cap
        ("epistemic", None),           # gap set = what the gap set needs
        ("reducible", None),           # candidate fix
    ]
    rows = [run_arm(reading, cap, SEED) for reading, cap in arms]

    hdr = (
        f"{'arm':<20}{'noise share':>13}{'x fair':>8}{'comp':>8}{'short':>8}"
        f"{'gate rho':>10}{'A1':>5}{'A2':>5}  verdict"
    )
    print(
        f"\nE1.4  Can the gap set tell 'hard' from 'random'?"
        f"   ({N_LEARNABLE} learnable + {N_NOISE} noise, {CYCLES} cycles)\n"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['arm']:<20}{r['noise_share_late']:>13.3f}"
            f"{r['magnet_multiple']:>8.1f}{r['learnable_competence']:>8.3f}"
            f"{r['competence_shortfall']:>8.3f}{r['gate_calibration_rho']:>10.2f}"
            f"{'ok' if r['A1_not_a_noise_magnet'] else 'no':>5}"
            f"{'ok' if r['A2_gate_calibrated'] else 'no':>5}"
            f"  {r['verdict']}"
        )
    print(
        f"\n  noise share  fraction of the practice budget spent on pure coin-flip"
        f" regions, second half of the run"
        f"\n  x fair       that share divided by {rows[0]['fair_share']:.3f}, the share"
        f" they would get by headcount (A1 needs <= {MAGNET_LIMIT})"
        f"\n  comp/short   mean competence reached on learnable regions, and how far"
        f" short of their ceilings that is"
        f"\n  gate rho     Spearman correlation between the arm's variance signal and"
        f" true error rate (A2 needs >= {CALIBRATION_MIN})\n"
    )

    sweep = budget_sweep(SEED)
    print("  Budget sweep -- is the wasted practice free, or only prepaid?\n")
    shdr = f"{'cycles':>8}{'predictive':>13}{'reducible':>12}{'lost':>9}{'noise share':>13}"
    print(shdr)
    print("-" * len(shdr))
    for s in sweep:
        print(
            f"{s['cycles']:>8}{s['predictive_competence']:>13.3f}"
            f"{s['reducible_competence']:>12.3f}"
            f"{s['competence_lost_to_magnet']:>9.3f}{s['predictive_noise_share']:>13.3f}"
        )
    print(
        "\n  Mean competence on learnable regions under each gap reading, at"
        "\n  shortening practice horizons. 'lost' is what the noise magnet costs"
        "\n  in capability once budget is actually scarce.\n"
    )

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_4_aleatoric_magnet.json"
    out.write_text(
        json.dumps({"seed": SEED, "cycles": CYCLES, "rows": rows, "budget_sweep": sweep}, indent=2)
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
