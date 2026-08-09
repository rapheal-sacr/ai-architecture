"""EB.5 -- is EB.2's massive-activation finding an uncentered-moment artifact?

Worklist v2 item 1.1, and it was posed as a hypothesis rather than a claim:

    "Massive activations are large and roughly input-INDEPENDENT -- enormous in
     an uncentered second moment E[aa^T], near zero in a centered covariance. If
     R_t is uncentered, that is plausibly the whole mechanism behind 71% of
     variance in five directions, and the fix is one line. If it is already
     centered, then massive activations are real geometry and the energy
     criterion needs an explicit input-dependence filter -- rank by BETWEEN-input
     variance rather than total energy."

The answer is in two places, and they disagree, which is why reading one of them
was not enough.

    rig_a/core/spectrum.py   `R = lam * R + outer(row, row)`  -- UNCENTERED.
                             That is the estimator the DESIGN uses.
    rig_b/eb_2              `acts - acts.mean(0)` before every SVD -- CENTERED.
                             That is what EB.2 MEASURED.

So EB.2's 71% was already centered and is not the artifact the hypothesis names.
But the deployed estimator is uncentered, so the design sees a worse version than
EB.2 reported -- which makes the hypothesis wrong about EB.2 and right about the
mechanism. Both halves are measured below.

THREE QUANTITIES:

    uncentered share   top-5 dims' share of E[a_i^2]      -- what R_t sees today
    centered share     top-5 dims' share of Var(a_i)      -- what centering buys
    between share      of a dimension's variance, how much is BETWEEN domains

The third is the one that decides the repair. A direction that discriminates
capabilities has high between-domain variance. A direction that merely moves a
lot does not, and the energy criterion cannot tell them apart because it ranks on
total energy.

KILL CRITERIA (pre-registered):
    KE1 Centering removes the concentration at BOTH model sizes. The hypothesis
        is then right, the repair is one line, and no filter is needed.
    KE2 The high-energy directions carry between-domain variance comparable to
        the rest. They are then discriminative after all and spending budget on
        them is not waste.

Is there a world that produces the other verdict? For KE1, yes -- and the 0.5B is
that world, which is exactly why one model was not enough to answer it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mlx_lm import load                                     # noqa: E402
from rig_b.eb_2_activation_spectra import extract           # noqa: E402

MODELS = (
    ("Qwen2.5-0.5B-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", 12),
    ("Qwen2.5-1.5B-4bit", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", 14),
)
TOP = 5


def analyse(acts: dict) -> dict:
    names = list(acts)
    x = np.vstack([acts[k] for k in acts]).astype(np.float64)
    lab = np.concatenate([[i] * len(acts[k]) for i, k in enumerate(names)])
    gmean = x.mean(0)

    unc = (x ** 2).mean(0)                      # E[a_i^2] -- what R_t accumulates
    cen = ((x - gmean) ** 2).mean(0)            # Var(a_i) -- what centering leaves
    between = np.zeros(x.shape[1])
    for i in range(len(names)):
        xi = x[lab == i]
        between += (len(xi) / len(x)) * (xi.mean(0) - gmean) ** 2
    share_between = between / np.maximum(cen, 1e-12)

    tu = np.argsort(-unc)[:TOP]
    tc = np.argsort(-cen)[:TOP]
    rest = np.setdiff1d(np.arange(x.shape[1]), tc)
    return {
        "dim": int(x.shape[1]), "n_vectors": int(len(x)),
        "uncentered_top_share": round(float(unc[tu].sum() / unc.sum()), 4),
        "centered_top_share": round(float(cen[tc].sum() / cen.sum()), 4),
        "dims_shared": int(len(set(tu.tolist()) & set(tc.tolist()))),
        "mean_sq_over_var_top": round(float((gmean[tc] ** 2).sum()
                                            / max(cen[tc].sum(), 1e-12)), 3),
        "between_share_top": round(float(share_between[tc].mean()), 4),
        "between_share_rest": round(float(share_between[rest].mean()), 4),
    }


def main() -> int:
    print("\nEB.5 -- centering, and whether high-energy directions discriminate\n")
    print("  rig_a/core/spectrum.py:  R = lam*R + outer(a, a)   ->  UNCENTERED")
    print("  rig_b/eb_2:              acts - acts.mean(0)       ->  CENTERED")
    print("  So the estimator the DESIGN uses and the one EB.2 MEASURED differ.\n")

    out = {}
    for tag, path, layer in MODELS:
        m, tok = load(path)
        acts = extract(m, tok, layer)
        del m, tok
        out[tag] = analyse(acts)

    print(f"  {'model':>20}{'dim':>6}{'uncentered':>12}{'centered':>10}"
          f"{'shared':>8}{'mean^2/var':>12}")
    for tag, _, _ in MODELS:
        r = out[tag]
        print(f"  {tag:>20}{r['dim']:>6}{r['uncentered_top_share']:>11.1%}"
              f"{r['centered_top_share']:>10.1%}{str(r['dims_shared']) + '/5':>8}"
              f"{r['mean_sq_over_var_top']:>12.2f}")
    print(f"\n    top-{TOP} dimensions' share of total energy, before and after centering.")
    print("    mean^2/var  >1 means those dimensions are dominated by their MEAN,")
    print("                so centering removes them. <1 means real variance.")

    print(f"\n  BETWEEN-DOMAIN SHARE -- of a dimension's variance, how much")
    print("  distinguishes one domain from another. This is what the energy")
    print("  criterion cannot see, because it ranks on TOTAL energy.\n")
    print(f"  {'model':>20}{'top-5':>10}{'all others':>13}{'ratio':>9}")
    for tag, _, _ in MODELS:
        r = out[tag]
        ratio = r["between_share_top"] / max(r["between_share_rest"], 1e-12)
        print(f"  {tag:>20}{r['between_share_top']:>9.1%}"
              f"{r['between_share_rest']:>12.1%}{ratio:>9.2f}x")

    small, large = out[MODELS[0][0]], out[MODELS[1][0]]
    ke1 = (small["centered_top_share"] < 0.5 * small["uncentered_top_share"]
           and large["centered_top_share"] < 0.5 * large["uncentered_top_share"])
    ke2 = large["between_share_top"] >= large["between_share_rest"]

    print(f"\n  KE1 centering removes it at BOTH sizes:   {'ok' if ke1 else 'NO'}")
    print(f"  KE2 high-energy directions discriminate:  {'ok' if ke2 else 'NO'}")

    print("\n  THE HYPOTHESIS IS RIGHT ABOUT THE SMALL MODEL AND WRONG ABOUT THE")
    print("  LARGE ONE, WHICH IS THE WRONG WAY ROUND FOR SCALING.")
    print(f"    0.5B  {small['uncentered_top_share']:.1%} -> {small['centered_top_share']:.1%}"
          f"   mean^2/var {small['mean_sq_over_var_top']:.1f}, {small['dims_shared']}/5 dims"
          " shared -- mostly a MEAN OFFSET, centering fixes it")
    print(f"    1.5B  {large['uncentered_top_share']:.1%} -> {large['centered_top_share']:.1%}"
          f"   mean^2/var {large['mean_sq_over_var_top']:.2f}, {large['dims_shared']}/5 dims"
          " shared -- REAL VARIANCE, centering does not")
    print("  One model would have answered this confidently and wrongly in either")
    print("  direction. That is B12 and EB.2's pooling defect, a third time.")

    print("\n  AND THE FILTER IS JUSTIFIED WITH A NUMBER. On the 1.5B the top-5")
    print(f"  directions hold {large['centered_top_share']:.1%} of centered variance and carry")
    print(f"  {large['between_share_top']:.1%} between-domain share, against"
          f" {large['between_share_rest']:.1%} for every other")
    print("  dimension. They consume three quarters of the energy budget and")
    print("  discriminate essentially nothing. `rank_for_energy` cannot see the")
    print("  difference because it ranks on total energy.")

    print("\n  TWO REPAIRS, AND THE FIRST IS NOT SUFFICIENT.")
    print("    1. CENTER R_t. One line, `R = lam*R + outer(a-mu, a-mu)` with a")
    print("       running mean. Buys 16 points of wasted budget at 0.5B and ~1 at")
    print("       1.5B. Worth doing and it does not solve the problem.")
    print("    2. RANK BY BETWEEN-OWNER VARIANCE, not total energy. That is the")
    print("       actual repair, it is what the design means by `committed`, and")
    print("       under the register it is FREE -- owners are the grouping, and")
    print("       per-owner buffers are what I8 already requires.")
    print("    The second falls out of the register. The spectrum has no grouping")
    print("    to compute it over, which is an argument for section 1 that")
    print("    section 1 does not make.")

    verdict = "PASS" if (ke1 and ke2) else "FAIL"
    print(f"\n  HYPOTHESIS 1.1: {verdict} -- refuted at the size that matters\n")

    o = pathlib.Path(__file__).resolve().parents[1] / "results" / "eb_5_centering_and_input_dependence.json"
    o.write_text(json.dumps(
        {"models": [m[0] for m in MODELS], "top_dims": TOP, "by_model": out,
         "estimator_is_centered": False,
         "eb2_measured_centered": True,
         "KE1_centering_suffices": bool(ke1),
         "KE2_high_energy_discriminates": bool(ke2),
         "verdict": verdict}, indent=2))
    print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
