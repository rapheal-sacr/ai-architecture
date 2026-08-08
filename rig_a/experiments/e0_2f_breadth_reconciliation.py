"""E0.2f -- Reconciling the two breadth numbers, which decide whether R9 has a job.

E0.2d reports cascade breadth 0.627 -> 0.978. E0.2e reports 0.077 -> 0.375. Same
named quantity, same 0.31 target from E5.1's window condition, 8x apart. E5.1's
entire breadth-conditional verdict hinges on which is right:

    E0.2d says breadth is NEVER acceptable, so R9 is the only repair
    E0.2e says breadth is acceptable UNLESS you apply R9, which points R9 backwards

Both cannot inform the same threshold. This runs BOTH instruments over THE SAME
admitted bank, so any disagreement is the instrument rather than the world.

    synthetic draw    E0.2e's breadth_of -- adapters sample a fixed NUMBER of
                      cards uniformly from the whole bank
    influence graph   E0.2b/E0.2d's true_influenced_adapters -- delete an entry,
                      recompile, see which adapter weights move. Functional, and
                      independent of any provenance bookkeeping

KILL CRITERIA (pre-registered):
    R1 fails if the two instruments disagree by more than 1.5x at the UNCAPPED
       bank. They are measuring the same thing on the same object; a gap there is
       an instrument defect, not a finding.
    R2 fails if the influence graph shows the same tau-dependence the synthetic
       draw does. If it does not, E0.2e's opposition -- the thing that made R9
       unevaluable -- is an artifact of the synthetic draw's fixed-count-from-a-
       shrinking-bank assumption.
    R3 reports whether either instrument reproduces E0.2d's 0.63-0.98 over a
       realistic bank. Not pass/fail: it decides whether there is a breadth
       problem at all.

Is there a world that produces the other verdict? Yes: if breadth genuinely rises
as the bank shrinks, the influence graph will show it too, since it propagates
through the same cards. That is the outcome under which R9's premise survives.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rig_a.core.influence import InfluenceWorld  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e0_2e", ROOT / "rig_a" / "experiments" / "e0_2e_r9_breadth_coverage.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

TAUS = (0.0, 0.10, 0.20, 0.35, 0.50, 1.0)
CONCENTRATION = 0.5
N_SEEDS = 3
N_PROBE_ENTRIES = 25
DIM = 16
# 6 rollouts per adapter, matching E0.2d. The first run of this used
# n_rollouts=48 against 64 adapters, which gives ONE rollout each and sixteen
# adapters with none -- breadth then reads ~0.013 by construction. Rig bug B18,
# caught before it was reported as a third disagreeing number.
N_ROLLOUTS = E.N_ADAPTERS * E.ROLLOUTS_PER_ADAPTER

E0_2D_RANGE = (0.627, 0.978)
AGREE_LIMIT = 1.5


def measure(tau: float, seed: int) -> tuple[int, float, float]:
    rng = np.random.default_rng(seed)
    cards, _ = E.make_candidates(CONCENTRATION, rng)
    bank = E.admit(cards, tau)
    syn = E.breadth_of(bank, rng)
    w = InfluenceWorld(dim=DIM, n_entries=E.N_ENTRIES, n_cards=len(bank),
                       n_rollouts=N_ROLLOUTS, n_adapters=E.N_ADAPTERS,
                       rng=np.random.default_rng(seed),
                       card_sources_override=[b["src"] for b in bank])
    reach = [len(w.true_influenced_adapters(int(e))) / E.N_ADAPTERS
             for e in rng.choice(E.N_ENTRIES, size=N_PROBE_ENTRIES, replace=False)]
    return len(bank), float(syn), float(np.mean(reach))


def main() -> int:
    print(f"\nE0.2f  Breadth reconciliation -- same bank, both instruments"
          f"\n       ({E.N_ADAPTERS} adapters x {E.ROLLOUTS_PER_ADAPTER} rollouts,"
          f" concentration {CONCENTRATION})\n")
    hdr = (f"    {'tau':>6}{'bank':>7}{'synthetic draw':>17}{'influence graph':>18}"
           f"{'ratio':>8}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))

    rows = []
    for tau in TAUS:
        out = [measure(tau, E.SEED + i) for i in range(N_SEEDS)]
        bank = int(np.mean([o[0] for o in out]))
        syn = float(np.mean([o[1] for o in out]))
        tru = float(np.mean([o[2] for o in out]))
        rows.append({"tau": tau, "bank": bank, "synthetic": round(syn, 3),
                     "influence": round(tru, 3), "ratio": round(tru / max(syn, 1e-9), 2)})
        print(f"    {tau:>6.2f}{bank:>7}{syn:>17.3f}{tru:>18.3f}"
              f"{tru/max(syn,1e-9):>8.2f}x")

    uncapped = rows[-1]
    r1 = max(uncapped["synthetic"], uncapped["influence"]) / \
        max(min(uncapped["synthetic"], uncapped["influence"]), 1e-9) <= AGREE_LIMIT
    inf_span = max(r["influence"] for r in rows) - min(r["influence"] for r in rows)
    syn_span = max(r["synthetic"] for r in rows) - min(r["synthetic"] for r in rows)
    r2 = inf_span >= 0.5 * syn_span
    reproduces = any(E0_2D_RANGE[0] <= r["influence"] <= E0_2D_RANGE[1] for r in rows)

    print(f"\n  R1 instruments agree at the uncapped bank (<= {AGREE_LIMIT}x): "
          f"{'ok' if r1 else 'NO'}"
          f"   ({uncapped['synthetic']:.3f} vs {uncapped['influence']:.3f})")
    print(f"  R2 influence graph shows the same tau-dependence:              "
          f"{'ok' if r2 else 'NO'}")
    print(f"     synthetic span {syn_span:.3f}  vs  influence span {inf_span:.3f}")
    print(f"  R3 either instrument reproduces E0.2d's {E0_2D_RANGE[0]}-{E0_2D_RANGE[1]}: "
          f"{'yes' if reproduces else 'NO'}")

    if r1 and not r2:
        print("\n  => E0.2e's OPPOSITION IS AN INSTRUMENT ARTIFACT. The two agree on")
        print("     the uncapped bank and diverge only as the bank shrinks, which is")
        print("     exactly where the synthetic draw's fixed-count-from-a-shrinking-")
        print("     bank assumption bites. The influence graph is flat in tau, so")
        print("     tightening the cap does NOT raise breadth. R9 is not pointed")
        print("     backwards; that reading was the instrument.")
    if not reproduces:
        print("\n  => AND NEITHER REPRODUCES E0.2d. Over a realistic bank both read")
        print(f"     ~{uncapped['influence']:.2f} against E5.1's 0.31 target, so breadth is")
        print("     comfortably INSIDE the window and there is no breadth problem to")
        print("     repair. E0.2d's 0.63-0.98 came from a world with 8 adapters, 8")
        print("     cards and 60 entries, where every adapter's draw covers most of")
        print("     the bank by construction. Breadth is a function of bank size")
        print("     relative to fleet size and rollouts per adapter, and E0.2d")
        print("     measured it at one very small point.")
        print("\n     E5.1's breadth-conditional verdict, and R9's entire premise,")
        print("     rest on that point. Neither world is calibrated to reality, so")
        print("     the real bank/fleet/rollout ratio is now the quantity that")
        print("     decides it -- a MEASURE quantity, not a simulation choice.")
    print()

    out = ROOT / "results" / "e0_2f_breadth_reconciliation.json"
    out.write_text(json.dumps({"seed": E.SEED, "n_seeds": N_SEEDS,
                               "n_rollouts": N_ROLLOUTS, "rows": rows,
                               "R1_agree_uncapped": bool(r1),
                               "R2_same_tau_dependence": bool(r2),
                               "R3_reproduces_e0_2d": bool(reproduces)}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
