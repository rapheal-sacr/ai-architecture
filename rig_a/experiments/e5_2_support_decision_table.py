"""E5.2 -- The decision table for support redundancy. Not a finding; a lookup.

Support redundancy is the second of E5.1's four gating quantities, and unlike H
it is NOT measurable on this rig: it is a property of harvested-probe provenance
in a real deployment, and this project has no interaction corpus. Producing a
synthetic number here and calling it a measurement is precisely what B9 and B11
already did twice.

So this is not an experiment. It is the decision rule the measurement will need,
computed in advance, so that when someone samples real harvested probes the
answer is a lookup rather than another sweep.

WHAT GETS MEASURED IN THE REAL WORLD (docs/support_redundancy_procedure.md):

    k   support size -- how many ledger entries a harvested probe's expected
        outcome actually depends on
    m   how many of those k must survive for the probe to still pass

Stated as "m of k" rather than as a fraction, because fractions are ambiguous at
small k: at k = 3, thresholds 0.50 and 0.34 are the SAME constraint (both mean
2 of 3), and the first version of this table reported them as separate rows with
identical numbers.

TWO MECHANISMS, REPORTED SEPARATELY -- and conflating them was the first
version's real defect. It produced a table where k = 10 looked WORSE than k = 3,
which is backwards for anything called redundancy:

    COMPILE ADEQUACY (I11). At draw fraction f, an item needs m of its k entries
        actually DRAWN before decay is even relevant. At f = 0.25 only 16% of
        3-of-3 items clear that bar against 47% of 3-of-10 items. Redundancy
        helps.
    DECAY ROBUSTNESS (C6). GIVEN an item passed, does decay break it?
        Redundancy helps here too.

Both favour redundancy, but over-forgetting is a rate CONDITIONAL on having
passed, so the denominator moves with k and the combined number is non-monotone.
The two tables below are each monotone in the direction they claim to inform;
the single table that preceded them was not, and was unusable as a lookup. B13.

NO KILL CRITERIA, deliberately. There is no claim under test. Registering one
would imply this measures the design, and it does not -- it measures the
consequences of a parameter nobody has measured yet.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SHAPES = ((3, 2), (5, 2), (5, 3), (8, 2), (8, 3), (10, 3), (10, 4))
DECAY_RATES = (0.05, 0.15, 0.25)
DRAW_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
REGIONS = 16
PER_REGION = 50
ITEMS_PER_REGION = 40
N_SEEDS = 4
SEED = 20260806
C6_TOLERANCE = 0.20
I11_FLOOR = 0.80          # a compile that covers under 80% of a region's probes
                         # has not adequately covered it


def measure(k: int, m: int, draw_fraction: float, decay_rate: float, seed: int):
    """Returns (worst-region pass rate before decay, worst-region over-forgetting)."""
    rng = np.random.default_rng(seed)
    n = REGIONS * PER_REGION
    region = np.repeat(np.arange(REGIONS), PER_REGION)
    rates = 1.0 / np.arange(1, REGIONS + 1, dtype=float)
    rates /= rates.sum()
    usage = rates[region] * rng.uniform(0.5, 1.5, size=n)

    items = []
    for r in range(REGIONS):
        pool = np.where(region == r)[0]
        for _ in range(ITEMS_PER_REGION):
            items.append((r, set(rng.choice(pool, size=k, replace=False).tolist())))

    cap = max(int(draw_fraction * n), 1)
    per = max(cap // REGIONS, 1)

    def stratified(live):
        out = []
        for r in range(REGIONS):
            p = live[region[live] == r]
            if len(p):
                out.extend(rng.choice(p, size=min(per, len(p)), replace=False).tolist())
        return set(out[:cap])

    before = stratified(np.arange(n))
    # per-region decay, the weighting-rule-compliant policy
    keep = []
    for r in range(REGIONS):
        pool = np.where(region == r)[0]
        order = pool[np.argsort(usage[pool])]
        keep.extend(order[int(decay_rate * len(order)):].tolist())
    after = stratified(np.sort(np.array(keep)))

    pass_rates, over_rates = [], []
    for r in range(REGIONS):
        sel = [s for (rr, s) in items if rr == r]
        passed = [s for s in sel if len(s & before) >= m]
        pass_rates.append(len(passed) / len(sel))
        if len(passed) >= 5:
            lost = [s for s in passed if len(s & after) < m]
            over_rates.append(len(lost) / len(passed))
    # worst region = lowest coverage, highest forgetting
    return min(pass_rates), (max(over_rates) if over_rates else 0.0)


def agg(k, m, f, d):
    out = [measure(k, m, f, d, SEED + i) for i in range(N_SEEDS)]
    return float(np.mean([x[0] for x in out])), float(np.mean([x[1] for x in out]))


def main() -> int:
    print("\nE5.2  Support redundancy -- decision table, not a measurement\n")
    print(f"  {REGIONS} regions, stratified draw and per-region decay,"
          f" mean of {N_SEEDS} seeds.")
    print("  Support written as 'm of k'. Worst region reported, per the")
    print("  weighting rule.\n")

    rows = []

    print("  TABLE A -- COMPILE ADEQUACY (I11): worst-region fraction of probes with")
    print(f"  enough support DRAWN to pass at all. Floor {I11_FLOOR:.2f}.\n")
    hdr = f"    {'k':>4}{'m':>4}" + "".join(f"{'f=' + format(f, '.2f'):>12}"
                                            for f in DRAW_FRACTIONS)
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for k, m in SHAPES:
        cells = []
        for f in DRAW_FRACTIONS:
            pr, _ = agg(k, m, f, 0.15)
            cells.append(f"{pr:>9.3f}{'  ok' if pr >= I11_FLOOR else '  --'}")
        print(f"    {k:>4}{m:>4}" + "".join(cells))

    print(f"\n  TABLE B -- DECAY ROBUSTNESS (C6): worst-region over-forgetting GIVEN")
    print(f"  the probe passed. Full draw, so I11 is not confounding."
          f" Tolerance {C6_TOLERANCE:.2f}.\n")
    hdr2 = f"    {'k':>4}{'m':>4}" + "".join(f"{'d=' + format(d, '.2f'):>12}"
                                             for d in DECAY_RATES)
    print(hdr2); print("    " + "-" * (len(hdr2) - 4))
    for k, m in SHAPES:
        cells = []
        for d in DECAY_RATES:
            pr, ov = agg(k, m, 1.0, d)
            rows.append({"k": k, "m": m, "decay_rate": d, "draw_fraction": 1.0,
                         "pass_before": round(pr, 3), "over_forget_worst": round(ov, 3),
                         "clears_C6": bool(ov <= C6_TOLERANCE)})
            cells.append(f"{ov:>9.3f}{'  ok' if ov <= C6_TOLERANCE else '  --'}")
        print(f"    {k:>4}{m:>4}" + "".join(cells))

    print("\n  THE LOOKUP. Shapes clearing C6 at every swept decay rate:")
    ok_shapes = []
    for k, m in SHAPES:
        vals = [r for r in rows if r["k"] == k and r["m"] == m]
        if all(v["clears_C6"] for v in vals):
            ok_shapes.append((k, m))
            print(f"      {m} of {k}")
    if not ok_shapes:
        print("      none")

    print("\n  So: sample real harvested probes, measure (k, m), and read off")
    print("  whether C6 binds. If real probes rest on 3 entries needing 2, C6 is")
    print("  live and the draw cap must stay high. If they rest on 8-10 needing")
    print("  2-3, C6 stops binding and the cap can come down -- which relaxes the")
    print("  recompile cost that E0.1's A6 was managing.\n")

    out = ROOT / "results" / "e5_2_support_decision_table.json"
    out.write_text(json.dumps(
        {"seed": SEED, "regions": REGIONS, "c6_tolerance": C6_TOLERANCE,
         "i7_floor": I11_FLOOR, "shapes_clearing_C6": ok_shapes, "rows": rows},
        indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
