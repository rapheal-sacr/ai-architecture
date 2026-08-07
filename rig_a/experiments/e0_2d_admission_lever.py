"""E0.2d -- Is card-bank admission control actually a lever on cascade breadth?

E0.2c reported `D3_admission_is_a_lever: true` on the strength of a sweep that
moved entry multiplicity from 1 to 6 and observed BOTH mean card cosine
(0.046 -> 0.779) and cascade breadth (63.1% -> 98.8%) rising together. That is a
common cause, not an intervention. It establishes that multiplicity drives
breadth. It does not establish that intervening on cosine changes breadth, and
the latter is the claim the design needs, because cosine is what SESA's
admission rule actually scores.

The two quantities are different objects:

    mean card cosine     similarity of card CONTENT -- what admission scores
    cascade breadth      driven by PROVENANCE overlap -- how many cards share
                         a source entry

E0.2c's world made them track each other by construction, because card content
was literally the mean of its source entries. Real cards are distillations: two
cards can capture different patterns in the same entries and come out
near-orthogonal in content while sharing every source. SESA's cos <= 0.93 is a
redundancy filter on content. Nothing in it bounds provenance overlap.

METHOD -- two panels that separate the common cause:

    A  vary provenance (multiplicity), content coupled as in E0.2c
    B  HOLD PROVENANCE FIXED and vary content rotation, which moves card cosine
       while every source set stays identical

If breadth is flat across panel B while cosine collapses, then cosine is not a
lever on breadth, and E0.2c's D3 inverts.

KILL CRITERIA (pre-registered):
    L1 fails if breadth does not respond to provenance overlap in panel A --
       that would mean the causal story is wrong entirely.
    L2 fails if breadth DOES NOT respond to content cosine in panel B, i.e.
       breadth range across the rotation sweep is under 5 points while cosine
       moves more than 0.3. L2 failing is the interesting outcome: it means the
       design's only stated admission instrument acts on the wrong axis, and no
       setting of 0.93 controls deletion cost.

Is there a world that produces the other verdict? Yes, and it is E0.2c's: if
card content is the raw source mean, cosine and provenance are the same
measurement and breadth tracks cosine exactly. Panel B at rotation 0 is that
world, and it is included so the contrast is visible rather than asserted.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import InfluenceWorld  # noqa: E402

DIM = 16
N_ENTRIES = 60
N_CARDS = 8
N_ROLLOUTS = 48
N_ADAPTERS = 8
SEED = 20260806
N_WORLDS = 8

MULTIPLICITIES = (1, 2, 3, 4, 6)
ROTATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_M = 4                       # panel B holds provenance here
BREADTH_RANGE_LIMIT = 0.05        # L2
COSINE_MOVE_MIN = 0.30            # L2


def measure(mult: int, rotation: float) -> dict:
    breadths, cosines, overlaps = [], [], []
    for w_i in range(N_WORLDS):
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=N_CARDS,
            n_rollouts=N_ROLLOUTS, n_adapters=N_ADAPTERS,
            rng=np.random.default_rng(SEED + w_i),
            card_overlap=float(mult), content_rotation=rotation,
        )
        cosines.append(w.mean_card_cosine())
        overlaps.append(w.provenance_overlap())
        for eid in range(N_ENTRIES):
            t = w.true_influenced_adapters(eid)
            if t:
                breadths.append(len(t))
    return {
        "multiplicity": mult,
        "content_rotation": rotation,
        "mean_card_cosine": round(float(np.mean(cosines)), 3),
        "provenance_overlap": round(float(np.mean(overlaps)), 3),
        "cascade_fraction": round(float(np.mean(breadths)) / N_ADAPTERS, 3),
    }


def main() -> int:
    panel_a = [measure(m, 0.0) for m in MULTIPLICITIES]
    panel_b = [measure(FIXED_M, t) for t in ROTATIONS]

    print(f"\nE0.2d  Is admission control a lever on cascade breadth?"
          f"   ({N_WORLDS} worlds, {N_ADAPTERS} adapters)\n")

    print("Panel A - vary PROVENANCE (multiplicity), content coupled  [E0.2c's sweep]")
    print("-" * 74)
    print(f"  {'cards/entry':>12}{'prov overlap':>14}{'card cosine':>13}{'cascade':>10}")
    for r in panel_a:
        print(f"  {r['multiplicity']:>12}{r['provenance_overlap']:>14.3f}"
              f"{r['mean_card_cosine']:>13.3f}{r['cascade_fraction']:>10.1%}")

    print(f"\nPanel B - HOLD provenance fixed (m={FIXED_M}), vary content rotation")
    print("-" * 74)
    print(f"  {'rotation':>12}{'prov overlap':>14}{'card cosine':>13}{'cascade':>10}")
    for r in panel_b:
        print(f"  {r['content_rotation']:>12.2f}{r['provenance_overlap']:>14.3f}"
              f"{r['mean_card_cosine']:>13.3f}{r['cascade_fraction']:>10.1%}")

    a_breadth = [r["cascade_fraction"] for r in panel_a]
    b_breadth = [r["cascade_fraction"] for r in panel_b]
    b_cos = [r["mean_card_cosine"] for r in panel_b]
    b_prov = [r["provenance_overlap"] for r in panel_b]

    l1 = (max(a_breadth) - min(a_breadth)) > BREADTH_RANGE_LIMIT
    cos_moved = (max(b_cos) - min(b_cos)) > COSINE_MOVE_MIN
    l2 = (max(b_breadth) - min(b_breadth)) > BREADTH_RANGE_LIMIT

    print(f"\n  panel A breadth range: {max(a_breadth) - min(a_breadth):.3f}"
          f"   (provenance moves breadth)")
    print(f"  panel B cosine range:  {max(b_cos) - min(b_cos):.3f}"
          f"   provenance range: {max(b_prov) - min(b_prov):.3f} (held fixed)")
    print(f"  panel B breadth range: {max(b_breadth) - min(b_breadth):.3f}"
          f"   (does cosine move breadth?)")

    print(f"\n  L1 breadth responds to provenance:  {'ok' if l1 else 'NO'}")
    print(f"  L2 breadth responds to cosine:      {'ok' if l2 else 'NO'}"
          f"   (cosine actually moved: {'yes' if cos_moved else 'no'})")

    if l1 and not l2 and cos_moved:
        print("\n  => E0.2c's D3 INVERTS. Content cosine moves freely with provenance")
        print("     held fixed, and cascade breadth does not follow. SESA's cos <= 0.93")
        print("     scores content; breadth is set by provenance overlap. The design has")
        print("     NO stated lever on cascade breadth -- this is a missing control")
        print("     surface, not a threshold that needs retuning.")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_2d_admission_lever.json"
    out.write_text(json.dumps(
        {"seed": SEED, "panel_a": panel_a, "panel_b": panel_b,
         "L1_breadth_tracks_provenance": bool(l1),
         "L2_breadth_tracks_cosine": bool(l2),
         "cosine_actually_moved": bool(cos_moved),
         "d3_inverts": bool(l1 and not l2 and cos_moved)}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
