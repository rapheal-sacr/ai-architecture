"""E0.2c -- The deletion ceiling under policies that are not the worst available.

E0.2b computed a tombstone-rate ceiling of `capacity / (cascade x recompile)`
and this plan promoted it to "the one genuinely architectural finding." The
arithmetic is right and the policy it assumes is the most expensive one in the
space: eager, per-tombstone, fully-recompiling, with parallelism as the only
relief. Three separate things were folded into that number.

COMPLIANCE THROUGHPUT IS NOT COMPETENCE THROUGHPUT. What deletion requires is
that the entry stop influencing outputs. That is satisfied by DISABLING the
affected adapters, immediately -- the gate already routes to retrieval during
plasticity transients, so the fallback path exists and L7 already relies on it.
Recompilation restores competence; it is not what makes deletion sound. E0.2b
therefore reported a service-quality limit as though it were a correctness rate
limit, and those have very different consequences.

BATCHING AMORTISES, AND MOST WHERE E0.2b SAID IT WAS WORST. If cascades overlap
heavily, the union over a window saturates after a few tombstones, so N
deletions cost one pass over the union rather than N passes over each cascade.
Cost per deletion then FALLS as cascade breadth rises -- inverting the shape of
E0.2b's surface at exactly the end it called catastrophic.

CASCADE BREADTH IS A KNOB THE DESIGN ALREADY HOLDS. L3 admission control caps
card-bank cosine at 0.93, which bounds how much cards duplicate one another,
which bounds how many cards one entry feeds, which is cascade breadth. So the
result is not a ceiling but a COUPLING between card-bank density and deletion
throughput, mediated by a threshold the design already sets.

KILL CRITERIA (pre-registered):
    D1 fails if correctness latency under disable-then-recompile is not
       O(disable cost) -- i.e. if disabling does not in fact decouple
       compliance from recompilation.
    D2 fails if batching does not reduce per-deletion cost at high cascade
       breadth, i.e. if the union does not saturate.
    D3 fails if cascade breadth is insensitive to card-bank overlap.

D3's RESULT HAS BEEN SUPERSEDED -- see E0.2d. This sweep varies entry
multiplicity and observes card cosine AND cascade breadth rising together, which
is a common cause, not an intervention. E0.2d holds provenance fixed and moves
content cosine alone: breadth does not follow. So `D3_admission_is_a_lever` as
reported here is WRONG. Admission control scores card CONTENT; breadth is set by
PROVENANCE overlap, and in this world those coincide only because card content
is built from source entries.

TWO FURTHER LIMITS ON THE NUMBERS BELOW.

Fleet size is 8 and BATCH_SIZES runs to 16, so at the largest batches the window
exceeds the fleet and every batch necessarily touches every adapter -- "the union
saturates at fleet size" is then arithmetic, not a discovered property. E0.2b
used a fleet of 64, so the eager-vs-batched comparison ACROSS the two experiments
is confounded by an 8x fleet difference and the raw deletions/day figures are not
comparable. The general form is what to quote:

    cost per deletion = |union(B)| x recompile / B,  and |union(B)| -> fleet
    so batching gain  = breadth x B / fleet, capped at B once breadth ~ fleet

The gain is the batch size and it is unbounded in B, which is why breadth drops
out. But that RELOCATES the cost rather than removing it: B is bounded by how
long competence may stay degraded, because the window must fill before the
recompile fires. At a low deletion arrival rate a large window means a long
degraded period. Competence-restoration latency as a function of arrival rate is
the honest successor to E0.2b's ceiling, and it is NOT measured here.

D1's ratio is 8h/1s -- two chosen constants, not a measurement. It is sound as a
scale argument (disabling is orders of magnitude cheaper than recompiling) and
should not be quoted to three significant figures.

Is there a world that produces the other verdict? For D2, yes: with disjoint
cascades the union grows linearly and batching buys nothing, which is precisely
the low-overlap end of the sweep below. For D3, yes: if retrieval-flip effects
dominate, breadth could stay flat as overlap falls.
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

RECOMPILE_HOURS = 8.0
DISABLE_SECONDS = 1.0          # flipping an adapter off is a config write
PARALLEL_RECOMPILES = 4
BATCH_SIZES = (1, 2, 4, 8, 16)
MULTIPLICITIES = (1, 2, 3, 4, 6)   # cards each participating entry feeds


def measure(mult: int) -> dict:
    breadths, unions = [], {b: [] for b in BATCH_SIZES}
    cosines = []

    for w_i in range(N_WORLDS):
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=N_CARDS,
            n_rollouts=N_ROLLOUTS, n_adapters=N_ADAPTERS,
            rng=np.random.default_rng(SEED + w_i), card_overlap=float(mult),
        )
        cosines.append(w.mean_card_cosine())
        rng = np.random.default_rng(SEED + 500 + w_i)

        for eid in range(N_ENTRIES):
            t = w.true_influenced_adapters(eid)
            if t:
                breadths.append(len(t))

        for b in BATCH_SIZES:
            for _ in range(12):
                batch = rng.choice(N_ENTRIES, size=b, replace=False).tolist()
                unions[b].append(len(w.union_cascade(batch)))

    breadth = float(np.mean(breadths))
    out = {
        "entry_multiplicity": mult,
        "mean_card_cosine": round(float(np.mean(cosines)), 3),
        "cascade_breadth": round(breadth, 2),
        "cascade_fraction": round(breadth / N_ADAPTERS, 3),
        "batches": [],
    }

    cap_hours_per_day = 24.0 * PARALLEL_RECOMPILES
    for b in BATCH_SIZES:
        union = float(np.mean(unions[b]))
        cost_per_deletion = union * RECOMPILE_HOURS / b
        out["batches"].append({
            "batch": b,
            "union_adapters": round(union, 2),
            "adapter_hours_per_deletion": round(cost_per_deletion, 2),
            "deletions_per_day": round(cap_hours_per_day / cost_per_deletion, 2),
        })
    return out


def main() -> int:
    rows = [measure(m) for m in MULTIPLICITIES]

    print(f"\nE0.2c  Deletion throughput under realistic policies"
          f"   ({N_WORLDS} worlds, {N_ADAPTERS} adapters, {RECOMPILE_HOURS}h recompile)\n")

    print("D3 - is cascade breadth a function of card-bank duplication?")
    print("-" * 72)
    print(f"  {'cards/entry':>12}{'mean card cos':>15}{'cascade':>10}{'fraction':>11}")
    for r in rows:
        print(f"  {r['entry_multiplicity']:>12}{r['mean_card_cosine']:>15.3f}"
              f"{r['cascade_breadth']:>10.2f}{r['cascade_fraction']:>11.1%}")

    print("\nD2 - does batching amortise, and where most?")
    print("-" * 72)
    hdr = f"  {'cards/entry':>12}" + "".join(f"{'b=' + str(b):>11}" for b in BATCH_SIZES)
    print(hdr + "     (deletions/day)")
    for r in rows:
        print(f"  {r['entry_multiplicity']:>12}"
              + "".join(f"{c['deletions_per_day']:>11.2f}" for c in r["batches"]))

    print("\n  union size (adapters touched per window):")
    for r in rows:
        print(f"  {r['entry_multiplicity']:>12}"
              + "".join(f"{c['union_adapters']:>11.2f}" for c in r["batches"]))

    # D2: batching must help at the high-overlap end
    hi = rows[-1]["batches"]
    d2 = hi[-1]["deletions_per_day"] > hi[0]["deletions_per_day"]
    # D3: breadth must respond to overlap
    d3 = abs(rows[-1]["cascade_fraction"] - rows[0]["cascade_fraction"]) > 0.10
    # D1: disabling is orders of magnitude cheaper than recompiling
    disable_ratio = (RECOMPILE_HOURS * 3600) / DISABLE_SECONDS
    d1 = disable_ratio > 100

    print("\nD1 - compliance vs competence")
    print("-" * 72)
    print(f"  disable: {DISABLE_SECONDS:.0f}s per adapter    recompile:"
          f" {RECOMPILE_HOURS:.0f}h per adapter    ratio {disable_ratio:,.0f}x")
    print("  Correctness (entry stops influencing output) is bought by the first.")
    print("  Competence is restored by the second. Only the second is rate-limited.")

    print(f"\n  D1 disabling decouples compliance from recompile: {'ok' if d1 else 'NO'}"
          f"   (scale argument, not a measured ratio)")
    print(f"  D2 batching amortises:                            {'ok' if d2 else 'NO'}"
          f"   (fleet={N_ADAPTERS} < max batch {max(BATCH_SIZES)}: saturation is")
    print(f"                                                         partly arithmetic;"
          f" quote the general form)")
    print(f"  D3 admission control is a lever on breadth:       SUPERSEDED by E0.2d,"
          f" which inverts it\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_2c_deletion_policies.json"
    out.write_text(json.dumps(
        {"seed": SEED, "recompile_hours": RECOMPILE_HOURS, "rows": rows,
         "D1_disable_decouples": bool(d1), "D2_batching_amortises": bool(d2),
         "D3_admission_is_a_lever": bool(d3)}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
