"""E0.2b -- Unlearning: what the cascade misses, and what it costs when it fires.

REBUILD of E0.2, whose transitive arm was withdrawn as a tautology: its
`true_influencers` and its `record_provenance` executed the same loop body, so
the "correct" answer was the mechanism's own definition and no world could have
produced another verdict.

Two changes make this a measurement rather than a restatement.

GROUND TRUTH IS FUNCTIONAL. An entry influences an adapter iff deleting it moves
that adapter's weights -- computed by running the world twice. That is the exact
operation a tombstone performs, and it is independent of every provenance
policy, including ones nobody has enumerated.

THERE IS A THIRD PATH. Rollouts are not pre-assigned to cards; the system
retrieves a card per query, and which card wins depends on card values, hence on
their source entries. Deleting an entry can hand a rollout to a DIFFERENT card,
changing its content discontinuously. Provenance records the card that was
selected; the card that would have been selected was never run and is recorded
nowhere. No set-based closure can capture this, which is what makes it a real
test of `transitive` rather than a mirror of it.

TWO QUESTIONS, and the second is the one the first E0.2 owed and never produced.

    Q1 RECALL. Of the adapters a deletion truly moves, what fraction does each
       provenance policy actually invalidate?

    Q2 THE CEILING. When the cascade fires correctly it invalidates a large
       share of adapters, and adapters are "hours-days" per the design's own
       substrate table. That implies a maximum sustainable tombstone rate above
       which unlearning-by-construction is unaffordable. Below that rate L7's
       strongest claim is real; above it, deletion silently queues.

KILL CRITERIA (pre-registered):
    C1 fails if `transitive` recall < 1.0 -- i.e. a correct-looking provenance
       policy still misses adapters that deletion genuinely moves.
    C2 fails if the sustainable tombstone rate falls below 1/day at any
       plausible recompile cost, since a deletion mechanism that cannot absorb
       one request per day is not an operational capability.

Is there a world that could produce the other verdict? For C1, yes: a world
where retrieval never flips (one card, or queries far from every boundary) gives
`transitive` recall exactly 1.0. The flip count is reported alongside, so a
null result is visible as a null result rather than as a pass. For C2, yes --
cheap recompiles or a small cascade put the ceiling in the thousands.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import POLICIES, InfluenceWorld  # noqa: E402

DIM = 16
N_ENTRIES = 60
N_CARDS = 8
N_ROLLOUTS = 48
N_ADAPTERS = 8
SEED = 20260806
N_WORLDS = 12

# Q2 parameters. "hours-days" from the L7 substrate table.
RECOMPILE_HOURS = (2, 8, 24)
PARALLEL_RECOMPILES = 4


def run_worlds() -> dict:
    recall = {p: [] for p in POLICIES}
    over_fire = {p: [] for p in POLICIES}
    true_sizes, flips, missed_examples = [], [], []

    for w_i in range(N_WORLDS):
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=N_CARDS,
            n_rollouts=N_ROLLOUTS, n_adapters=N_ADAPTERS,
            rng=np.random.default_rng(SEED + w_i),
        )
        for eid in range(N_ENTRIES):
            truth = w.true_influenced_adapters(eid)
            if not truth:
                continue
            true_sizes.append(len(truth))
            flips.append(w.selection_flips(eid))

            for p in POLICIES:
                fired = w.policy_invalidates(eid, p)
                recall[p].append(len(fired & truth) / len(truth))
                # adapters invalidated that deletion did not actually move
                over_fire[p].append(len(fired - truth))
                if p == "transitive" and not truth <= fired:
                    missed_examples.append(
                        {"world": w_i, "entry": eid,
                         "missed_adapters": sorted(truth - fired),
                         "selection_flips": w.selection_flips(eid)}
                    )

    return {
        "policies": {
            p: {
                "mean_recall": round(float(np.mean(recall[p])), 4),
                "perfect_recall_fraction": round(
                    float(np.mean([r >= 1.0 for r in recall[p]])), 4),
                "mean_over_fire": round(float(np.mean(over_fire[p])), 3),
            }
            for p in POLICIES
        },
        "mean_true_cascade": round(float(np.mean(true_sizes)), 2),
        "mean_true_cascade_fraction": round(float(np.mean(true_sizes)) / N_ADAPTERS, 3),
        "mean_selection_flips": round(float(np.mean(flips)), 2),
        "tombstones_causing_a_flip": round(float(np.mean([f > 0 for f in flips])), 3),
        "n_transitive_misses": len(missed_examples),
        "miss_examples": missed_examples[:5],
    }


def ceiling(mean_cascade: float) -> list[dict]:
    """Q2: the maximum sustainable tombstone rate."""
    out = []
    for hours in RECOMPILE_HOURS:
        adapter_hours_per_day = 24.0 * PARALLEL_RECOMPILES
        cost_per_tombstone = mean_cascade * hours
        out.append({
            "recompile_hours_per_adapter": hours,
            "adapter_hours_per_tombstone": round(cost_per_tombstone, 2),
            "sustainable_tombstones_per_day": round(
                adapter_hours_per_day / cost_per_tombstone, 2),
        })
    return out


def ceiling_surface(fleet_size: int = 64) -> list[dict]:
    """The ceiling as a function of cascade breadth, not at one measured point.

    The 69% cascade fraction above is a property of THIS world's card overlap
    and rollout-to-adapter assignment, and a real deployment with more adapters
    and more specialised cards could sit anywhere on this axis. Quoting a single
    tombstones-per-day figure would repeat the B4 error: reporting a
    parameter-dependent number as though it were structural.

    What IS structural is the shape. Cost per tombstone scales linearly with
    both cascade breadth and recompile time, so the ceiling falls as their
    product -- and the design's own substrate table already fixes one factor at
    "hours-days".
    """
    out = []
    for frac in (0.05, 0.10, 0.25, 0.50, 0.70):
        row = {"cascade_fraction": frac, "adapters_touched": round(frac * fleet_size, 1)}
        for hours in RECOMPILE_HOURS:
            cost = frac * fleet_size * hours
            row[f"per_day_at_{hours}h"] = round(24.0 * PARALLEL_RECOMPILES / cost, 2)
        out.append(row)
    return out


def main() -> int:
    res = run_worlds()
    ceil = ceiling(res["mean_true_cascade"])

    c1 = res["policies"]["transitive"]["mean_recall"] >= 1.0
    c2 = all(c["sustainable_tombstones_per_day"] >= 1.0 for c in ceil)

    print(f"\nE0.2b  Unlearning: cascade recall, and the tombstone-rate ceiling"
          f"   ({N_WORLDS} worlds, {N_ADAPTERS} adapters each)\n")

    print("Q1 - recall against functionally-computed ground truth")
    print("-" * 66)
    print(f"  {'policy':<14}{'mean recall':>13}{'always complete':>18}{'over-fires':>13}")
    for p in POLICIES:
        s = res["policies"][p]
        print(f"  {p:<14}{s['mean_recall']:>13.4f}"
              f"{s['perfect_recall_fraction']:>18.3f}{s['mean_over_fire']:>13.2f}")
    print(f"\n  true cascade: {res['mean_true_cascade']} adapters per tombstone"
          f" ({res['mean_true_cascade_fraction']:.0%} of the fleet)")
    print(f"  retrieval flips: {res['mean_selection_flips']} rollouts reassigned"
          f" per tombstone; {res['tombstones_causing_a_flip']:.0%} of tombstones"
          f" cause at least one")
    print(f"  transitive misses: {res['n_transitive_misses']}")
    if res["miss_examples"]:
        ex = res["miss_examples"][0]
        print(f"    e.g. world {ex['world']} entry {ex['entry']}:"
              f" adapters {ex['missed_adapters']} moved but were not invalidated"
              f" ({ex['selection_flips']} rollouts changed card)")

    print("\nQ2 - what the correct cascade costs")
    print("-" * 66)
    print(f"  {'recompile hrs':>15}{'adapter-hrs/tombstone':>24}"
          f"{'sustainable /day':>19}")
    for c in ceil:
        print(f"  {c['recompile_hours_per_adapter']:>15}"
              f"{c['adapter_hours_per_tombstone']:>24.1f}"
              f"{c['sustainable_tombstones_per_day']:>19.2f}")
    print(f"\n  at {PARALLEL_RECOMPILES} parallel recompiles, 24h/day")

    surf = ceiling_surface()
    print("\n  Ceiling as a function of cascade breadth (fleet of 64 adapters),")
    print("  since 69% is a property of this world and not of the design:\n")
    print(f"    {'cascade':>9}{'touched':>9}"
          + "".join(f"{str(h)+'h /day':>12}" for h in RECOMPILE_HOURS))
    for row in surf:
        print(f"    {row['cascade_fraction']:>9.0%}{row['adapters_touched']:>9.1f}"
              + "".join(f"{row[f'per_day_at_{h}h']:>12.2f}" for h in RECOMPILE_HOURS))
    print("\n    Sustainable tombstones per day. Below 1.0, deletion requests"
          "\n    queue faster than they clear and unlearning-by-construction"
          "\n    stops being an operational capability.")

    print(f"\n  C1 transitive recall == 1.0:            {'ok' if c1 else 'NO'}")
    print(f"  C2 sustainable rate >= 1/day always:    {'ok' if c2 else 'NO'}")
    print(f"\n  VERDICT: {'PASS' if (c1 and c2) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_2b_influence_and_ceiling.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "n_adapters": N_ADAPTERS,
         "recall": res, "ceiling": ceil, "ceiling_surface": ceiling_surface(),
         "C1_transitive_complete": bool(c1), "C2_rate_operational": bool(c2),
         "verdict": "PASS" if (c1 and c2) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
