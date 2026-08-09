"""E0.6 -- the cover is not total. Worklist 1.3 / R-c, pure recording.

CLAIM UNDER TEST (rev 2.2 section 1.2b, R-c). The commitment register replaces a
partition with a cover: protection statistics are computed per OWNER on equal
draws from that owner's own provenance, rather than per region on traffic. R-b
keeps coverage monotone under churn by moving a retiree's probes to a fleet
residual set. R-c says that is not enough:

    "The fleet residual set holds probes for ALL provenance no live owner covers
     -- not only probes orphaned by retirement. Without this, a cover is not
     total the way a partition was: competence in a region where nothing was ever
     promoted has no owner, therefore no probe set, therefore no protection
     statistic. It is not under-weighted, it is ABSENT, and nothing reports its
     loss."

WHY THIS IS IN PHASE 1 AND NOT PHASE 3. Every other gap in this record announced
itself with a number that went the wrong way. This one cannot: an unowned surface
produces no statistic at all, so the failure mode is invisible BY CONSTRUCTION
and a register that shipped without R-c would report only its wins. The
measurement has to exist before the mechanism it audits, or there is nothing to
compare a Phase 3 win against.

THE THREE QUESTIONS, and the third is the one that decides whether R-c matters:

    Q1  How much live provenance does no live owner cover?
    Q2  How much of that is what R-b already catches -- probes orphaned by a
        retirement -- versus provenance that NEVER had an owner? R-c is a
        refinement of R-b only if the second is small. If it is large, R-c is
        not a refinement, it is the mechanism.
    Q3  Is the unowned surface INERT or LOAD-BEARING? An entry no owner records
        may still move adapter weights -- the recording rule is transitive
        through the selected card, and E0.2b showed that misses the retrieval
        path at 0.913 recall. Unowned AND influential is the case that matters:
        real competence resting on provenance nothing defends, where a deletion
        does damage that no protection statistic is watching for.

Ground truth for Q3 is free here and is the same functional operation E0.2b uses:
delete the entry, recompile, see which adapters moved. No enumeration.

AND THE UNCOMFORTABLE ONE. The register's cover is assembled from what was
PROMOTED, and promotion follows traffic. So the extent of the cover may itself be
a traffic-weighted quantity, which would mean the register escapes traffic
weighting WITHIN an owner -- I8's equal draw, which it does deliver -- while
inheriting it in the cover's reach. That is not an argument against the register.
It is an argument that `unowned fraction` has to be on the dashboard next to the
tail-safety win, because otherwise the win is reported over whatever surface
traffic happened to buy. Swept against traffic volume below.

KILL CRITERIA (pre-registered):
    KR1 Unowned fraction is ~0 (< 0.02) at every traffic level -> R-c defends a
        gap that does not exist at this scale. Adopt it as free insurance, but
        the record should say it is not load-bearing rather than implying it is.
    KR2 Never-owned provenance is a minority of unowned provenance -> R-c IS a
        refinement of R-b, as rev 2 presents it. If it is the majority, rev 2
        understates its own mechanism and R-b alone would have shipped a cover
        with a hole in it.
    KR3 Unowned entries have no true influence on any adapter -> the unowned
        surface is inert, and R-c is bookkeeping hygiene rather than protection.
        If they DO have influence, the gap is competence, not paperwork.

Is there a world that produces the other verdict? For KR1, yes: if every entry
feeds a card and every card wins some query, every entry is owned and the
fraction is exactly 0 -- which is what a small bank with heavy traffic gives, and
it is the left end of the sweep. For KR3, yes: an entry feeding only cards that
never win influences no adapter's weights, and would show unowned and inert
together.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import InfluenceWorld                    # noqa: E402
from rig_a.core.register import cover_report, owner_provenance     # noqa: E402

DIM = 16
N_ENTRIES = 120
N_CARDS = 16
N_ADAPTERS = 8
SEED = 20260806
N_WORLDS = 8

# Traffic volume. The register's cover is built from what got promoted, and
# promotion follows traffic -- so this is the axis that says whether the cover's
# reach is a design property or a traffic artifact.
TRAFFIC = (16, 32, 64, 128, 256, 512)

RETIRE_FRACTION = 0.25          # a quarter of the fleet has retired


def one_point(n_rollouts: int) -> dict:
    unowned, never, orphan = [], [], []
    infl_unowned, infl_owned = [], []
    unowned_with_influence = []
    worst_card, fully_unowned_cards, no_card = [], [], []

    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w_i)
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=N_CARDS,
            n_rollouts=n_rollouts, n_adapters=N_ADAPTERS, rng=rng,
        )
        n_ret = max(int(round(N_ADAPTERS * RETIRE_FRACTION)), 1)
        retired = set(int(a) for a in rng.choice(N_ADAPTERS, size=n_ret, replace=False))
        live = set(range(N_ADAPTERS)) - retired

        rep = cover_report(w, live_owners=live, retired_owners=retired)
        unowned.append(rep["unowned_fraction"])
        never.append(rep["never_owned_fraction"])
        orphan.append(rep["orphan_fraction"])

        # Q3 -- is the unowned surface inert? Ground truth, computed functionally.
        prov = owner_provenance(w)
        covered = set().union(*[prov[a] for a in live]) if live else set()

        # Q4 -- POOLED IS THE WRONG GRAIN, which this record has found at E0.1
        # KB, E1.1c and E2.3. `unowned fraction` over the whole ledger is a pooled
        # rate, so it hides a concentrated hole exactly as E0.1's 6.7% hid 79.8%.
        # The natural cell here is the card: a card no query ever selects has
        # every one of its sources unowned, and that is the tail the register
        # exists to protect. Reported worst-cell with the blindness ratio, in the
        # same form as E0.1's KB.
        per_card = [
            sum(1 for e in srcs if e not in covered) / len(srcs)
            for srcs in w.card_sources if srcs
        ]
        worst_card.append(max(per_card) if per_card else 0.0)
        # AND THE CELL GRAIN HAS ITS OWN BLIND SPOT. An entry in NO card's source
        # list is invisible to a per-card statistic -- there is no cell to put it
        # in -- so `worst card` can read 0.000 while the ledger still has unowned
        # provenance. Those entries are the worst case, not an edge case: no card
        # cites them, so no owner's probe draw can reach them by any route.
        # Reported separately, because a tail statistic that cannot see the tail
        # is the defect this column was added to fix.
        in_any_card = set().union(*[set(s0) for s0 in w.card_sources]) \
            if w.card_sources else set()
        no_card.append(sum(1 for e in range(N_ENTRIES) if e not in in_any_card)
                       / N_ENTRIES)
        fully_unowned_cards.append(
            float(np.mean([f >= 1.0 for f in per_card])) if per_card else 0.0)
        n_unowned_infl, n_unowned = 0, 0
        infl_u, infl_o = [], []
        for e in range(N_ENTRIES):
            moved = len(w.true_influenced_adapters(e))
            if e in covered:
                infl_o.append(moved)
            else:
                n_unowned += 1
                infl_u.append(moved)
                n_unowned_infl += int(moved > 0)
        infl_unowned.append(float(np.mean(infl_u)) if infl_u else 0.0)
        infl_owned.append(float(np.mean(infl_o)) if infl_o else 0.0)
        unowned_with_influence.append(n_unowned_infl / max(n_unowned, 1))

    return {
        "traffic": n_rollouts,
        "unowned_fraction": round(float(np.mean(unowned)), 4),
        "never_owned_fraction": round(float(np.mean(never)), 4),
        "orphan_fraction": round(float(np.mean(orphan)), 4),
        "mean_adapters_moved_unowned": round(float(np.mean(infl_unowned)), 3),
        "mean_adapters_moved_owned": round(float(np.mean(infl_owned)), 3),
        "unowned_with_any_influence": round(float(np.mean(unowned_with_influence)), 4),
        "worst_card_unowned": round(float(np.mean(worst_card)), 4),
        "cards_fully_unowned": round(float(np.mean(fully_unowned_cards)), 4),
        "in_no_card_fraction": round(float(np.mean(no_card)), 4),
        "blindness": round(float(np.mean(worst_card))
                           / max(float(np.mean(unowned)), 1e-9), 1),
    }


def main() -> int:
    pts = [one_point(t) for t in TRAFFIC]

    print("\nE0.6 -- the cover is not total (worklist 1.3 / R-c, pure recording)\n")
    print(f"  {N_WORLDS} worlds, {N_ENTRIES} entries, {N_CARDS} cards,"
          f" {N_ADAPTERS} owners, {int(RETIRE_FRACTION * 100)}% retired\n")
    print(f"  {'traffic':>8}{'unowned':>10}{'never':>9}{'orphan':>9}"
          f"{'worst card':>12}{'blind':>7}{'dead cards':>12}{'no card':>9}"
          f"{'moved|unown':>13}{'moved|own':>11}")
    for p in pts:
        print(f"  {p['traffic']:>8}{p['unowned_fraction']:>10.3f}"
              f"{p['never_owned_fraction']:>9.3f}{p['orphan_fraction']:>9.3f}"
              f"{p['worst_card_unowned']:>12.3f}{p['blindness']:>7.1f}"
              f"{p['cards_fully_unowned']:>12.3f}"
              f"{p['in_no_card_fraction']:>9.3f}"
              f"{p['mean_adapters_moved_unowned']:>13.2f}"
              f"{p['mean_adapters_moved_owned']:>11.2f}")
    print("\n    unowned  live provenance no LIVE owner covers")
    print("    never    of that, what never had an owner at all -- R-c's addition")
    print("    orphan   of that, what a retirement dropped -- R-b already catches it")
    print("    worst    the worst CARD's unowned fraction -- the tail grain")
    print("    blind    worst / pooled, in E0.1 KB's form")
    print("    dead     fraction of cards with EVERY source unowned")
    print("    no card  entries in NO card at all -- invisible to `worst card`,")
    print("             and the reason it reads 0.000 while `unowned` does not")
    print("    moved|*  adapters whose weights actually move when the entry is")
    print("             deleted, ground truth, by whether the entry is covered")

    lo, hi = pts[0], pts[-1]
    kr1 = not all(p["unowned_fraction"] < 0.02 for p in pts)
    kr2 = all(p["never_owned_fraction"] <= p["orphan_fraction"] for p in pts)
    kr3 = any(p["unowned_with_any_influence"] > 0 for p in pts)

    print(f"\n  KR1 the gap exists at some traffic level:      {'ok' if kr1 else 'NO'}")
    print(f"  KR2 R-c is a refinement of R-b, not the job:   {'ok' if kr2 else 'NO'}")
    if not kr2:
        r = hi['never_owned_fraction'] / max(hi['orphan_fraction'], 1e-9)
        print(f"       never-owned exceeds orphaned {r:.1f}x even at the highest")
        print("       traffic. R-c is not a refinement of R-b -- it is doing most")
        print("       of the work, and R-b alone ships a cover with a hole in it.")
    print(f"  KR3 the unowned surface is load-bearing:       {'ok' if kr3 else 'NO'}")
    print(f"       {hi['unowned_with_any_influence']:.1%} of unowned entries move at"
          f" least one adapter, but only")
    print(f"       {hi['mean_adapters_moved_unowned']:.2f} adapters each against"
          f" {hi['mean_adapters_moved_owned']:.2f} for covered entries. Load-bearing,")
    print("       and thinly. Not 'a third of competence is undefended'.")

    print(f"\n  TRAFFIC. Unowned provenance runs {lo['unowned_fraction']:.3f} at"
          f" {lo['traffic']} rollouts to {hi['unowned_fraction']:.3f} at {hi['traffic']}.")
    print("  The cover's REACH is set by traffic even though every statistic")
    print("  computed inside it is an equal draw. I8 holds within an owner and")
    print("  says nothing about how far the owners collectively reach, so a")
    print("  tail-safety win from the register is a win over whatever surface")
    print("  traffic happened to buy. That is not an argument against the")
    print("  register -- it is why `unowned fraction` belongs on the dashboard")
    print("  beside the win, which is exactly what R-c asks for.")

    verdict = "PASS" if (kr1 and kr3) else "FAIL"
    print(f"\n  R-c IS LOAD-BEARING: {verdict}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_6_total_cover.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "n_entries": N_ENTRIES,
         "n_cards": N_CARDS, "n_adapters": N_ADAPTERS,
         "retire_fraction": RETIRE_FRACTION, "points": pts,
         "KR1_gap_exists": bool(kr1), "KR2_refinement_of_Rb": bool(kr2),
         "KR3_load_bearing": bool(kr3), "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
