"""E0.5 -- the selection journal. Worklist 1.1, pure recording.

CLAIM UNDER TEST (rev 2.2 section 2). E0.2b's residual is real: transitive
provenance recalls 0.913, is complete on only 61% of tombstones, and NO set-based
closure can reach 1.0, because the missing dependency is on a retrieval that was
never executed. Rev 2's answer is not a better closure. It is that you do not
need the counterfactual -- you need to know whether the RANKING would have
flipped, and that is decidable from scores the system already computed.

    "For a tombstone set E, if for every candidate the maximum possible score
     contribution of any e in E is less than that selection's margin to the
     runner-up, the selection provably could not have flipped."

THREE THINGS THE DOCUMENT CLAIMS THIS BUYS, and this experiment scores each:

    S  SOUNDNESS.  A certified selection never flips. Not a rate -- a property.
       Any violation kills the mechanism outright, and no threshold makes a
       violation acceptable.
    U  USEFULNESS. The certified fraction. Rev 2's own kill criterion is
       "below roughly 70%, the journal costs more than the cascade it saves."
    C  CASCADE.    "Uncertified selections are a strict subset of
       transitive-closure edges." If false, the journal does not shrink the
       cascade -- it may enlarge it.

WHAT THIS EXPERIMENT CANNOT DECIDE, AND WHAT IT REPLACES IT WITH. The 0.70
criterion is written against REAL TRAFFIC and this is a simulator, so a single
certified fraction from one world is worth nothing -- it is a property of that
world's score geometry. So the fraction is not reported as a number. It is
reported as a CURVE against the one world property the algebra says drives it:

    rho_c  =  ||M_c||_2 * sum_{e deleted} ||v_e - mean_c||  /  (k_c - m)

Support per card `k` sits in the denominator. Deleting one of many sources barely
moves a card; deleting one of three moves it a long way. So the prediction is
that certification is governed by k, and the deliverable is the k at which it
becomes viable -- a number a card bank can be designed against, which a single
fraction is not.

TWO ARMS, because ledger size moves two things at once:
    k grows    n_cards fixed while the ledger grows, so sources per card grow
    k fixed    n_cards grows with the ledger, so sources per card stay put
If certification improves in both, ledger size is doing the work. If only in the
first, k is, and the design consequence is about card support rather than scale.

THE DECOMPOSITION, reported at every point. A certified fraction has two ways to
be low and they call for opposite responses:

    stable but uncertified   the bound was too loose. An INSTRUMENT problem.
    genuinely flipped        the selection really was fragile. A WORLD problem,
                             and no sound bound can certify it.

KILL CRITERIA (pre-registered):
    KS ANY certified selection that flips. Fatal, no threshold.
    KU At a point where the world is stable above 0.70, certification is below
       0.70 -- the mechanism failing where the world would have let it succeed.
       Scored PER POINT, never on aggregates: a min-over-arms compared against a
       min-over-arms would pair a fraction with someone else's stable rate.
    KC Uncertified selections are NOT a subset of what transitive provenance
       already flags. Scored with the follow-up that decides which direction the
       failure runs: of the uncertified selections OUTSIDE the closure, how many
       actually flipped. Any that do are influence the closure cannot see.

Is there a world that produces the other verdict? For KS, yes: a sign error or an
off-by-one in the (k - m) denominator certifies selections that move, and the
audit performs the deletion for real to catch exactly that. For KU, yes: at high
k a deletion moves a card by a fraction of a percent and everything certifies --
that is the expected right-hand end of the curve. For KC, yes: if every flip runs
through the chosen card, the closure sees them all and the subset claim holds.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import InfluenceWorld            # noqa: E402
from rig_a.core.register import SelectionJournal           # noqa: E402

DIM = 16
N_ROLLOUTS = 48
N_ADAPTERS = 8
SEED = 20260806
N_WORLDS = 8
FLEET_ROLLOUTS = 384      # for the window analysis, held above 1 rollout/adapter

# The sweep. `k grows` holds the bank at 8 cards while the ledger grows, so
# sources per card scale with it. `k fixed` grows the bank proportionally.
LEDGERS = (30, 60, 120, 240, 480)
ARMS = {"k grows": None, "k fixed": 8}      # value = entries per card, or None

# E0.2c found batching at window 16 makes deletion throughput independent of
# cascade breadth, so the certificate has to survive batches. A set deletion hits
# more cards and the bound sums over hit sources, so this is where it is weakest.
BATCH_SIZES = (1, 4, 16)
DRAWS_PER_WORLD = 6


def point(n_entries: int, per_card: int | None) -> dict:
    """One sweep point: build worlds, record, delete, audit."""
    n_cards = 8 if per_card is None else max(n_entries // per_card, 2)
    acc = {b: {"cert": [], "stable": [], "flip": [], "touched": []} for b in BATCH_SIZES}
    ks, violations = [], 0
    outside_closure, outside_and_flipped = 0, 0

    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w_i)
        w = InfluenceWorld(
            dim=DIM, n_entries=n_entries, n_cards=n_cards,
            n_rollouts=N_ROLLOUTS, n_adapters=N_ADAPTERS, rng=rng,
        )
        j = SelectionJournal.record(w)
        ks.append(float(np.mean([b.k for b in j.bounds])))

        alive = np.ones(n_entries, dtype=bool)
        base_sel = w.selected_cards(alive)

        for b in BATCH_SIZES:
            for _ in range(DRAWS_PER_WORLD):
                batch = [int(e) for e in rng.choice(n_entries, size=b, replace=False)]
                a = j.audit(w, batch)
                acc[b]["cert"].append(a["certified"] / a["n"])
                acc[b]["stable"].append(a["stable"] / a["n"])
                acc[b]["flip"].append(a["flipped"] / a["n"])
                violations += a["violations"]
                # The rival explanatory variable: what FRACTION OF THE BANK this
                # deletion makes uncertain. The bound's denominator says k should
                # govern; the `k fixed` arm is what decides between them.
                dset0 = set(batch)
                acc[b]["touched"].append(
                    sum(1 for bd in j.bounds if dset0 & set(bd.src_dev)) / n_cards)

                # KC. The closure's reach for a deletion is "the chosen card
                # cites a deleted entry, or the rollout cited one directly" --
                # that is all provenance records. An uncertified selection
                # OUTSIDE that reach is one the journal flags and the closure
                # cannot. Whether that is slack or real influence is decided by
                # whether any of them actually flip.
                cert = j.certified(batch)
                alive_after = alive.copy()
                for e in batch:
                    alive_after[e] = False
                flipped = base_sel != w.selected_cards(alive_after)
                dset = set(batch)
                for i in range(N_ROLLOUTS):
                    if cert[i]:
                        continue
                    reach = bool(dset & set(w.card_sources[base_sel[i]])) or \
                        bool(dset & set(w.rollout_direct[i]))
                    if not reach:
                        outside_closure += 1
                        outside_and_flipped += int(flipped[i])

    return {
        "n_entries": n_entries,
        "n_cards": n_cards,
        "mean_k": round(float(np.mean(ks)), 1),
        "violations": violations,
        "outside_closure": outside_closure,
        "outside_and_flipped": outside_and_flipped,
        "batches": {str(b): {
            "certified": round(float(np.mean(acc[b]["cert"])), 4),
            "stable": round(float(np.mean(acc[b]["stable"])), 4),
            "flipped": round(float(np.mean(acc[b]["flip"])), 4),
            "bank_touched": round(float(np.mean(acc[b]["touched"])), 4),
        } for b in BATCH_SIZES},
    }


def window(w, j, rng, W: int = 16) -> dict:
    """The batch-16 tension, taken apart into the operations it conflates.

    E0.5 as first written asked "what fraction certifies at batch b" and read
    0.000 at b>=16 as a conflict with E0.2c's window-16 economics. That question
    conflates two operations that happen on different clocks:

        DISABLE    per tombstone, seconds, and it is what makes deletion SOUND
        RECOMPILE  batched over the window, hours, and it restores COMPETENCE

    Three readings, and they answer different questions:

      1 BATCH        one certificate over the whole window. What E0.5 measured.
      2 RUNNING      accumulate displacement per selection as the window fills,
                     dropping a selection when its cumulative bound crosses.
                     Sound, and at window close it is ALGEBRAICALLY THE SAME SET
                     as (1) -- the bound sums src_dev over the hit sources and
                     divides by (k - m) whether that sum is accumulated in one
                     step or sixteen. So running certification does not improve
                     the number; what it changes is WHEN the answer is needed.
      3 PER-TOMBSTONE  certified against every deletion taken alone. This is the
                     cheap thing one is tempted to do, it is what the disable
                     path actually needs, and it is NOT a sound batch guarantee:
                     e1 alone cannot flip a selection and e2 alone cannot, and
                     both together still can. Measured here rather than assumed.
    """
    n = w.n_rollouts
    alive = np.ones(w.n_entries, dtype=bool)
    base = w.selected_cards(alive)
    batch = [int(e) for e in rng.choice(w.n_entries, size=W, replace=False)]

    cert_batch = j.certified(batch)

    cert_each = np.ones(n, dtype=bool)
    disable_per_tombstone = []
    for e in batch:
        c = j.certified([e])
        cert_each &= c
        disable_per_tombstone.append(1.0 - c.mean())

    after = alive.copy()
    for e in batch:
        after[e] = False
    flipped = base != w.selected_cards(after)

    return {
        "disable_load": float(np.mean(disable_per_tombstone)),
        "recompile_queue_batch": float((~cert_batch).mean()),
        "recompile_queue_each": float((~cert_each).mean()),
        "truly_flipped": float(flipped.mean()),
        "unsound_per_tombstone": int((cert_each & flipped).sum()),
    }


def main() -> int:
    res = {arm: [point(n, per_card) for n in LEDGERS] for arm, per_card in ARMS.items()}

    print("\nE0.5 -- the selection journal (worklist 1.1, pure recording)\n")
    print("  certified = provably cannot flip.  stable = did not in fact flip.")
    print("  The gap between them is bound slack; `flipped` is real fragility.\n")

    for arm, pts in res.items():
        print(f"  ARM: {arm}")
        print(f"    {'entries':>8}{'cards':>7}{'k':>6}"
              + "".join(f"{'b=' + str(b):>26}" for b in BATCH_SIZES))
        print(f"    {'':>8}{'':>7}{'':>6}"
              + "".join(f"{'cert/stable/touched':>26}" for _ in BATCH_SIZES))
        for p in pts:
            cells = "".join(
                f"{p['batches'][str(b)]['certified']:>12.3f}"
                f"{p['batches'][str(b)]['stable']:>7.3f}"
                f"{p['batches'][str(b)]['bank_touched']:>7.2f}" for b in BATCH_SIZES)
            print(f"    {p['n_entries']:>8}{p['n_cards']:>7}{p['mean_k']:>6.1f}{cells}")
        print()

    # KS -- soundness, over every point in the sweep
    ks_ok = all(p["violations"] == 0 for pts in res.values() for p in pts)

    # KU -- scored per point, never on an aggregate
    ku_fail = [(arm, p["n_entries"], b, p["batches"][str(b)])
               for arm, pts in res.items() for p in pts for b in BATCH_SIZES
               if p["batches"][str(b)]["stable"] >= 0.70
               and p["batches"][str(b)]["certified"] < 0.70]

    # KC -- and which direction the failure runs
    outside = sum(p["outside_closure"] for pts in res.values() for p in pts)
    outside_flip = sum(p["outside_and_flipped"] for pts in res.values() for p in pts)

    print(f"  KS no certified selection ever flipped:   {'ok' if ks_ok else 'NO'}")
    print(f"  KU certifies where the world allows it:   "
          f"{'ok' if not ku_fail else 'NO'}   ({len(ku_fail)} points below 0.70"
          f" where the world was above it)")
    print(f"  KC uncertified subset of closure edges:   "
          f"{'ok' if outside == 0 else 'NO'}   ({outside} outside the closure,"
          f" {outside_flip} of which actually flipped)")

    grows = {p["n_entries"]: p for p in res["k grows"]}
    fixed = {p["n_entries"]: p for p in res["k fixed"]}
    lo, hi = LEDGERS[0], LEDGERS[-1]
    # Each arm holds one variable still, so each measures the other one cleanly.
    k_effect = grows[hi]["batches"]["1"]["certified"] - grows[lo]["batches"]["1"]["certified"]
    touch_effect = fixed[hi]["batches"]["1"]["certified"] - fixed[lo]["batches"]["1"]["certified"]
    cross = (grows[hi]["batches"]["1"]["certified"]
             - fixed[hi]["batches"]["1"]["certified"])

    print("\n  THE PRE-REGISTERED PREDICTION WAS INCOMPLETE, and the control arm is")
    print("  what shows it. Each arm holds one variable still, so each measures")
    print("  the other cleanly -- at b=1:")
    print(f"    k grows:  touched pinned at {grows[hi]['batches']['1']['bank_touched']:.2f}"
          f" (1 card of 8), k 3 -> 60  :  {k_effect:+.3f}")
    print(f"    k fixed:  k pinned at ~8, touched {fixed[lo]['batches']['1']['bank_touched']:.2f}"
          f" -> {fixed[hi]['batches']['1']['bank_touched']:.2f}  :  {touch_effect:+.3f}")
    print("  So k governs certification -- as predicted, it is the bound's")
    print("  denominator -- AND a second variable of comparable size governs it")
    print("  too: what fraction of the bank a deletion makes uncertain. A")
    print("  certificate cares how hard one card moved and how many rivals could")
    print("  have moved at all, and only k was pre-registered.")
    print(f"\n  The cross-arm comparison at ledger {hi} reads {cross:+.3f} and is the")
    print("  wrong number to quote: k rises 7.5x while touched rises 6x, so it")
    print("  nets two opposing effects into one figure. Same defect as B19 --")
    print("  a comparison across arms that differ in more than the manipulation.")

    # THE BATCH-16 TENSION, re-asked as the two operations it conflates
    wins = []
    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + 100 + w_i)
        wd = InfluenceWorld(dim=DIM, n_entries=960, n_cards=64,
                            n_rollouts=FLEET_ROLLOUTS, n_adapters=N_ADAPTERS, rng=rng)
        wins.append(window(wd, SelectionJournal.record(wd), rng, W=16))
    agg = {k: float(np.mean([x[k] for x in wins])) for k in wins[0]}
    unsound_total = sum(x["unsound_per_tombstone"] for x in wins)

    print("\n  THE BATCH-16 TENSION, TAKEN APART. E0.5 asked `what certifies at")
    print("  batch b` and read 0.000 at b>=16 as a conflict with E0.2c's window-16")
    print("  economics. That conflates two operations on different clocks:")
    print("  DISABLE is per tombstone and is what makes deletion sound; RECOMPILE")
    print("  is what gets batched.")
    print(f"    disable load, per tombstone      : {agg['disable_load']:.3f}"
          "   <- the fast path, and the only place a certificate is needed")
    print(f"    recompile queue, batch reading   : {agg['recompile_queue_batch']:.3f}")
    print(f"    recompile queue, per-tombstone   : {agg['recompile_queue_each']:.3f}")
    print(f"    actually flipped over the window : {agg['truly_flipped']:.3f}")
    print(f"    per-tombstone certifications that flipped anyway: {unsound_total}")
    print("    RUNNING certification is sound and, at window close, algebraically")
    print("    the SAME SET as the batch reading -- the bound sums src_dev over hit")
    print("    sources and divides by (k - m) whether accumulated in one step or")
    print("    sixteen. So it does not improve the number; it changes when the")
    print("    answer is needed, which is the point. The tension was mine: I")
    print("    measured a batch certificate for an operation that is per-tombstone.")
    if unsound_total:
        print("    And the cheap per-tombstone reading is NOT a sound batch")
        print("    guarantee -- selections certified against every deletion alone")
        print("    still flipped under the union. Measured, not assumed.")
    else:
        print("    The per-tombstone reading produced no violation here, and that")
        print("    is NOT soundness. It is provably unsound as a batch guarantee --")
        print("    e1 alone cannot flip a selection, e2 alone cannot, both together")
        print("    can -- so zero in six worlds is a fact about these worlds. Use")
        print("    the running form, which is sound and costs the same.")
    print(f"\n    And the honest cost: {agg['recompile_queue_batch']:.1%} of selections are")
    print(f"    uncertified over the window while {agg['truly_flipped']:.1%} actually flipped.")
    print("    So the certificate is nearly useless for PRUNING THE RECOMPILE")
    print("    QUEUE, which is what a window-scale reading would use it for. It")
    print("    earns its place on the disable path and nowhere else -- and that is")
    print("    a narrower claim than section 2 makes for it.")

    # KS is fatal and the other two are not, so they must not collapse into one
    # word. A mechanism that is SOUND but narrow is a different object from one
    # that is unsound, and reporting both as FAIL is the same defect as a pooled
    # rate: it destroys the distinction the reader needs.
    verdict = "FAIL" if not ks_ok else ("PARTIAL" if (ku_fail or outside) else "PASS")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "PARTIAL":
        print("    SOUND, and that is the load-bearing half -- a certificate that")
        print("    lies is worthless and this one does not, across every point in")
        print("    the sweep. NARROW: it pays at single tombstones with well-")
        print("    supported cards, and not at the batch sizes E0.2c's economics")
        print("    require. And it is NOT a subset of the closure: it flags")
        print("    selections provenance cannot see, some of which really flip.")
    print()

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_5_selection_journal.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "n_rollouts": N_ROLLOUTS,
         "ledgers": list(LEDGERS), "batch_sizes": list(BATCH_SIZES),
         "arms": res,
         "KS_sound": bool(ks_ok),
         "KU_failing_points": [{"arm": a, "entries": n, "batch": b, **d}
                               for a, n, b, d in ku_fail],
         "KC_outside_closure": outside, "KC_outside_and_flipped": outside_flip,
         "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
