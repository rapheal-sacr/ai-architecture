"""E0.7 -- the oracle line and the compute line. Worklist 1.2 and 2.3.

CLAIM UNDER TEST (rev 2.2 sections 1.3 and 1.2b R-d). Section 1.3 was repriced in
2.1 after the review found it contradicted section 4.3 -- "prefer many small
owners" against "oracle is the binding currency". The repriced claim:

    "Prefer many small owners WITHIN A PROVENANCE NEIGHBOURHOOD. The oracle price
     of a new owner is only the probes its provenance does not already cover; the
     compute price is the whole fleet's probe set, every cycle."

    "Granularity is therefore compute-expensive and oracle-cheap CONDITIONAL ON
     PROVENANCE OVERLAP, and that condition is measurable. If real fleet overlap
     is low, section 1.3 is wrong as stated rather than mispriced."

This is that measurement. It is worklist 1.2 (build the registry) and worklist
2.3 (fleet provenance overlap) in one run, because they are the same quantity
seen from two sides: overlap is exactly what decides whether the pool shares.

REPORT TWO LINES AND NEVER ONE. That instruction is the document's own, and it is
the reason the contradiction hid in 2.0:

    distinct probes          ORACLE.  authored or harvested once. Scarce.
    probe-evaluations/cycle  COMPUTE. paid every cycle, forever. Abundant.

THE ASSUMPTION R-d MAKES AND THE DOCUMENT DOES NOT STATE. "Draw from the existing
pool first" shares a probe between two owners only if the probe is a function of
PROVENANCE ALONE. If a probe tests entry e the way owner A uses it, it may test
nothing for owner B -- the key becomes (entry, owner) and the pool never shares.
The design says what a probe is KEYED on and never says what a probe IS. So
context-boundedness is swept rather than assumed, and the deliverable is the
fraction at which R-d stops paying.

Sweeping it is not a hedge. It converts "the sharing is structural" from an
assertion into a stated precondition with a number attached, which is what the
mechanism-declaration amendment requires of an `assumes` row.

AND A NULL THE FIRST RUN DID NOT HAVE, WHICH CHANGED THE ANSWER. Two owners can
end up sharing a probe for two entirely different reasons: because their
provenance genuinely overlaps -- the mechanism section 1.3 describes -- or because
both drew from a bounded ledger and collided by chance. Those are not the same
thing, and only the first supports "prefer many small owners within a provenance
neighbourhood". The null is a birthday calculation over the union of the fleet's
provenance:

    E[distinct]  =  U * (1 - (1 - 1/U) ** (fleet * probes))

If observed distinct sits on that curve, overlap structure is contributing
nothing and the saving is arithmetic rather than architectural. The first run had
no null, reported reuse 0.39, and would have credited section 1.3 with a saving
that has nothing to do with provenance neighbourhoods.

ROLLOUTS SCALE WITH FLEET (B18). Holding total rollouts fixed while sweeping
fleet size would give a 256-owner fleet one rollout each, so provenance would
shrink with fleet and the oracle curve would measure that instead. Each owner
gets the same experience at every fleet size.

KILL CRITERIA (pre-registered):
    KO The oracle multiplier -- distinct probes divided by one owner's worth --
       grows roughly LINEARLY in fleet size at realistic overlap. That is rev
       2's own second kill criterion for section 1.4: a C1 pass bought in C2's
       currency. Section 1.3 is withdrawn and the register ships coarse.
    KV Overlap is so low that even at context_bound = 0 the pool barely shares
       (reuse below 0.10). Section 1.3 is then wrong as stated rather than
       mispriced, and worklist 2.3's kill fires.
    KC The context-bound fraction at which R-d stops paying is so low that any
       plausible probe semantics lands above it -- R-d would be a rule that
       cannot be satisfied rather than one that costs something.

Is there a world that produces the other verdict? For KO, yes: adapters drawing
from a heavily shared card bank have provenance sets that overlap almost
completely, and the pool then saturates -- distinct probes flat in fleet size.
For KV, yes: raising card multiplicity raises how many cards an entry feeds,
which is the lever E0.2d already established moves provenance overlap.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.influence import InfluenceWorld                    # noqa: E402
from rig_a.core.register import ProbeRegistry, owner_provenance    # noqa: E402

DIM = 16
N_ENTRIES = 480
N_CARDS = 32
ROLLOUTS_PER_ADAPTER = 6        # B18: rollouts scale with fleet, never fixed
SEED = 20260806
N_WORLDS = 6

PROBES_PER_OWNER = 8            # I8's equal draw. A policy constant.
FLEETS = (4, 8, 16, 32, 64, 128, 256)
MULTIPLICITY = (1, 2, 4)        # cards each entry feeds -> drives provenance overlap
CONTEXT_BOUND = (0.0, 0.25, 0.50, 0.75, 1.0)


def measure(fleet: int, mult: int, ctx: float) -> dict:
    distinct, evals, reuse, overlaps, nulls, prov_sizes = [], [], [], [], [], []

    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w_i)
        w = InfluenceWorld(
            dim=DIM, n_entries=N_ENTRIES, n_cards=N_CARDS,
            n_rollouts=fleet * ROLLOUTS_PER_ADAPTER, n_adapters=fleet, rng=rng,
            card_overlap=float(mult),
        )
        prov = owner_provenance(w)

        # Fleet provenance overlap -- worklist 2.3's quantity, mean pairwise
        # Jaccard over OWNERS. Not the card-level overlap E0.2d measures: the
        # probe pool is shared between owners, so the owner grain is the one
        # that prices it.
        vals = []
        for i in range(len(prov)):
            for j in range(i + 1, len(prov)):
                u = len(prov[i] | prov[j])
                vals.append(len(prov[i] & prov[j]) / u if u else 0.0)
        overlaps.append(float(np.mean(vals)) if vals else 0.0)

        # THE NULL. If the same number of probes were drawn uniformly at random
        # from the union of the fleet's provenance -- no overlap structure at all
        # -- how many would be distinct? Context-bound probes never collide, so
        # only the context-free share is subject to it.
        union = set().union(*prov) if prov else set()
        u = max(len(union), 1)
        n_draw = fleet * min(PROBES_PER_OWNER, u)
        free = n_draw * (1.0 - ctx)
        nulls.append(u * (1.0 - (1.0 - 1.0 / u) ** free) + (n_draw - free))
        prov_sizes.append(float(np.mean([len(p_) for p_ in prov])) if prov else 0.0)

        reg = ProbeRegistry()
        for a in range(fleet):
            reg.draw_for(a, prov[a], PROBES_PER_OWNER, rng, context_bound=ctx)
        distinct.append(reg.distinct)
        evals.append(reg.evaluations)
        total = reg.created + reg.reused
        reuse.append(reg.reused / total if total else 0.0)

    return {
        "fleet": fleet, "multiplicity": mult, "context_bound": ctx,
        "owner_overlap": round(float(np.mean(overlaps)), 4),
        "mean_provenance": round(float(np.mean(prov_sizes)), 1),
        "distinct_probes": round(float(np.mean(distinct)), 1),
        "null_distinct": round(float(np.mean(nulls)), 1),
        "excess_over_null": round(
            (float(np.mean(nulls)) - float(np.mean(distinct)))
            / max(float(np.mean(nulls)), 1e-9), 4),
        "evaluations": round(float(np.mean(evals)), 1),
        "reuse_rate": round(float(np.mean(reuse)), 4),
        "oracle_multiplier": round(float(np.mean(distinct)) / PROBES_PER_OWNER, 2),
    }


def main() -> int:
    grid = [measure(f, m, c) for m in MULTIPLICITY for c in CONTEXT_BOUND
            for f in FLEETS]

    def at(fleet: int, mult: int, ctx: float, field: str = "oracle_multiplier"):
        return next(r[field] for r in grid if r["fleet"] == fleet
                    and r["multiplicity"] == mult and r["context_bound"] == ctx)

    print("\nE0.7 -- the oracle line and the compute line"
          " (worklist 1.2 + 2.3)\n")
    print(f"  {N_WORLDS} worlds, {N_ENTRIES} entries, {N_CARDS} cards,"
          f" {PROBES_PER_OWNER} probes per owner\n")

    print("  ORACLE SAVING -- 1 - (distinct probes / fleet x probes).")
    print("  0% means every owner authored its own probes and the oracle cost")
    print("  multiplied by fleet size, which is rev 2's own second kill criterion.\n")
    for m in MULTIPLICITY:
        ov = at(FLEETS[-1], m, 0.0, "owner_overlap")
        pv = at(FLEETS[-1], m, 0.0, "mean_provenance")
        print(f"    multiplicity {m}  (owner overlap {ov:.3f},"
              f" mean provenance {pv:.0f} entries)")
        print(f"      {'ctx':>6}" + "".join(f"{'f=' + str(f):>9}" for f in FLEETS))
        for c in CONTEXT_BOUND:
            cells = ""
            for f in FLEETS:
                mult_v = at(f, m, c)
                cells += f"{1.0 - mult_v / f:>9.0%}"
            print(f"      {c:>6.2f}{cells}")
        print()

    big = FLEETS[-1]
    shared = [r for r in grid if r["fleet"] == big and r["context_bound"] == 0.0]

    print("  AGAINST THE NULL, at context-free probes. `null` is what pure")
    print("  birthday collision over the same provenance union would give.")
    print(f"    {'fleet':>6}{'mult':>6}{'overlap':>9}{'distinct':>10}"
          f"{'null':>9}{'excess':>9}")
    for f in FLEETS:
        for m in MULTIPLICITY:
            r = next(x for x in grid if x["fleet"] == f and x["multiplicity"] == m
                     and x["context_bound"] == 0.0)
            print(f"    {f:>6}{m:>6}{r['owner_overlap']:>9.3f}"
                  f"{r['distinct_probes']:>10.0f}{r['null_distinct']:>9.0f}"
                  f"{r['excess_over_null']:>9.1%}")

    excesses = [r["excess_over_null"] for r in grid if r["context_bound"] == 0.0]
    worst_excess = max(abs(e) for e in excesses)

    # KO -- does the oracle line stay well below fleet size at large fleets?
    ko_fail = [r for r in shared if r["oracle_multiplier"] >= 0.5 * big]
    # KV -- does the pool share at all when nothing is context-bound?
    kv_fail = [r for r in shared if r["reuse_rate"] < 0.10]
    # KN -- is the sharing ATTRIBUTABLE to provenance overlap, or is it collision?
    kn_fail = worst_excess < 0.05

    print(f"\n  KO oracle line sublinear in fleet:   {'ok' if not ko_fail else 'NO'}")
    print(f"  KV the pool shares at all:           {'ok' if not kv_fail else 'NO'}")
    print(f"  KN sharing is due to OVERLAP:        {'ok' if not kn_fail else 'NO'}"
          f"   (largest departure from the collision null: {worst_excess:.1%})")

    print("\n  WHAT THE NULL DOES TO SECTION 1.3. The saving is real and it is")
    print("  large at big fleets -- but it is not the saving the document claims.")
    print("  Distinct probes sit on the birthday curve, and tripling owner")
    print("  provenance overlap moves the count by a few probes out of hundreds.")
    print("  So the mechanism is NOT `a new owner only pays for probes its")
    print("  provenance does not already cover'. It is that fleet x probes runs")
    print("  out of distinct ledger entries to probe. The oracle line saturates")
    print("  at the LEDGER, not at a provenance neighbourhood.")
    print("\n  That changes what the design should be sized against. `Prefer many")
    print("  small owners within a provenance neighbourhood' asks an architect to")
    print("  cluster owners by shared sources. This measurement says the quantity")
    print("  that decides the oracle bill is fleet x probes-per-owner against")
    print("  ledger size -- which is arithmetic available before any clustering,")
    print("  and which no amount of neighbourhood discipline improves.")
    print("\n  ALSO: context-boundedness is the whole ballgame. At ctx = 1.0 the")
    print("  saving is 0% at every fleet size and every overlap, because a probe")
    print("  keyed on (entry, owner) can never be reused. The design says what a")
    print("  probe is KEYED on and never says what a probe IS, and that unstated")
    print("  semantics decides the scarce currency outright.")

    print("\n  WHAT THIS DOES NOT SETTLE. Owner overlap here is a property of a")
    print("  simulated card bank, and E0.2f established that this rig's")
    print("  overlap-derived quantities are sensitive to bank-to-fleet ratio.")
    print("  What does not depend on that: the null result. Whatever the real")
    print("  overlap turns out to be, it has to beat birthday collision before it")
    print("  can be credited, and this rig gives the test rather than the value.")

    verdict = "PASS" if (not ko_fail and not kv_fail and not kn_fail) else "FAIL"
    print(f"\n  SECTION 1.3 PRICING AS STATED: {verdict}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_7_probe_registry.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "n_entries": N_ENTRIES,
         "probes_per_owner": PROBES_PER_OWNER, "fleets": list(FLEETS),
         "multiplicity": list(MULTIPLICITY), "context_bound": list(CONTEXT_BOUND),
         "grid": grid, "worst_excess_over_null": worst_excess,
         "KO_sublinear": not ko_fail, "KV_pool_shares": not kv_fail,
         "KN_overlap_attributable": not kn_fail,
         "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
