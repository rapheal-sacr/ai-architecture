"""E0.2e -- R9's curve: does provenance-aware admission have an acceptable point?

The last architectural gap in the record. E0.2d established the design has NO
lever on cascade breadth: SESA's cos <= 0.93 scores card CONTENT, while breadth
is set by PROVENANCE overlap, and moving content similarity with provenance held
fixed moves breadth by 0.009. E5.1 then made breadth the quantity that decides
whether an operating envelope exists at all -- the window at the design's own
profile is non-empty only below beta ~ 0.31, and the lowest breadth E0.2d
measured is 0.65.

So R9 -- provenance-aware admission, capping source-entry overlap between
admitted cards -- is the only proposed repair, and it does not exist yet.

STATE THE TENSION BEFORE BUILDING THE MECHANISM. Rejecting a card because it
shares provenance with an existing one means that card never exists, so the
distinction it would have drawn is never available. That is a COVERAGE cost, and
I7 is now the invariant that names exactly what it costs. R9 and I7 pull against
each other.

Every mechanism in this design that was specified without stating its tension has
been found out by an experiment -- the subspace budget, the blast-radius rule,
the three-lambda release rule, the staged ladder. So R9 is specified here as a
CURVE with an acceptable region marked, not as a threshold.

Both axes use machinery that already exists: breadth as E0.2d defines it,
coverage as E0.1/I7 define it.

KILL CRITERIA (pre-registered, then re-registered -- see below):
    G1 [ORIGINAL] fails if no admission threshold achieves breadth <= 0.31 while
       holding coverage at or above the I7 floor.
    G1 [RE-REGISTERED] fails if the curve CANNOT BE DRAWN -- if the net direction
       of breadth against tau depends on a coupling the design does not specify.
       Registering the original as if the curve were drawable assumed the answer.
    G2 fails if the DIRECT mechanism is insensitive to tau: if cards-per-entry
       does not fall as the cap tightens, R9's premise is wrong at the root.
    G3 reports both effects separately and names what must be specified.

WHY G1 HAD TO BE RE-REGISTERED. Tightening tau does two things at once and they
push opposite ways:

    cards per entry FALLS  6.87 -> 1.00   the direct R9 mechanism, and it works
    fleet reach     RISES  0.089 -> 0.375  because a tighter cap SHRINKS the bank,
                                           and each adapter still draws the same
                                           number of rollouts, so a surviving
                                           card is used by more adapters

Which dominates depends entirely on how the adapter fleet composes from the card
bank as the bank shrinks -- whether adapters keep drawing a fixed NUMBER of
rollouts or a fixed FRACTION of the bank, whether fleet size tracks bank size at
all. Parts I-III specify none of that.

So the honest result is neither "R9 works" nor "R9 has no acceptable point". It
is that R9 cannot be evaluated as specified, which is the same defect every other
mechanism in this record was found to have: proposed without stating the tension
it trades against. Choosing a coupling here to obtain a number would be choosing
the answer.

Is there a world that produces the other verdict? For G1, yes and it is the
optimistic one: if candidate cards are naturally provenance-disjoint, a tight cap
rejects almost nothing and coverage is untouched. The candidate pool's own
overlap structure is therefore swept, because whether it is generous or hostile
is exactly what decides this.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_REGIONS = 16
ENTRIES_PER_REGION = 40
N_ENTRIES = N_REGIONS * ENTRIES_PER_REGION
N_CANDIDATE_CARDS = 400
SOURCES_PER_CARD = 8
N_ADAPTERS = 64                        # the fleet E0.2d/E5.1 measure breadth over
ROLLOUTS_PER_ADAPTER = 6
SEED = 20260806
N_SEEDS = 6

TAUS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
POOL_CONCENTRATION = (0.2, 0.5, 0.9)   # how much candidate cards draw from a
                                       # shared hot pool -- the pool's own
                                       # overlap structure, swept
BREADTH_TARGET = 0.31                  # E5.1's window condition
COVERAGE_FLOOR = 0.80                  # legacy min-1 form, retained for audit
DENSITY_FLOOR = 3.0                    # I7 as a RATE: cards per region


def make_candidates(concentration: float, rng):
    """Candidate cards: a source-entry set and the region whose distinction it draws."""
    region = np.repeat(np.arange(N_REGIONS), ENTRIES_PER_REGION)
    rates = 1.0 / np.arange(1, N_REGIONS + 1, dtype=float)
    rates /= rates.sum()

    cards = []
    for _ in range(N_CANDIDATE_CARDS):
        r = int(rng.choice(N_REGIONS, p=rates))
        pool = np.where(region == r)[0]
        # `concentration` is how much of the card's support comes from that
        # region's HOT entries rather than anywhere in the region. High
        # concentration means candidate cards naturally overlap in provenance,
        # which is the hostile case for R9.
        # Zipfian draw within the region: `concentration` is the skew, so high
        # concentration means cards keep reaching for the same few entries and
        # overlap naturally. A previous version used a hard cutoff on a "hot"
        # slice, which at high concentration made every card's source set
        # IDENTICAL -- the bank froze at one card per region and tau stopped
        # moving anything, which the manipulation check caught.
        w = (np.arange(1, len(pool) + 1, dtype=float)) ** (-3.0 * concentration)
        w /= w.sum()
        src = set(rng.choice(pool, size=SOURCES_PER_CARD, replace=False,
                             p=w).tolist())
        cards.append({"region": r, "src": src})
    return cards, region


def admit(cards, tau: float):
    """Greedy provenance-aware admission: admit iff max Jaccard overlap <= tau."""
    bank = []
    for c in cards:
        ok = True
        for b in bank:
            u = len(c["src"] | b["src"])
            if u and len(c["src"] & b["src"]) / u > tau:
                ok = False
                break
        if ok:
            bank.append(c)
    return bank


def breadth_of(bank, rng) -> float:
    """E0.2d's quantity: mean fraction of the ADAPTER FLEET a single entry reaches.

    Card-level overlap is not the same measurement and gave 0.016-0.062 where
    E0.2d reports 0.63-0.99. The chain is entry -> cards -> rollouts -> adapters,
    and an entry feeding 2% of a 400-card bank can still reach most adapters,
    because each adapter trains on rollouts drawn from many cards. Breadth has to
    be propagated to the fleet or it is a different quantity wearing the name.
    """
    if not bank:
        return 0.0
    # each adapter trains on rollouts conditioned on a sample of the bank
    adapter_cards = [set(rng.choice(len(bank),
                                    size=min(ROLLOUTS_PER_ADAPTER, len(bank)),
                                    replace=False).tolist())
                     for _ in range(N_ADAPTERS)]
    card_of_entry = {}
    for i, b in enumerate(bank):
        for e in b["src"]:
            card_of_entry.setdefault(e, set()).add(i)
    if not card_of_entry:
        return 0.0
    reach = []
    for cards in card_of_entry.values():
        hit = sum(1 for ac in adapter_cards if ac & cards)
        reach.append(hit / N_ADAPTERS)
    return float(np.mean(reach))


def coverage_of(bank) -> dict:
    """I7's quantity -- as a RATE, not a min-1 test.

    A previous version returned `len({b["region"] for b in bank}) / N_REGIONS`:
    at least one card per region. That reported 1.000 at every tau and every pool
    structure, and it could not have reported anything else -- the bank only
    drops below one-card-per-region once it falls under ~16 cards total, while
    the quantity actually moving is a 9x change in per-region DENSITY.

    I7 as written is a rate: every region covered at a stated minimum per-region
    sampling rate, with regions below the floor recorded as uncovered. A min-1
    test discards the rate entirely. And the quantity that got binarised was
    per-region density on rare regions, which is what this entire record has been
    about -- so the tension may have been invisible rather than absent.

    Reported per the weighting rule: worst region, not the mean.
    """
    counts = np.zeros(N_REGIONS)
    for b in bank:
        counts[b["region"]] += 1
    return {"any": float((counts > 0).mean()),
            "mean_density": float(counts.mean()),
            "worst_density": float(counts.min()),
            "below_floor": float((counts < DENSITY_FLOOR).mean())}


def run(concentration: float, tau: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cards, _ = make_candidates(concentration, rng)
    bank = admit(cards, tau)
    cov = coverage_of(bank)
    return {"bank_size": len(bank), "breadth": breadth_of(bank, rng),
            "coverage": cov["any"], "mean_density": cov["mean_density"],
            "worst_density": cov["worst_density"], "below_floor": cov["below_floor"]}


def agg(concentration, tau):
    rs = [run(concentration, tau, SEED + i) for i in range(N_SEEDS)]

    def m(k):
        return float(np.mean([r[k] for r in rs]))

    return {"concentration": concentration, "tau": tau,
            "bank_size": round(m("bank_size"), 1),
            "breadth": round(m("breadth"), 3),
            "coverage": round(m("coverage"), 3),
            "acceptable": bool(m("breadth") <= BREADTH_TARGET
                               and m("coverage") >= COVERAGE_FLOOR)}


def cards_per_entry(bank) -> float:
    """The DIRECT R9 mechanism: how many cards a single entry feeds."""
    cpe = {}
    for i, b in enumerate(bank):
        for e in b["src"]:
            cpe.setdefault(e, set()).add(i)
    return float(np.mean([len(v) for v in cpe.values()])) if cpe else 0.0


def main() -> int:
    print("\nE0.2e  R9's curve -- and why it cannot be drawn as specified\n")

    rows = []
    for conc in POOL_CONCENTRATION:
        label = ("hostile" if conc >= 0.8 else "generous" if conc <= 0.3 else "middling")
        print(f"  candidate-pool provenance concentration {conc:.1f}  ({label})")
        hdr = (f"    {'tau':>6}{'bank':>8}{'cards/entry':>13}{'fleet reach':>13}"
               f"{'cov(any)':>10}{'density':>9}{'worst':>8}{'<floor':>8}")
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        for tau in TAUS:
            rs = [run(conc, tau, SEED + i) for i in range(N_SEEDS)]
            cpes = []
            for i in range(N_SEEDS):
                rng = np.random.default_rng(SEED + i)
                cards, _ = make_candidates(conc, rng)
                cpes.append(cards_per_entry(admit(cards, tau)))
            r = {"concentration": conc, "tau": tau,
                 "bank_size": round(float(np.mean([x["bank_size"] for x in rs])), 1),
                 "cards_per_entry": round(float(np.mean(cpes)), 2),
                 "fleet_reach": round(float(np.mean([x["breadth"] for x in rs])), 3),
                 "coverage": round(float(np.mean([x["coverage"] for x in rs])), 3),
                 "mean_density": round(float(np.mean([x["mean_density"] for x in rs])), 2),
                 "worst_density": round(float(np.mean([x["worst_density"] for x in rs])), 2),
                 "below_floor": round(float(np.mean([x["below_floor"] for x in rs])), 3)}
            rows.append(r)
            print(f"    {tau:>6.2f}{r['bank_size']:>8.1f}{r['cards_per_entry']:>13.2f}"
                  f"{r['fleet_reach']:>13.3f}{r['coverage']:>10.3f}"
                  f"{r['mean_density']:>9.2f}{r['worst_density']:>8.2f}"
                  f"{r['below_floor']:>8.3f}")
        print()

    # G2 -- does the direct mechanism respond at all?
    spans_cpe, spans_reach, opposed = [], [], []
    for conc in POOL_CONCENTRATION:
        sub = sorted([r for r in rows if r["concentration"] == conc],
                     key=lambda x: x["tau"])
        cpe_lo, cpe_hi = sub[0]["cards_per_entry"], sub[-1]["cards_per_entry"]
        rch_lo, rch_hi = sub[0]["fleet_reach"], sub[-1]["fleet_reach"]
        spans_cpe.append(cpe_hi - cpe_lo)
        spans_reach.append(rch_hi - rch_lo)
        opposed.append((cpe_hi - cpe_lo) * (rch_hi - rch_lo) < 0)

    g2 = min(spans_cpe) > 0.5
    g1 = not any(opposed)          # curve is drawable only if they do NOT oppose

    print("  G3  the two effects, tightest tau to loosest")
    print(f"      {'conc':>6}{'cards/entry':>14}{'fleet reach':>14}   directions")
    for conc, sc, sr, op in zip(POOL_CONCENTRATION, spans_cpe, spans_reach, opposed):
        print(f"      {conc:>6.1f}{sc:>+14.2f}{sr:>+14.3f}"
              f"{'   OPPOSED' if op else '   aligned'}")

    print(f"\n  G1 [re-registered] the curve is drawable:      {'ok' if g1 else 'NO'}")
    print(f"  G2 the direct mechanism responds to tau:       {'ok' if g2 else 'NO'}")

    if not g1:
        print("\n  => R9 CANNOT BE EVALUATED AS SPECIFIED. Tightening the cap reduces")
        print("     cards-per-entry -- the mechanism works -- and simultaneously")
        print("     shrinks the card bank, which raises how much of the adapter")
        print("     fleet each surviving card reaches. Which dominates depends on")
        print("     how the fleet composes from the bank as the bank shrinks, and")
        print("     Parts I-III specify none of it: not whether adapters draw a")
        print("     fixed NUMBER of rollouts or a fixed FRACTION of the bank, nor")
        print("     whether fleet size tracks bank size at all.")
        print("\n     WHAT MUST BE SPECIFIED BEFORE R9 CAN BE DESIGNED:")
        print("       1. how many distinct cards an adapter's training draw uses")
        print("       2. whether that is absolute or a fraction of the bank")
        print("       3. whether the fleet grows with the bank")
        print("\n     AND THE COVERAGE SIDE IS ALSO UNEVALUABLE. Reported as a rate")
        print("     rather than as at-least-one-card-per-region, per-region density")
        print("     falls sharply with tau and the worst region falls fastest --")
        print("     which is the I7 tension R9 was expected to trade against, and")
        print("     which the earlier min-1 metric could not see. So R9 is DOUBLY")
        print("     unevaluable: the breadth side needs three specification")
        print("     decisions, and the coverage side needs I7 stated as the rate it")
        print("     already is rather than as a binary.")
    print()

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_2e_r9_breadth_coverage.json"
    out.write_text(json.dumps({"seed": SEED, "breadth_target": BREADTH_TARGET,
                               "coverage_floor": COVERAGE_FLOOR, "rows": rows,
                               "G1_curve_drawable": bool(g1),
                               "G2_direct_mechanism_responds": bool(g2),
                               "directions_opposed": opposed}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
