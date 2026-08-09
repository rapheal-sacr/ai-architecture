"""Asset B's specification, as predicates over the asset rather than hopes about the run.

Worklist v2 §4: "Pre-specify before acquiring, in each case, the manipulation
check that would fire if the asset doesn't support the measurement -- EB.3's
format control is the model. Write it as a predicate over the asset, not as a
hope about the run."

EB.3 is why. It ran, the manipulation check fired twice, and the deliverable
turned out to be a specification for what the run needed. That specification cost
a run. This one costs nothing, and it is executable BEFORE the acquisition.

WHAT ASSET B UNBLOCKS -- three measurements, not four. EB.6 took `r` off the
blocked list by measuring it with a generator and predicates, neither corpus-shaped.

    M1  checkability-difficulty correlation   -- locates E2.1 on its own sweep
    M2  certified fraction                    -- section 2's re-registered kill
    M3  entry-degree distribution             -- section 3's fan-out kill

M2's FULL FORM NEEDS MORE THAN A LOG. Section 2's kill was re-registered against
"fraction of the fleet live during a deletion window", and a log contains no
fleet. The certified fraction is computable from the log; AVAILABILITY is the log
composed with a compiled bank and a fleet over it -- Rig A machinery that already
exists from E0.6/E1.7. Stated here because the spec otherwise reads as though M2
rides on the log alone, and it does not.

SCHEMA. One JSON object per interaction, JSONL:

    {
      "query_id":     str,
      "domain":       str,        # REQUIRED. See P1d -- capture-time, never inferred
      "domain_source":str,        # "capture" | "inferred"
      "source":       str,        # which corpus/deployment this came from. See P4
      "candidates":   [{"card_id": str, "score": float}],
      "chosen":       str,
      "scorer":       str,        # "lexical"|"embedding_sum"|"cross_encoder"|...
      "k_logged":     int,
      "next_score":   float|null, # the (k+1)-th score, or null if k is the full set
      "outcome":      "pass"|"fail"|null,
      "resolved_by":  str,        # "executable"|"replay"|"sealed_suite"|"judge"|"human"
      "difficulty":   float|null,
      "difficulty_source": str,   # never "outcome". See P1b
      "cards":        {card_id: [entry_id, ...]},
      "bank_admission":     str,  # "none" if unfiltered. See P3a
      "bank_size":    int, "fleet_size": int, "rollouts_per_adapter": float   # P3c
    }

Run:  python tools/validate_asset_b.py <log.jsonl>
      python tools/validate_asset_b.py --self-test

THE SELF-TEST COVERS EVERY PREDICATE, and `check_record.py` asserts that it does.
A validator whose checks cannot fire is the defect this record has found four
times (B7, B8, B20, A4-usage). Two instances of it were sitting in this file:
`P3c` could not fail on any asset, and the self-test covered 6 of 11 while the
docstring claimed all -- and the unfireable predicate sat inside the untested
set, so the two defects concealed each other.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter

INDEPENDENT = {"executable", "replay", "human"}   # not the model under test
DECOMPOSABLE = {"lexical", "embedding_sum", "embedding", "bm25"}

# ---------------------------------------------------------------------------
# Thresholds, DERIVED from the precision the measurement needs rather than
# chosen. Both were hardcoded, which is the shape ρ had before R1 and rank had
# before R12 -- a constant with no argument attached is a decision nobody made.
# ---------------------------------------------------------------------------

# P0. M1 estimates a correlation on the independently-resolved subset. SE of a
# correlation is ~1/sqrt(n-3); E2.1's sweep is read at steps of 0.25, so
# resolving one step at 2 SE needs 2/sqrt(n-3) <= 0.125.
CORR_STEP, CORR_SIGMAS = 0.25, 2.0
MIN_INDEPENDENT = int(math.ceil((CORR_SIGMAS / (CORR_STEP / 2)) ** 2)) + 3   # 259

# P3b. Section 3's kill separates a top-1% tail an ORDER OF MAGNITUDE above
# F_max from one merely 3x above. The top 1% must hold enough entries that its
# mean degree has relative precision better than the gap between those, i.e.
# 1/sqrt(m) <= 0.25 -> m >= 16, and n >= 100m.
TAIL_PRECISION = 0.25
MIN_TAIL = int(math.ceil(1 / TAIL_PRECISION ** 2))          # 16
MIN_ENTRIES = 100 * MIN_TAIL                                # 1600


def _spread(xs) -> bool:
    xs = [x for x in xs if x is not None]
    return len(xs) >= 2 and min(xs) != max(xs)


def predicates(rows: list) -> list:
    """Each returns (id, ok, detail, consequence). Computable from the asset alone."""
    out = []
    n = max(len(rows), 1)
    r0 = rows[0] if rows else {}

    # ---- P0 / P0a, cross-cutting: laundering and resolver independence ----
    res = Counter(r.get("resolved_by") for r in rows)
    indep = sum(v for k, v in res.items() if k in INDEPENDENT)
    out.append((
        "P0 enough independent resolutions", indep >= MIN_INDEPENDENT,
        f"{indep} resolved by {sorted(INDEPENDENT)} (need >= {MIN_INDEPENDENT}); "
        f"mix = {dict(res)}",
        f"E2.1's laundering, inherited by every measurement at once. The floor is "
        f"DERIVED: M1 reads E2.1's sweep at steps of {CORR_STEP}, and resolving one "
        f"step at {CORR_SIGMAS} SE of a correlation needs n >= {MIN_INDEPENDENT}."))
    out.append((
        "P0a resolver disjoint from scorer",
        all(r.get("resolved_by") != r.get("scorer") for r in rows)
        and all(r.get("resolved_by") for r in rows),
        f"every row records `resolved_by`; "
        f"{sum(1 for r in rows if r.get('resolved_by') == r.get('scorer'))} share it with `scorer`",
        "OUTCOME PROVENANCE. P0 alone cannot catch a silent failure -- a log where "
        "the thing that scored the candidate also resolved its outcome looks "
        "complete and is circular. Recording HOW each outcome was resolved and "
        "checking the resolver is disjoint from the scorer is the thesis applied "
        "to the asset."))

    # ---- M1 -----------------------------------------------------------------
    diff = [r.get("difficulty") for r in rows]
    have = sum(d is not None for d in diff)
    out.append((
        "P1a difficulty recorded", have / n >= 0.80,
        f"{have}/{len(rows)} carry a difficulty value",
        "M1 has no x-axis."))
    out.append((
        "P1b difficulty independent of outcome",
        all(r.get("difficulty_source") not in (None, "outcome") for r in rows),
        f"{sum(1 for r in rows if r.get('difficulty_source') in (None, 'outcome'))}"
        " rows derive difficulty from the outcome or do not say",
        "THE E0.2 TRAP FOR THIS ASSET, and it CANNOT BE RETROFITTED. If difficulty "
        "is read off success, checkability-vs-difficulty is a correlation between "
        "a variable and itself and no world returns anything else."))
    out.append((
        "P1c difficulty has spread", _spread(diff),
        f"difficulty range {'varies' if _spread(diff) else 'CONSTANT'}",
        "A constant axis gives an unestimable correlation."))
    out.append((
        "P1d domain recorded at capture",
        all(r.get("domain") for r in rows)
        and all(r.get("domain_source") == "capture" for r in rows),
        f"{sum(1 for r in rows if not r.get('domain'))} rows lack a domain; "
        f"{sum(1 for r in rows if r.get('domain_source') != 'capture')} not capture-time",
        "E2.1's quantity is PER-DOMAIN difficulty bias -- mean 0.095, worst 0.176 "
        "at correlation 1.0 -- so a log without a domain label yields only a "
        "pooled correlation and M1 cannot produce its own statistic. "
        "AND IT CANNOT BE RETROFITTED: a domain label applied after capture is an "
        "INFERRED partition, which is E0.1-K4 and E3.3's territory exactly."))

    # ---- M2 -----------------------------------------------------------------
    sc = Counter(r.get("scorer") for r in rows)
    dec = sum(v for k, v in sc.items() if k in DECOMPOSABLE)
    out.append((
        "P2a decomposable scorer", dec / n >= 0.80,
        f"{dec}/{len(rows)} scored by {sorted(DECOMPOSABLE)}; mix = {dict(sc)}",
        "Section 2's stated tension as an admission check: a cross-encoder means "
        "NO certificate exists and M2 cannot be run at all."))
    tail_ok = sum(1 for r in rows
                  if r.get("next_score") is not None
                  or r.get("k_logged", 0) >= len(r.get("candidates", [])) > 0)
    out.append((
        "P2b rival tail bounded", tail_ok / n >= 0.90,
        f"{tail_ok}/{len(rows)} log the full candidate set or the (k+1)-th score",
        "E0.5: 75.9% of uncertified selections are RIVAL RISE, and a rival "
        "outside top-k cannot be bounded at all. Without the (k+1)-th score M2 "
        "measures the best case and omits its own dominant failure mode."))
    prov = sum(1 for r in rows
               if r.get("cards") and all(c["card_id"] in r["cards"]
                                         for c in r.get("candidates", [])))
    out.append((
        "P2c candidate provenance present", prov / n >= 0.90,
        f"{prov}/{len(rows)} carry provenance for EVERY logged candidate",
        "The bound needs each source entry's deviation from its card's mean. "
        "Provenance for the chosen card alone is not enough -- the rival moves."))
    margins = []
    for r in rows:
        s = sorted((c["score"] for c in r.get("candidates", [])), reverse=True)
        if len(s) >= 2:
            margins.append(s[0] - s[1])
    out.append((
        "P2d margins have spread", _spread(margins),
        f"{len(margins)} margins, {'varying' if _spread(margins) else 'CONSTANT'}",
        "Uniform margins make certified fraction 1.0 or 0.0 by construction."))

    # ---- M3 -----------------------------------------------------------------
    out.append((
        "P3a bank is unconstrained",
        all(r.get("bank_admission") in (None, "none", "unconstrained") for r in rows),
        f"{sum(1 for r in rows if r.get('bank_admission') not in (None,'none','unconstrained'))}"
        " rows report an admission filter",
        "CANNOT BE RETROFITTED. A pre-filtered bank's degree distribution is the "
        "FILTER'S, not the structure's, and M3 answers a question about someone "
        "else's threshold. E0.2e is the precedent."))
    deg = Counter()
    for r in rows:
        for srcs in (r.get("cards") or {}).values():
            for e in srcs:
                deg[e] += 1
    out.append((
        "P3b tail is resolvable", len(deg) >= MIN_ENTRIES,
        f"{len(deg)} distinct entries cited (need >= {MIN_ENTRIES})",
        f"DERIVED, not chosen. Separating a 10x tail from a 3x one needs the top "
        f"1% to hold >= {MIN_TAIL} entries for {TAIL_PRECISION:.0%} relative precision, "
        f"so n >= {MIN_ENTRIES}. The previous floor of 500 sat exactly on the cliff "
        f"its own consequence string named -- E1.1d's pattern."))
    need = ("bank_size", "fleet_size", "rollouts_per_adapter")
    out.append((
        "P3c ratios recorded", all(k in r0 and r0.get(k) is not None for k in need),
        f"present: {[k for k in need if k in r0 and r0.get(k) is not None]}",
        "E0.2f: breadth quantities are sensitive to bank-to-fleet-to-rollout "
        "ratio, so all three must be recoverable or the degree number is not "
        "interpretable. THIS PREDICATE PREVIOUSLY CHECKED `'cards' in rows[0]` "
        "and could not fail on any usable asset -- an operation that cannot "
        "report failure, which the method commitments tabulate three times."))

    # ---- P4: representativeness, bounded rather than closed -----------------
    srcs = {r.get("source") for r in rows if r.get("source")}
    out.append((
        "P4 at least two independent sources", len(srcs) >= 2,
        f"{len(srcs)} distinct source(s): {sorted(srcs) or 'none recorded'}",
        "Representativeness cannot be checked WITHIN one log, but heterogeneity "
        "ACROSS logs can. Two sources converts unknown representativeness into "
        "measured spread -- the same visibility trade as everything else here. "
        "It bounds the problem; it does not close it."))
    return out


def report(rows: list, label: str) -> bool:
    print(f"\n  {label}: {len(rows)} interactions\n")
    ok_all = True
    for pid, ok, detail, why in predicates(rows):
        print(f"  {'ok' if ok else 'FIRES':>6}  {pid}")
        print(f"          {detail}")
        if not ok:
            ok_all = False
            for line in why.split(". "):
                if line.strip():
                    print(f"          -> {line.strip().rstrip('.')}.")
        print()
    return ok_all


BREAKS = ("P0", "P0a", "P1a", "P1b", "P1c", "P1d",
          "P2a", "P2b", "P2c", "P2d", "P3a", "P3b", "P3c", "P4")


def _synthetic(break_: str = "") -> list:
    rows = []
    n = MIN_ENTRIES + 400
    for i in range(n):
        cid, rid = f"c{i%200}", f"c{(i+1)%200}"
        cards = {cid: [f"e{(i*7+j) % (MIN_ENTRIES+200)}" for j in range(4)],
                 rid: [f"e{(i*11+j) % (MIN_ENTRIES+200)}" for j in range(4)]}
        r = {
            "query_id": f"q{i}", "domain": f"d{i%8}", "domain_source": "capture",
            "source": "corpusA" if i % 2 else "corpusB",
            "candidates": [{"card_id": cid, "score": 0.9 - (i % 7) * 0.05},
                           {"card_id": rid, "score": 0.4 + (i % 5) * 0.03}],
            "chosen": cid, "scorer": "embedding_sum",
            "k_logged": 2, "next_score": 0.3,
            "outcome": "pass" if i % 3 else "fail",
            "resolved_by": ["executable", "judge", "replay", "human"][i % 4],
            "difficulty": (i % 10) / 10.0, "difficulty_source": "pre_registered",
            "cards": cards, "bank_admission": "none",
            "bank_size": 200, "fleet_size": 16, "rollouts_per_adapter": 6.0,
        }
        b = break_
        if b == "P0":   r["resolved_by"] = "sealed_suite"
        if b == "P0a":  r["resolved_by"] = r["scorer"]
        if b == "P1a":  r["difficulty"] = None
        if b == "P1b":  r["difficulty_source"] = "outcome"
        if b == "P1c":  r["difficulty"] = 0.5
        if b == "P1d":  r["domain_source"] = "inferred"
        if b == "P2a":  r["scorer"] = "cross_encoder"
        if b == "P2b":
            r["next_score"] = None
            r["candidates"] = r["candidates"] + [{"card_id": "cX", "score": 0.1}]
            r["cards"]["cX"] = ["e1"]          # keep P2c satisfied
        if b == "P2c":  r["cards"] = {cid: cards[cid]}
        if b == "P2d":  r["candidates"] = [{"card_id": cid, "score": 0.9},
                                           {"card_id": rid, "score": 0.4}]
        if b == "P3a":  r["bank_admission"] = "cosine<=0.93"
        if b == "P3b":  r["cards"] = {cid: [f"e{j}" for j in range(4)],
                                     rid: [f"e{j}" for j in range(4, 8)]}
        if b == "P3c":  r.pop("bank_size")
        if b == "P4":   r["source"] = "corpusA"
        rows.append(r)
    return rows


def self_test() -> int:
    print("\nSELF-TEST -- EVERY predicate is checked against an asset built to")
    print("violate exactly it, and the pass direction is checked too. Two")
    print("instances of `an operation that cannot report failure` were sitting in")
    print("this file: P3c could not fail on any asset, and the self-test covered")
    print("6 of 11 while the docstring claimed all -- with the unfireable one")
    print("inside the untested set, so the defects concealed each other.\n")

    fails = 0
    base = _synthetic()
    ok = report(base, "adequate asset")
    if not ok:
        fails += 1                       # 1a.4: this was printed and never read
    print(f"  adequate asset passes everything: "
          f"{'ok' if ok else 'NO -- the validator rejects a compliant asset'}\n")
    print("  " + "-" * 70)

    defined = [p for p, _, _, _ in predicates(base)]
    uncovered = [p for p in defined if not any(p.startswith(b + " ") for b in BREAKS)]
    if uncovered:
        print(f"  UNCOVERED predicates (no break case): {uncovered}")
        fails += len(uncovered)

    for b in BREAKS:
        fired = {p for p, o, _, _ in predicates(_synthetic(b)) if not o}
        hit = any(p.startswith(b + " ") for p in fired)
        extra = sorted(p.split()[0] for p in fired if not p.startswith(b + " "))
        print(f"  break {b:<5} -> {'ok  ' if hit else 'MISS'}"
              f"  fires: {sorted(p.split()[0] for p in fired) or 'NOTHING'}"
              + (f"   (also: {extra})" if extra else ""))
        if not hit:
            fails += 1
    print()
    if fails:
        print(f"  {fails} failure(s). Fix before use.\n")
        return 1
    print(f"  all {len(defined)} predicates fire on an asset built to violate them,")
    print("  and a compliant asset passes every one.\n")
    return 0


def main(argv: list) -> int:
    if "--self-test" in argv:
        return self_test()
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = report(rows, str(path))
    print("  ASSET B IS ADEQUATE" if ok else
          "  ASSET B DOES NOT SUPPORT EVERY MEASUREMENT -- see the fired predicates")
    print("  A fired predicate names the ONE measurement it blocks. Read them per")
    print("  measurement, never as a total. P1b, P1d and P3a are the three that")
    print("  CANNOT BE RETROFITTED -- each needs a decision at capture time.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
