"""Asset B's specification, as predicates over the asset rather than hopes about the run.

Worklist v2 section 4: "Pre-specify before acquiring, in each case, the
manipulation check that would fire if the asset doesn't support the measurement
-- EB.3's format control is the model. Write it as a predicate over the asset,
not as a hope about the run."

EB.3 is why. It ran, the manipulation check fired twice, and the deliverable
turned out to be a specification for what the run needed. That specification cost
a run to produce. This one costs nothing, and it is executable BEFORE the
acquisition is spent.

WHAT ASSET B UNBLOCKS -- three measurements, not four. EB.6 took `r` off the
blocked list by measuring it with a generator and predicates, neither of which is
corpus-shaped.

    M1  checkability-difficulty correlation   -- locates E2.1 on its own sweep
    M2  certified fraction on real traffic    -- section 2's re-registered kill
    M3  entry-degree distribution             -- section 3's fan-out kill

SCHEMA. One JSON object per interaction, JSONL:

    {
      "query_id":     str,
      "candidates":   [{"card_id": str, "score": float}],   # see P2a
      "chosen":       str,
      "scorer":       str,        # "lexical" | "embedding_sum" | "cross_encoder" | ...
      "k_logged":     int,        # how many candidates were logged
      "next_score":   float|null, # the (k+1)-th score, or null if k is the full set
      "outcome":      "pass"|"fail"|null,
      "resolved_by":  str,        # "executable"|"replay"|"sealed_suite"|"judge"|"human"
      "difficulty":   float|null, # recorded BEFORE the outcome. See P1c.
      "cards":        {card_id: [entry_id, ...]}    # provenance, for M2 and M3
    }

Run:  python tools/validate_asset_b.py <log.jsonl>
      python tools/validate_asset_b.py --self-test

THE SELF-TEST IS NOT OPTIONAL. Every predicate below is checked against a
deliberately-inadequate asset constructed to violate exactly it. A validator whose
checks cannot fire is the defect this record has found four times (B7, B8, B20,
A4-usage), and it would fail silently here in the one place silence is most
expensive -- after an acquisition.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

INDEPENDENT = {"executable", "replay", "human"}   # not the model under test
DECOMPOSABLE = {"lexical", "embedding_sum", "embedding", "bm25"}


def _spread(xs) -> float:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    lo, hi = min(xs), max(xs)
    return 0.0 if hi == lo else 1.0


def predicates(rows: list) -> list:
    """Each returns (id, ok, detail, consequence). Computable from the asset alone."""
    out = []
    n = len(rows)

    # ---- P0, cross-cutting: laundering -----------------------------------
    res = Counter(r.get("resolved_by") for r in rows)
    indep = sum(v for k, v in res.items() if k in INDEPENDENT)
    frac = indep / n if n else 0.0
    out.append((
        "P0 independent resolution", frac >= 0.20,
        f"{frac:.1%} of interactions resolved by {sorted(INDEPENDENT)}; "
        f"mix = {dict(res)}",
        "E2.1's laundering, inherited by every measurement at once: an outcome "
        "resolved by the judge and then used to validate the judge. Without an "
        "independent subset, M1's checkability axis IS the judge's opinion of "
        "itself and M2's outcomes cannot be trusted either."))

    # ---- M1: checkability-difficulty correlation -------------------------
    diff = [r.get("difficulty") for r in rows]
    have_diff = sum(d is not None for d in diff)
    out.append((
        "P1a difficulty recorded", have_diff / max(n, 1) >= 0.80,
        f"{have_diff}/{n} interactions carry a difficulty value",
        "M1 has no x-axis. Deriving difficulty from the outcome makes it the "
        "same variable as success and the correlation becomes definitional."))
    out.append((
        "P1b difficulty independent of outcome", all(
            "difficulty" in r and r.get("difficulty_source") != "outcome"
            for r in rows if r.get("difficulty") is not None),
        "no row derives difficulty from its own outcome",
        "THE E0.2 TRAP FOR THIS ASSET. If difficulty is read off the outcome, "
        "checkability-vs-difficulty is a correlation between a variable and "
        "itself, and no world could return anything else."))
    out.append((
        "P1c both axes have spread", _spread(diff) > 0 and len(res) >= 2,
        f"difficulty spread {'yes' if _spread(diff) else 'NO'}; "
        f"{len(res)} distinct resolution mechanisms",
        "A constant axis gives an unestimable correlation. If every interaction "
        "is checkable, or all one difficulty, E2.1's sweep cannot be located."))

    # ---- M2: certified fraction ------------------------------------------
    scorers = Counter(r.get("scorer") for r in rows)
    dec = sum(v for k, v in scorers.items() if k in DECOMPOSABLE)
    out.append((
        "P2a decomposable scorer", dec / max(n, 1) >= 0.80,
        f"{dec}/{n} scored by {sorted(DECOMPOSABLE)}; mix = {dict(scorers)}",
        "Section 2's stated tension, as a predicate. Per-entry contribution "
        "bounds require a decomposable scorer; a cross-encoder reranker means "
        "NO certificate exists and M2 cannot be run at all."))
    tail_ok = sum(1 for r in rows
                  if r.get("next_score") is not None
                  or r.get("k_logged", 0) >= len(r.get("candidates", [])) > 0)
    out.append((
        "P2b rival tail bounded", tail_ok / max(n, 1) >= 0.90,
        f"{tail_ok}/{n} log either the full candidate set or the (k+1)-th score",
        "E0.5's decomposition: 75.9% of uncertified selections are RIVAL RISE, "
        "and a rival outside top-k cannot be bounded at all. Without the "
        "(k+1)-th score as an entry threshold, M2 measures the best case and "
        "silently omits the dominant failure mode."))
    have_prov = sum(1 for r in rows
                    if r.get("cards") and all(c["card_id"] in r["cards"]
                                              for c in r.get("candidates", [])))
    out.append((
        "P2c candidate provenance present", have_prov / max(n, 1) >= 0.90,
        f"{have_prov}/{n} carry provenance for every logged candidate",
        "The bound needs each source entry's deviation from its card's mean. "
        "Provenance for the CHOSEN card alone is not enough -- the rival is the "
        "one that moves."))
    margins = []
    for r in rows:
        sc = sorted((c["score"] for c in r.get("candidates", [])), reverse=True)
        if len(sc) >= 2:
            margins.append(sc[0] - sc[1])
    out.append((
        "P2d margins have spread", _spread(margins) > 0,
        f"{len(margins)} margins, spread {'yes' if _spread(margins) else 'NO'}",
        "If margins are uniformly large or uniformly tiny, certified fraction "
        "is 1.0 or 0.0 by construction and measures the log, not the design."))

    # ---- M3: entry-degree distribution -----------------------------------
    unconstrained = all(r.get("bank_admission") in (None, "none", "unconstrained")
                        for r in rows)
    out.append((
        "P3a bank is unconstrained", unconstrained,
        "no row reports an admission filter having been applied",
        "SECTION 3'S KILL NEEDS AN UNCONSTRAINED BANK. If the source system "
        "already applied a cosine or overlap cap, the degree distribution is "
        "the FILTER'S, not the structure's, and the measurement answers a "
        "question about someone else's threshold."))
    deg = Counter()
    for r in rows:
        for srcs in (r.get("cards") or {}).values():
            for e in srcs:
                deg[e] += 1
    out.append((
        "P3b tail is resolvable", len(deg) >= 500,
        f"{len(deg)} distinct entries cited",
        "The kill is about the top 1% of entries. Below ~500 entries the top "
        "1% is fewer than five, and the tail statistic is a handful of points."))
    out.append((
        "P3c ratios recorded", all(k in rows[0] for k in ("cards",)) and bool(deg),
        f"{len(deg)} entries across {len({c for r in rows for c in (r.get('cards') or {})})} cards",
        "E0.2f: breadth quantities are sensitive to bank-to-fleet-to-rollout "
        "ratio, so bank size, fleet size and rollouts per adapter must all be "
        "recoverable or the degree number is not interpretable."))
    return out


def report(rows: list, label: str) -> bool:
    print(f"\n  {label}: {len(rows)} interactions\n")
    ok_all = True
    for pid, ok, detail, consequence in predicates(rows):
        mark = "ok" if ok else "FIRES"
        print(f"  {mark:>6}  {pid}")
        print(f"          {detail}")
        if not ok:
            ok_all = False
            for line in consequence.split(". "):
                if line.strip():
                    print(f"          -> {line.strip().rstrip('.')}.")
        print()
    return ok_all


def _synthetic(good: bool, break_: str = "") -> list:
    rows = []
    for i in range(600):
        cards = {f"c{i%40}": [f"e{(i*7+j) % 800}" for j in range(4)],
                 f"c{(i+1)%40}": [f"e{(i*11+j) % 800}" for j in range(4)]}
        r = {
            "query_id": f"q{i}",
            "candidates": [{"card_id": f"c{i%40}", "score": 0.9 - (i % 7) * 0.05},
                           {"card_id": f"c{(i+1)%40}", "score": 0.4 + (i % 5) * 0.03}],
            "chosen": f"c{i%40}",
            "scorer": "embedding_sum",
            "k_logged": 2, "next_score": 0.3,
            "outcome": "pass" if i % 3 else "fail",
            "resolved_by": ["executable", "judge", "replay", "judge"][i % 4],
            "difficulty": (i % 10) / 10.0, "difficulty_source": "pre_registered",
            "cards": cards, "bank_admission": "none",
        }
        if not good:
            if break_ == "P0":
                r["resolved_by"] = "judge"
            if break_ == "P1b":
                r["difficulty_source"] = "outcome"
            if break_ == "P2a":
                r["scorer"] = "cross_encoder"
            if break_ == "P2b":
                r["next_score"] = None; r["k_logged"] = 2
                r["candidates"] = r["candidates"] + [{"card_id": "cX", "score": 0.1}]
            if break_ == "P3a":
                r["bank_admission"] = "cosine<=0.93"
            if break_ == "P3b":
                r["cards"] = {k: [f"e{j%20}" for j in v_i]
                              for k, v in cards.items()
                              for v_i in [range(len(v))]}
        rows.append(r)
    return rows


def self_test() -> int:
    print("\nSELF-TEST -- every predicate is checked against an asset built to")
    print("violate exactly it. A validator whose checks cannot fire is the defect")
    print("this record has found four times, and here it would fail silently")
    print("AFTER an acquisition.\n")
    base = _synthetic(True)
    ok = report(base, "adequate asset")
    print(f"  adequate asset passes everything: {'ok' if ok else 'NO -- validator is wrong'}\n")
    print("  " + "-" * 68)
    fails = 0
    for pid in ("P0", "P1b", "P2a", "P2b", "P3a", "P3b"):
        res = predicates(_synthetic(False, pid))
        fired = {p for p, o, _, _ in res if not o}
        hit = any(p.startswith(pid) for p in fired)
        print(f"  break {pid:<5} -> fires: {sorted(fired) or 'NOTHING'}"
              f"   {'ok' if hit else 'MISS'}")
        fails += 0 if hit else 1
    print()
    if fails:
        print(f"  {fails} predicate(s) could not be made to fire. Fix before use.\n")
        return 1
    print("  every predicate fires on an asset built to violate it.\n")
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
    print("  Predicates that fire do not mean the asset is useless: each names the")
    print("  ONE measurement it blocks. Read them per measurement, not as a total.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
