"""E0.1 -- I4, verified recompilability. The largest unmeasured claim in the design.

CLAIM UNDER TEST, as amended (docs/wam_amendment_I2_I4_rev2.md, A):

    For an adapter A promoted with provenance P, and S(A) the provenance-indexed
    suite items A passed at promotion: recompiling from P's surviving image
    yields A'. Partition S(A) by support status --

        surviving support         A' must PASS   (failing = over-forgetting)
        fully tombstoned support  A' must FAIL   (passing = under-forgetting)
        partially tombstoned      unconstrained  (reported as a coverage gap)

Part II section G calls I4 "the invariant the entire design leans on" and "a
design intention rather than a tested property". Every other tier's safety
argument reduces to it.

WHY THE UNAMENDED FORM CANNOT FAIL. "Every active adapter carries the ledger
entry set it was compiled from and can be regenerated from it" is a statement
about records. A system that records provenance perfectly and recompiles to
something useless satisfies it, and with a deterministic recompiler it is true
by construction. So this tests the amended, two-sided, verified form.

AND THE WEIGHTING RULE APPLIES TO THIS EXPERIMENT'S OWN INSTRUMENT -- the third
place the same defect could land and the most dangerous. A recompile that
preserves frequent-region competence and loses rare-region competence registers
as I4 HOLDING under any pooled count. That is precisely the predicted failure,
because the provenance most likely to have decayed or been superseded is the
sparse, rare-region kind: use-based decay evicts what is rarely referenced. So
the arm most likely to show over-forgetting is the arm a pooled metric is
blindest to. Every rate below is reported worst-region alongside pooled, with
the ratio as a blindness factor.

MECHANISM. Recompile cost is O(|L3 slice|) -- the material actually re-read, not
the provenance IDs, which are integers and tiny. So the draw is capped and the
question is which entries the cap admits. A usage-weighted draw is the natural
implementation and it is exactly the traffic weighting the rule forbids for a
protection statistic.

ARMS (amendment rev 2):
    A0  control      nothing varies. A' must equal A, or every other arm is
                     uninterpretable. Its job is to be a control, not a result.
    A1  ledger drift use-based decay between compile and recompile
    A2  tombstone    delete supporting entries
    A3  harness      change the draw policy, one component at a time
    A4  ontology     re-partition regions only -- nothing else moves
    A5  cost         |P| and ledger length; report MAX, since |P| is heavy-tailed
    A6  draw-bound   retain |P| in full, cap the L3 draw. Ordered BEFORE any |P|
                     cap is designed, because provenance IDs are tiny and the cap
                     may be attacking the wrong quantity

KILL CRITERIA (pre-registered):
    K0 A0 fails if A' != A at all.
    K1 A1 fails if any surviving-support item that A passed, A' fails.
    K2 A2 fails if any fully-tombstoned item that A passed, A' still passes.
    K4 A4 fails if a pure ontology re-partition changes competence at all -- the
       ontology is not supposed to be load-bearing for adapter competence, and if
       it is, Root 3 reaches further than Part III claims.
    K6 A6 fails if capping the draw does NOT bound cost, or if it bounds cost
       only by sacrificing worst-region competence.
    KB (applies to every arm) fails if worst-region over-forgetting exceeds the
       pooled rate by 2x or more -- a pooled-only E0.1 would then report a pass
       while rare regions lose competence.

Is there a world that produces the other verdict? For K1, yes: with no draw cap
and no decay, every surviving item keeps full support and A' = A. That is A0,
and it is included so a null result is visible as a null. For KB, yes: a
stratified draw allocates equally across regions and the blindness ratio goes to
1 -- swept in A6.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

N_REGIONS = 12
ENTRIES_PER_REGION = 50
N_ENTRIES = N_REGIONS * ENTRIES_PER_REGION
N_ITEMS = 360
SUPPORT_MIN, SUPPORT_MAX = 2, 5
PASS_THRESHOLD = 0.5          # fraction of an item's support that must be drawn
DRAW_CAP = 300                # |L3 slice| -- what recompile actually re-reads
SEED = 20260806
N_SEEDS = 8
BLINDNESS_LIMIT = 2.0

POLICIES = ("usage", "uniform", "stratified")


class World:
    """A ledger, a provenance-indexed suite, and a recompiler with a bounded draw."""

    def __init__(self, rng, ontology_shift: int = 0):
        self.rng = rng
        self.region = np.repeat(np.arange(N_REGIONS), ENTRIES_PER_REGION)
        # Zipfian reference rates: what use-based decay and a usage-weighted
        # draw both key off. The rare tail is the long-tail personal knowledge
        # the ledger exists to preserve.
        rates = 1.0 / np.arange(1, N_REGIONS + 1, dtype=float)
        self.rates = rates / rates.sum()
        self.usage = self.rates[self.region] * rng.uniform(0.5, 1.5, size=N_ENTRIES)

        # Suite items, each supported by entries drawn from ONE region, so
        # per-region competence is well defined.
        self.items = []
        for _ in range(N_ITEMS):
            r = int(rng.choice(N_REGIONS, p=self.rates))
            k = int(rng.integers(SUPPORT_MIN, SUPPORT_MAX + 1))
            pool = np.where(self.region == r)[0]
            self.items.append({"region": r,
                               "support": set(rng.choice(pool, size=k, replace=False).tolist())})

        self.alive = np.ones(N_ENTRIES, dtype=bool)
        self.provenance = set(range(N_ENTRIES))
        # A4's only moving part: which region an entry is assigned to. Rotating
        # the assignment re-partitions without touching the ledger, the weights,
        # or the harness.
        self.ontology_shift = ontology_shift

    def strata(self) -> np.ndarray:
        return (self.region + self.ontology_shift) % N_REGIONS

    # -- the recompiler ----------------------------------------------------

    def draw(self, policy: str, cap: int | None) -> set[int]:
        """The L3 slice: what a recompile actually re-reads. Cost is O(|draw|).

        DETERMINISM, PER PATH -- read this before interpreting any `identical`
        column. Three arms in this experiment report `identical: yes` and they
        have three different statuses, distinguishable ONLY by which draw path
        they take. That was a fact about the code nobody had written down, so it
        had to be re-derived by reading, and a clean result sat under a cloud in
        the meantime.

            uncapped (cap=None, or len(live) <= cap)
                DETERMINISTIC. Returns the whole live set. No rng touched.

            policy="usage"
                DETERMINISTIC. `argsort(-usage)` is a total order on a fixed
                array; no rng touched. Two compiles with the same live set return
                the SAME draw, always.

            policy="uniform"
                STOCHASTIC. `rng.choice` over the live set.

            policy="stratified"
                STOCHASTIC. `rng.choice` within each region.

        EVERY ARM, AND WHICH PATH IT TAKES. Five of the nine are deterministic
        and therefore NEED NO NULL-TREATMENT ROW; four reach `rng.choice` and do.
        Listing only the interesting ones is what left A1a under a cloud it did
        not deserve, so the table is complete rather than illustrative.

            DETERMINISTIC -- no rng, null derivable from the code path

            A0 control        usage, no drift.  `identical: yes` is TRUE BY
                              CONSTRUCTION and that is its job -- A0 is a control,
                              not a result. If it ever came out `no`, the
                              recompiler would be non-deterministic and every
                              other arm uninterpretable.

            A1a uncapped      cap=None, so the EARLY RETURN below fires before the
                              policy branch is ever reached. The draw is the whole
                              live set; the only thing decay varies is which
                              entries are alive. The matched null is DERIVABLE,
                              not empirical: alive unchanged => equal draws =>
                              equal passes => over-forgetting exactly 0. A control
                              here would have been dead weight. Its 12.4x blindness
                              factor has no draw policy in it and stands.

            A1b usage cap     usage, with decay. `identical: yes` is a FINDING:
                              use-based decay removes the least-used entries,
                              which the usage-ordered cap had already excluded, so
                              the draw cannot move. That is what "I4 is a relative
                              invariant" rests on -- and it rests on the draw
                              being deterministic AND usage-ordered, not on
                              determinism alone.

            A2 tombstone      usage. Deletion changes `alive`, not the ordering
                              rule, so the draw moves only through the live set.

            A4 usage draw     usage, with an ontology shift. `identical: yes` was
                              a BUG (B8): the usage path never reads `strata()`,
                              so the manipulation could not reach the measured
                              quantity. Scored on the stratified arm instead, K4
                              FAILS. Retained as a control, asserted inert.

            STOCHASTIC -- reaches rng.choice, so a null-treatment row is required

            A3 harness        uniform. Its 0.216 is measured against a USAGE-draw
                              baseline, so it confounds harness drift with
                              resampling. Component-granular stamping loses its
                              measured support until the matched null runs (2.4).
            A4 strat          stratified. The arm K4 is scored on.
            A6 uniform        uniform, with decay.
            A6 stratified     stratified, with decay. NOT matched to A4: A6 carries
                              decay=0.25 and A4 carries none, so the
                              stratified-vs-usage tradeoff is measured across a
                              decay difference. Open control defect (2.5).

        Same shape as `identical: yes` three times over: one construction, one
        finding, one bug. The annotation is the difference.
        """
        live = np.array(sorted(self.provenance & set(np.where(self.alive)[0].tolist())))
        # THE EARLY RETURN. `cap is None` exits before the policy branch, so an
        # uncapped arm is deterministic and policy-independent no matter what
        # `policy` says. A1a is that arm.
        if cap is None or len(live) <= cap:
            return set(live.tolist())
        if policy == "usage":
            order = live[np.argsort(-self.usage[live])]
            return set(order[:cap].tolist())
        if policy == "uniform":
            return set(self.rng.choice(live, size=cap, replace=False).tolist())
        # stratified: equal budget per region -- the weighting-rule-compliant draw
        strata = self.strata()
        per = max(cap // N_REGIONS, 1)
        out: list[int] = []
        for r in range(N_REGIONS):
            pool = live[strata[live] == r]
            if len(pool):
                take = min(per, len(pool))
                out.extend(self.rng.choice(pool, size=take, replace=False).tolist())
        return set(out[:cap])

    def compile(self, policy="usage", cap=DRAW_CAP) -> dict:
        d = self.draw(policy, cap)
        passes = np.array([len(it["support"] & d) / len(it["support"]) >= PASS_THRESHOLD
                           for it in self.items])
        return {"draw": d, "cost": len(d), "passes": passes}

    # -- support classification (amendment rev 2, three categories) ---------

    def support_class(self) -> np.ndarray:
        dead = set(np.where(~self.alive)[0].tolist())
        out = np.empty(N_ITEMS, dtype=object)
        for i, it in enumerate(self.items):
            hit = it["support"] & dead
            if not hit:
                out[i] = "surviving"
            elif hit == it["support"]:
                out[i] = "fully"
            else:
                out[i] = "partial"
        return out


def rates_by_region(w: World, num: np.ndarray, den: np.ndarray
                    ) -> tuple[float, float, float]:
    """Pooled rate, worst-region rate, and the blindness ratio between them.

    Both masks are FULL LENGTH over items; `den` selects the eligible population
    and `num` the events within it. Passing a pre-masked subarray and then
    indexing it with a full-length region selector is a length mismatch, which is
    how the first run of this crashed.
    """
    regions = np.array([it["region"] for it in w.items])
    pooled = float(num[den].mean()) if den.any() else 0.0
    per = []
    for r in range(N_REGIONS):
        sel = den & (regions == r)
        if sel.sum() >= 5:
            per.append(float(num[sel].mean()))
    worst = max(per) if per else pooled
    return pooled, worst, worst / max(pooled, 1e-9)


def arm(name: str, seed: int, policy="usage", cap=DRAW_CAP,
        decay=0.0, tombstone=0.0, ontology_shift=0,
        recompile_policy=None) -> dict:
    """`recompile_policy` is what makes a HARNESS-DRIFT arm possible at all.

    Without it, compile and recompile both run under `policy`, so an arm that
    varies nothing else varies NOTHING between the two compiles -- the rng
    advances and a stochastic draw comes out different, and that difference is
    the resampling floor, not a treatment. B20: the original A3 was exactly that
    and was labelled `harness drift`.
    """
    rng = np.random.default_rng(seed)
    w = World(rng)
    before = w.compile(policy, cap)

    # ---- what varies between compile and recompile ----------------------
    if decay > 0:
        # use-based decay: least-referenced entries go first, which is exactly
        # the rare-region tail
        n = int(decay * N_ENTRIES)
        order = np.argsort(w.usage)
        w.provenance -= set(order[:n].tolist())
    if tombstone > 0:
        n = int(tombstone * N_ENTRIES)
        w.alive[rng.choice(N_ENTRIES, size=n, replace=False)] = False
    w.ontology_shift = ontology_shift

    after = w.compile(recompile_policy or policy, cap)
    cls = w.support_class()

    surviving = cls == "surviving"
    fully = cls == "fully"
    partial = cls == "partial"

    # over-forgetting: passed before, surviving support, fails now
    over = before["passes"] & surviving & ~after["passes"]
    # under-forgetting: passed before, fully tombstoned, still passes
    under = before["passes"] & fully & after["passes"]

    o_pool, o_worst, o_blind = rates_by_region(w, over, surviving)
    u_pool = float(under[fully].mean()) if fully.any() else 0.0

    return {
        "arm": name, "policy": policy, "cap": cap,
        "recompile_policy": recompile_policy or policy,
        "identical": bool(np.array_equal(before["passes"], after["passes"])),
        "cost_before": before["cost"], "cost_after": after["cost"],
        "n_surviving": int(surviving.sum()), "n_fully": int(fully.sum()),
        "n_partial": int(partial.sum()),
        "partial_share": round(float(partial.mean()), 3),
        "over_pooled": round(o_pool, 4), "over_worst": round(o_worst, 4),
        "over_blindness": round(o_blind, 2),
        "under_pooled": round(u_pool, 4),
    }


def many(name, **kw) -> dict:
    rs = [arm(name, SEED + i, **kw) for i in range(N_SEEDS)]

    def m(k):
        return float(np.mean([r[k] for r in rs]))

    def sd(k):
        return float(np.std([r[k] for r in rs]))

    return {"arm": name, "policy": rs[0]["policy"], "cap": rs[0]["cap"],
            "recompile_policy": rs[0]["recompile_policy"],
            "identical": all(r["identical"] for r in rs),
            "over_pooled_sd": round(sd("over_pooled"), 4),
            "over_worst_sd": round(sd("over_worst"), 4),
            "partial_share": round(m("partial_share"), 3),
            "over_pooled": round(m("over_pooled"), 4),
            "over_worst": round(m("over_worst"), 4),
            "over_blindness": round(m("over_blindness"), 2),
            "under_pooled": round(m("under_pooled"), 4),
            "cost_max": int(max(r["cost_after"] for r in rs))}


def main() -> int:
    print(f"\nE0.1  I4 -- verified recompilability   ({N_ENTRIES} entries,"
          f" {N_ITEMS} items, {N_REGIONS} regions, {N_SEEDS} seeds)\n")

    rows = [
        many("A0 control", decay=0.0, tombstone=0.0),
        many("A1a drift, uncapped", decay=0.25, cap=None),
        many("A1b drift, usage cap", decay=0.25),
        many("A2 tombstone", tombstone=0.15),
        # NULL-TREATMENT ROWS, one per policy that reaches rng.choice
        # (worklist 2.4). Nothing varies between compile and recompile; whatever
        # these report is the resampling floor of that draw policy, and every
        # stochastic arm below has to be read against its own row.
        many("N-uniform null", policy="uniform"),
        many("N-strat null", policy="stratified"),
        # A3 REBUILT (B20). The old A3 was `policy="uniform"` and nothing else,
        # so it recompiled under the SAME policy it compiled under and varied
        # nothing at all -- it was the uniform null wearing a treatment's name.
        # A harness component changing between compile and recompile is a draw
        # policy that DIFFERS across the two, which is what this is.
        many("A3 harness drift", policy="usage", recompile_policy="uniform"),
        many("A4 ontology, usage draw", ontology_shift=5),
        many("A4 ontology, strat draw", ontology_shift=5, policy="stratified"),
        many("A6 draw uniform", policy="uniform", decay=0.25),
        many("A6 draw stratified", policy="stratified", decay=0.25),
    ]

    hdr = (f"  {'arm':<22}{'ident':>7}{'partial':>9}{'over pool':>11}"
           f"{'over worst':>12}{'blind':>7}{'under':>8}{'cost max':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['arm']:<22}{('yes' if r['identical'] else 'no'):>7}"
              f"{r['partial_share']:>9.3f}{r['over_pooled']:>11.4f}"
              f"{r['over_worst']:>12.4f}{r['over_blindness']:>7.2f}"
              f"{r['under_pooled']:>8.4f}{r['cost_max']:>10}")
    print("  over pool/worst  over-forgetting on SURVIVING-support items")
    print("  blind            worst-region / pooled -- what a pooled-only E0.1 misses")
    print("  under            under-forgetting on FULLY-tombstoned items")
    print("  partial          share of items I4 says NOTHING about (coverage gap)")
    print("\n  SCOPE LIMIT on A2: this world has no retrieval-mediated card selection,")
    print("  so E0.2b's ~8.7% irreducible under-forgetting cannot appear here. K2")
    print("  passing is a statement about THIS influence model, not a contradiction")
    print("  of E0.2b, whose whole point was a path set-based provenance cannot see.")

    a0, a1a, a1b, a2, nuni, nstrat, a3, a4u, a4s, a6u, a6s = rows

    # MANIPULATION CHECKS -- verify each arm varied what it claims to vary.
    # A1a is the arm that must move; A1b's inertness is a FINDING, not an error,
    # and is reported below rather than asserted away.
    assert not a1a["identical"], \
        "A1a manipulation did not take: uncapped decay changed nothing"
    assert a2["partial_share"] > 0, \
        "A2 manipulation did not take: no tombstoned support to classify"
    # A4's manipulation can only reach the draw under a policy that CONSULTS the
    # partition. Under `usage` the shift touches nothing, so a null there is
    # definitional -- the same defect as unanimity reporting zero unsafe. The
    # stratified arm is the one that can fail.
    assert a4u["identical"], \
        "A4 usage arm unexpectedly moved -- the usage draw should not read strata"

    k0 = a0["identical"]
    k1 = a1a["over_pooled"] == 0.0
    k2 = a2["under_pooled"] == 0.0
    # K4 DIFFERENCED (worklist 2.4). `identical` cannot be differenced and is
    # False for any stochastic arm by construction, so scoring K4 on it scored
    # the draw policy rather than the ontology. The ontology's effect is A4-strat
    # MINUS the stratified null, and it counts as real only if it clears the
    # null's own seed spread.
    k4_effect = a4s["over_pooled"] - nstrat["over_pooled"]
    k4_worst_effect = a4s["over_worst"] - nstrat["over_worst"]
    k4 = abs(k4_effect) <= nstrat["over_pooled_sd"]
    kb = max(r["over_blindness"] for r in rows) < BLINDNESS_LIMIT

    print("\n  A4 must be scored on the stratified arm. Under a usage draw the")
    print("    partition is never consulted, so a null there is definitional -- the")
    print("    same defect class as an arm reporting zero on a quantity its own")
    print("    rule already excludes. Usage arm: identical by construction.")

    print("\n  A1b is INERT, and that is the finding, not a null result.")
    print("    Under a usage-weighted capped draw, use-based decay of the tail")
    print("    changes nothing: those entries were never in the draw, so the")
    print("    competence they support was already absent at COMPILE time. I4")
    print("    compares A against A' and both are equally impoverished.")
    print("    => I4 is a RELATIVE invariant. It checks recompile fidelity, not")
    print("       compile adequacy, and cannot see competence that was never")
    print("       compiled in the first place.")

    print("\n  A5/A6 -- capping the draw bounds cost; the question is what it costs.")
    print(f"    uncapped |P|=600 draw cost   : {a1a['cost_max']}")
    print(f"    capped at DRAW_CAP={DRAW_CAP}       : max observed"
          f" {max(r['cost_max'] for r in rows if r['cap'] is not None)}")
    print(f"    uniform draw    : over pooled {a6u['over_pooled']:.4f}"
          f"  worst {a6u['over_worst']:.4f}  blindness {a6u['over_blindness']:.2f}")
    print(f"    stratified draw : over pooled {a6s['over_pooled']:.4f}"
          f"  worst {a6s['over_worst']:.4f}  blindness {a6s['over_blindness']:.2f}")
    print("    Stratified trades a HIGHER pooled rate for a lower worst-region")
    print("    rate -- exactly the weighting rule's tradeoff, and it does not")
    print("    reach 1.0x blindness.")
    print("    B19 WITHDRAWN. It claimed this comparison crossed a decay")
    print("    difference. It does not: both A6 arms carry decay=0.25 and differ")
    print("    only in draw policy, so the tradeoff is matched and citable. What")
    print("    is true is narrower -- A6 is not a matched control for A4, which")
    print("    bears on K4's differencing and not on this number.")
    print("    And the nulls are NOT the right comparator for A6. Both A6 arms")
    print("    carry decay, which shrinks provenance 600 -> 450 against a fixed")
    print("    300-entry cap, so the recompile draw covers a LARGER fraction of a")
    print("    smaller live set than the null's does. A6-uniform lands"
          f" {a6u['over_pooled'] - nuni['over_pooled']:+.4f} against its")
    print(f"    null and A6-stratified {a6s['over_pooled'] - nstrat['over_pooled']:+.4f}"
          " -- differences that measure the draw")
    print("    fraction, not the policy. The within-A6 comparison is the valid one")
    print("    because decay is held equal across it.")

    print("\n  NULL-TREATMENT ROWS AND WHAT THEY MOVE (worklist 2.4).")
    print("  Nothing varies between compile and recompile in a null row, so its")
    print("  number is the RESAMPLING FLOOR of that draw policy. Every stochastic")
    print("  arm is read against its own row, never against A0.")
    print(f"    {'':<22}{'pooled':>10}{'+-sd':>8}{'worst':>10}{'+-sd':>8}")
    for r in (nuni, nstrat, a3, a4s, a6u, a6s):
        print(f"    {r['arm']:<22}{r['over_pooled']:>10.4f}"
              f"{r['over_pooled_sd']:>8.4f}{r['over_worst']:>10.4f}"
              f"{r['over_worst_sd']:>8.4f}")
    print(f"\n    A4-strat MINUS its null : pooled {k4_effect:+.4f}"
          f"   worst {k4_worst_effect:+.4f}")
    print(f"    the null's own seed sd  : pooled {nstrat['over_pooled_sd']:.4f}")
    print("    K4 is scored on that difference, not on `identical`, which is")
    print("    False for any stochastic arm whatever the ontology does.")

    print("\n  B20 -- A3 IS REBUILT AND THE OLD NUMBER IS WITHDRAWN.")
    print("    The old A3 was `policy=\"uniform\"` and nothing else, so it")
    print("    recompiled under the SAME policy it compiled under. Nothing about")
    print("    the harness differed between the two compiles; the rng advanced and")
    print("    a stochastic draw came out different. Its 0.216 was the uniform")
    print("    resampling floor wearing a treatment's name, and it is now the")
    print("    N-uniform row. A3 proper compiles under usage and recompiles under")
    print("    uniform, which is a harness component actually changing.")
    print(f"    A3 rebuilt : pooled {a3['over_pooled']:.4f}"
          f"  worst {a3['over_worst']:.4f}")

    print(f"\n  K0 A0 recompiler deterministic:                {'ok' if k0 else 'NO'}")
    print(f"  K1 A1 no over-forgetting on surviving support: {'ok' if k1 else 'NO'}")
    print(f"  K2 A2 no under-forgetting on dead support:     {'ok' if k2 else 'NO'}")
    print(f"  K4 A4 ontology re-partition inert:             {'ok' if k4 else 'NO'}"
          f"   (scored on the STRATIFIED arm)")
    print(f"  KB worst-region within {BLINDNESS_LIMIT}x of pooled:        "
          f"{'ok' if kb else 'NO'}")
    print(f"\n  VERDICT: {'PASS' if (k0 and k1 and k2 and k4 and kb) else 'FAIL'}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_1_verified_recompilability.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_seeds": N_SEEDS, "draw_cap": DRAW_CAP, "rows": rows,
         "K0_deterministic": bool(k0), "K1_no_over_forgetting": bool(k1),
         "K2_no_under_forgetting": bool(k2), "K4_ontology_inert": bool(k4),
         "KB_not_blind": bool(kb),
         "k4_effect_pooled": round(k4_effect, 4),
         "k4_effect_worst": round(k4_worst_effect, 4),
         "null_seed_sd_pooled": nstrat["over_pooled_sd"],
         "verdict": "PASS" if (k0 and k1 and k2 and k4 and kb) else "FAIL"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
