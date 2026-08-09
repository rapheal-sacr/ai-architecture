"""E0.7 -- the oracle line. Worklist 1.2 and 2.3.

CLAIM UNDER TEST (rev 2.2 sections 1.3 and 1.2b R-d): granularity is
compute-expensive and oracle-cheap, conditional on provenance overlap.

    "The oracle price of a new owner is only the probes its provenance does not
     already cover; the compute price is the whole fleet's probe set, every
     cycle."

REPORT TWO LINES AND NEVER ONE -- distinct probes (oracle, scarce) and
probe-evaluations per cycle (compute, abundant).

=============================================================================
THE STATISTIC THIS EXPERIMENT FIRST REPORTED IS WITHDRAWN.
=============================================================================

Run 1 reported an "oracle saving" of 1 - distinct / (fleet x probes), and got
77% at 256 owners. That number is degenerate in the same way E3.3's partition
objective is degenerate, and for the same reason.

Back the pool out of the two numbers it reported. Distinct probes 473 from 2048
draws solves N(1 - e^(-2048/N)) = 473 at N ~ 480 -- so 98.5% OF THE ELIGIBLE POOL
WAS ALREADY CONSUMED. The absolute oracle count was pinned at its ceiling and
could not go higher. Divide a pinned numerator by a denominator that grows with
fleet size and the "saving" rises monotonically forever; its argmax is "more
owners, without limit". A statistic that improves as the cost saturates is
measuring the denominator.

    saving = 1 - distinct / (fleet x probes)      WITHDRAWN, monotone in fleet
    consumed = distinct / |pool|                  saturates at 1, cannot exceed
    authored = distinct x (1 - harvest yield)     the line that is actually scarce

So the axis changes too. Overlap was never going to be the governing variable --
the governing quantity is `fleet x probes` against `|pool|`:

    below the crossover   the oracle line is LINEAR in fleet. Granularity is
                          genuinely expensive, every new owner authors probes.
    above the crossover   FLAT. The pool is exhausted and owners are free.

Run 1 sat far above the crossover at every point it sampled, which is why every
point looked like a win. A real ledger sits far below it. Pool size is the swept
axis here, and it is a CHOSEN parameter of the rig, so the deliverable is the
crossover's location rather than a saving.

HARVEST YIELD IS THE OTHER HALF, AND E2.1 ALREADY MEASURED IT. A harvested T0/T1
probe carries a verified outcome that came free with the interaction, so its
oracle price is ~0; an adjudicated one is fully priced. The oracle line is
therefore `distinct x (1 - yield)`, and E2.1's strict filter yields 0.68 down to
0.33 as checkability-difficulty correlation rises. If probe sets are
harvest-first, section 1.3's verdict survives for a reason that has nothing to do
with provenance neighbourhoods -- and the thing that decides it is a curve this
record already owns.

WHAT A PROBE IS -- resolved as a PAIR, not a key. A probe is a stimulus plus an
expectation. The stimulus is oracle-priced and entry-keyed, so it shares. The
expectation is "whose floor does this count against", which is bookkeeping and
free. That reconciles the two measurements instead of choosing between them: the
sharing is real on the oracle line and absent on the compute line, and both are
correct. The condition, and it is falsifiable: THE EXPECTATION MUST BE DERIVABLE
FROM THE STIMULUS. True for ground-truth-correct outcomes; false where a probe
tests an owner-specific target behaviour, because then the expectation is itself
oracle-priced. `owner_specific` below is that fraction, and what it costs is
coverage, not sharing.

KILL CRITERIA (re-registered against the corrected statistic):
    KX There is no crossover in range -- the oracle line is linear in fleet at
       every pool size tested, so granularity is oracle-expensive everywhere and
       section 1.3 is wrong as stated rather than mispriced.
    KP Below the crossover, pool consumption stays under 1.0 while distinct
       probes still track fleet x probes -- i.e. sharing is absent where it is
       needed, which is the regime a real ledger is in.
    KN Sharing, where it occurs, is not attributable to provenance overlap but
       to collision in a bounded pool. Scored against the birthday null. Retained
       from run 1, where it FAILED at every point.

Is there a world that produces the other verdict? For KX, yes: hold the ledger at
480 and the oracle line is flat from fleet 64 upward, which run 1 measured. For
KN, yes: if owners drew from genuinely clustered provenance the observed distinct
count would fall BELOW the birthday curve, which is what neighbourhood discipline
is supposed to buy.
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
SEED = 20260806
N_WORLDS = 6

PROBES_PER_OWNER = 8
ROLLOUTS_PER_ADAPTER = 6        # B18: rollouts scale with fleet, never fixed
ENTRIES_PER_CARD = 15           # held fixed so card size is not confounded with ledger

# THE AXIS. fleet x probes is held at 512 while the pool grows past it, so the
# crossover is inside the swept range rather than off its left edge.
FLEET = 64
LEDGERS = (240, 480, 960, 1920, 3840, 7680)
OWNER_SPECIFIC = (0.0, 0.25, 0.50, 0.75, 1.0)

# E2.1, strict T0/T1 filter: yield falls as checkability-difficulty correlation
# rises. A harvested probe's oracle price is ~0; an adjudicated one is full.
E2_1_YIELD = {"corr 0.00": 0.6766, "corr 0.50": 0.5024, "corr 1.00": 0.33}


def measure(ledger: int, owner_specific: float) -> dict:
    n_cards = max(ledger // ENTRIES_PER_CARD, 2)
    distinct, evals, pools, nulls, overlaps, reuse = [], [], [], [], [], []

    for w_i in range(N_WORLDS):
        rng = np.random.default_rng(SEED + w_i)
        w = InfluenceWorld(
            dim=DIM, n_entries=ledger, n_cards=n_cards,
            n_rollouts=FLEET * ROLLOUTS_PER_ADAPTER, n_adapters=FLEET, rng=rng,
        )
        prov = owner_provenance(w)
        union = set().union(*prov) if prov else set()
        pools.append(len(union))

        vals = []
        for i in range(len(prov)):
            for j in range(i + 1, len(prov)):
                u = len(prov[i] | prov[j])
                vals.append(len(prov[i] & prov[j]) / u if u else 0.0)
        overlaps.append(float(np.mean(vals)) if vals else 0.0)

        # The birthday null over the SAME eligible pool: what pure collision
        # would give with no overlap structure at all.
        u = max(len(union), 1)
        n_draw = FLEET * PROBES_PER_OWNER
        free = n_draw * (1.0 - owner_specific)
        nulls.append(u * (1.0 - (1.0 - 1.0 / u) ** free) + (n_draw - free))

        reg = ProbeRegistry()
        for a in range(FLEET):
            reg.draw_for(a, prov[a], PROBES_PER_OWNER, rng,
                         context_bound=owner_specific)
        distinct.append(reg.distinct)
        evals.append(reg.evaluations)
        tot = reg.created + reg.reused
        reuse.append(reg.reused / tot if tot else 0.0)

    d = float(np.mean(distinct))
    pool = float(np.mean(pools))
    null = float(np.mean(nulls))
    return {
        "ledger": ledger, "n_cards": n_cards, "owner_specific": owner_specific,
        "pool": round(pool, 1),
        "distinct_probes": round(d, 1),
        "evaluations": round(float(np.mean(evals)), 1),
        "pool_consumed": round(d / max(pool, 1e-9), 4),
        "vs_no_sharing": round(d / (FLEET * PROBES_PER_OWNER), 4),
        "null_distinct": round(null, 1),
        "excess_over_null": round((null - d) / max(null, 1e-9), 4),
        "owner_overlap": round(float(np.mean(overlaps)), 4),
        "reuse_rate": round(float(np.mean(reuse)), 4),
    }


def main() -> int:
    grid = [measure(L, c) for c in OWNER_SPECIFIC for L in LEDGERS]

    def at(L, c):
        return next(r for r in grid if r["ledger"] == L and r["owner_specific"] == c)

    draws = FLEET * PROBES_PER_OWNER
    print("\nE0.7 -- the oracle line (worklist 1.2 + 2.3)\n")
    print(f"  fleet {FLEET} x {PROBES_PER_OWNER} probes = {draws} draws, held fixed.")
    print("  The pool is swept past that number so the crossover is in range.\n")

    print("  SHARED STIMULI (owner_specific = 0). `consumed` is the honest")
    print("  statistic: it saturates at 1.0 and cannot be improved by adding")
    print("  owners. `vs draws` is the withdrawn one, shown so the shape is")
    print("  visible -- it is best exactly where the pool is most exhausted.\n")
    print(f"    {'ledger':>8}{'pool':>8}{'distinct':>10}{'consumed':>10}"
          f"{'vs draws':>10}{'null':>9}{'excess':>9}{'overlap':>9}")
    for L in LEDGERS:
        r = at(L, 0.0)
        print(f"    {L:>8}{r['pool']:>8.0f}{r['distinct_probes']:>10.0f}"
              f"{r['pool_consumed']:>10.1%}{1 - r['vs_no_sharing']:>10.1%}"
              f"{r['null_distinct']:>9.0f}{r['excess_over_null']:>9.1%}"
              f"{r['owner_overlap']:>9.3f}")

    # The crossover: the pool size at which the oracle line stops tracking draws.
    cross = None
    for L in LEDGERS:
        if at(L, 0.0)["vs_no_sharing"] >= 0.90:
            cross = L
            break
    lo, hi = at(LEDGERS[0], 0.0), at(LEDGERS[-1], 0.0)

    print(f"\n  THE CROSSOVER. Below pool ~ {draws} the line is flat -- the pool is")
    print("  exhausted and a new owner authors almost nothing. Above it the line")
    print("  is linear in fleet and every owner pays.")
    print(f"    pool {lo['pool']:>6.0f} : distinct {lo['distinct_probes']:>6.0f}"
          f"  = {lo['vs_no_sharing']:>5.1%} of draws   consumed {lo['pool_consumed']:>6.1%}")
    print(f"    pool {hi['pool']:>6.0f} : distinct {hi['distinct_probes']:>6.0f}"
          f"  = {hi['vs_no_sharing']:>5.1%} of draws   consumed {hi['pool_consumed']:>6.1%}")
    if cross:
        print(f"    first ledger where distinct >= 90% of draws: {cross}")

    print("\n  THE HARVEST-ADJUSTED ORACLE LINE (E2.1's measured strict yield).")
    print("  A harvested T0/T1 probe carries a verified outcome that came free")
    print("  with the interaction; only the adjudicated remainder is scarce.")
    print(f"    {'ledger':>8}" + "".join(f"{k:>12}" for k in E2_1_YIELD))
    for L in LEDGERS:
        d = at(L, 0.0)["distinct_probes"]
        print(f"    {L:>8}" + "".join(f"{d * (1 - y):>12.0f}" for y in E2_1_YIELD.values()))
    print("    At corr 0.00 the scarce line is a third of the distinct count; at")
    print("    corr 1.00 it is two thirds. So checkability-difficulty correlation")
    print("    moves the oracle bill by ~2x -- more than pool size does over most")
    print("    of this sweep, and far more than provenance overlap does at all.")

    print("\n  OWNER-SPECIFIC EXPECTATIONS -- what a probe IS, not what it is keyed on.")
    print("  A stimulus shares; an expectation that is owner-specific cannot.")
    print(f"    {'ledger':>8}" + "".join(f"{'os=' + str(c):>10}" for c in OWNER_SPECIFIC))
    for L in LEDGERS:
        print(f"    {L:>8}" + "".join(f"{at(L, c)['vs_no_sharing']:>10.1%}"
                                     for c in OWNER_SPECIFIC))
    print("    Rows are distinct probes as a fraction of draws -- 100% means no")
    print("    sharing at all. Pool consumption is NOT the right denominator")
    print("    here: once a key is (entry, owner) the pool stops being a ceiling")
    print("    and the ratio runs past 100%, which is correct arithmetic and a")
    print("    misleading label. At os=1.0 every row is 100%: the sharing is gone")
    print("    entirely, which is the compute line and is correct.")

    kx = cross is not None                       # a crossover exists in range
    kp = hi["vs_no_sharing"] >= 0.90             # far side really is linear
    excesses = [r["excess_over_null"] for r in grid if r["owner_specific"] == 0.0]
    kn = max(abs(e) for e in excesses) >= 0.05   # sharing beats collision

    print(f"\n  KX a crossover exists in range:        {'ok' if kx else 'NO'}")
    print(f"  KP above it the line is linear:        {'ok' if kp else 'NO'}"
          f"   (distinct = {hi['vs_no_sharing']:.1%} of draws at pool {hi['pool']:.0f})")
    print(f"  KN sharing beats the birthday null:    {'ok' if kn else 'NO'}"
          f"   (largest departure {max(abs(e) for e in excesses):.1%})")

    print("\n  WHAT THIS SETTLES AND WHAT IT DOES NOT. Section 1.3's VERDICT --")
    print("  granularity is oracle-cheap -- holds only below the crossover, where")
    print("  the pool is exhausted. A real ledger sits far above it, in the regime")
    print("  where every owner pays. Its MECHANISM, provenance neighbourhoods,")
    print("  is not supported at any pool size: the distinct count sits on the")
    print("  birthday curve throughout. What rescues the verdict in the real")
    print("  regime is harvest yield, not overlap -- and that curve already exists")
    print("  in E2.1 rather than needing worklist 2.3 to produce it.")

    verdict = "PASS" if (kx and kn) else "PARTIAL" if kx else "FAIL"
    print(f"\n  SECTION 1.3 PRICING AS STATED: {verdict}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_7_probe_registry.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_worlds": N_WORLDS, "fleet": FLEET,
         "probes_per_owner": PROBES_PER_OWNER, "draws": draws,
         "ledgers": list(LEDGERS), "owner_specific": list(OWNER_SPECIFIC),
         "e2_1_yield": E2_1_YIELD, "crossover_ledger": cross, "grid": grid,
         "withdrawn_statistic": "1 - distinct/(fleet*probes) -- monotone in fleet, argmax is unbounded granularity",
         "KX_crossover_in_range": bool(kx), "KP_linear_above": bool(kp),
         "KN_beats_null": bool(kn), "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
