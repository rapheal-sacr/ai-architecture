"""E4.2 -- Does the blast-radius fixed point actually seal the Assay?

CLAIM UNDER TEST (Part I section 4, "Blast-radius rule"):

    "L9 may edit anything downstream of L1. It may **never** edit: L1's schema,
     the Assay, the promotion gate's tier requirements, or this rule. That is
     the fixed point. Self-modification is real and bounded, and the boundary
     is written down rather than assumed."

And Part I section 8, I5:

    "no loop may write to the artifact that scores it."

The rule names artifacts L9 may not *edit*. That is a weaker property than the
one the safety story needs, which is that L9 cannot *change what the Assay
concludes*. Two ways the weaker rule can hold while the stronger one fails:

    REACHABILITY. A verifier that L9 may not edit can still run on code L9 may
        edit. Part I section 4's own editable list includes "retrieval policy",
        and T1 counterfactual replay replays real queries -- through retrieval.
        Edit the retrieval policy and T1 returns different verdicts, without
        anyone editing T1.

    THRESHOLDS vs REQUIREMENTS. The same list includes "L7's gate thresholds".
        The forbidden list covers "the promotion gate's tier requirements".
        Those are different objects: eps, eps_reg and noise_leeway are
        thresholds; "T2 required for weight promotion" is a requirement. So L9
        may lower the bar arbitrarily while every tier requirement stays
        untouched and the rule reports compliance.

METHOD:
    Part 1  static reachability. Encode the artifact taxonomy, Part I section
            4's editable and forbidden lists verbatim, and each Assay tier's
            execution dependencies. Compute what the Assay transitively depends
            on and intersect with the editable set.
    Part 2  injection. Move only permitted thresholds and measure the change in
            promotion rate, with tier requirements held fixed throughout.

KILL CRITERIA (pre-registered):
    S1 fails if the Assay's transitive execution dependencies intersect the
       L9-editable set at all.
    S2 fails if edits confined to permitted thresholds move the promotion rate
       by more than 10 percentage points while every tier requirement is
       unchanged.

DELIVERABLE: a corrected minimal non-editable set -- the closure that would
have to be sealed for the rule to mean what it says.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

SEED = 20260806
PROMOTION_RATE_LIMIT = 0.10   # S2 threshold

# -- the artifact taxonomy and what each thing needs in order to run --------
#
# DEPENDS[x] = artifacts whose behaviour x's execution depends on. Not "what x
# is made of" -- what can change x's output without editing x.

DEPENDS: dict[str, list[str]] = {
    # Assay tiers
    "assay_t0": ["dispatch", "tool_schemas"],           # compilers/tests are invoked by the harness
    "assay_t1": ["l1_entries", "retrieval_policy", "l2_pages", "l3_views"],
    "assay_t2": ["sealed_probes", "retrieval_policy", "l2_pages", "l3_views"],
    "assay_t3": ["judge_model", "retrieval_policy"],
    # the gate that consumes them
    "promotion_gate": [
        "assay_t0", "assay_t1", "assay_t2", "assay_t3",
        "l7_gate_thresholds", "promotion_gate_tier_requirements",
    ],
    # derived views the verifiers run through
    "l2_pages": ["l1_entries", "l7_adapters"],
    "l3_views": ["l1_entries", "signature_ontology"],
    "retrieval_policy": [],
    "dispatch": [],
    "tool_schemas": [],
    "sealed_probes": [],
    "judge_model": [],
    "l1_entries": ["l1_schema"],
    "l1_schema": [],
    "l7_adapters": [],
    "l7_gate_thresholds": [],
    "signature_ontology": ["l1_entries"],
    "promotion_gate_tier_requirements": [],
    "blast_radius_rule": [],
}

# Part I section 4, verbatim: "retrieval policy (top-k, rerank, when to
# re-inject ledger text before synthesis), tool schemas and dispatch, the L5
# halting head's decision-point predicate, the compile schedule, L7's gate
# thresholds, and L6's own policy prompt."
L9_EDITABLE = {
    "retrieval_policy",
    "tool_schemas",
    "dispatch",
    "l5_decision_predicate",
    "l7_compile_schedule",
    "l7_gate_thresholds",
    "l6_policy_prompt",
}

# Part I section 4, verbatim: "L1's schema, the Assay, the promotion gate's
# tier requirements, or this rule."
NEVER_EDITABLE = {
    "l1_schema",
    "assay_t0", "assay_t1", "assay_t2", "assay_t3",
    "promotion_gate_tier_requirements",
    "blast_radius_rule",
}


def closure(roots: list[str]) -> set[str]:
    """Everything reachable from `roots` through DEPENDS."""
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        for dep in DEPENDS.get(node, []):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def static_analysis() -> dict:
    assay_roots = ["assay_t0", "assay_t1", "assay_t2", "assay_t3"]
    assay_deps = closure(assay_roots)
    leaks = sorted(assay_deps & L9_EDITABLE)

    gate_deps = closure(["promotion_gate"])
    gate_leaks = sorted(gate_deps & L9_EDITABLE)

    per_tier = {
        t: sorted(closure([t]) & L9_EDITABLE) for t in assay_roots
    }

    # Naively, the fix is "seal the whole closure". That over-seals badly: the
    # closure contains l1_entries, l2_pages, l3_views and l7_adapters, which the
    # system exists to keep updating. Freezing them freezes the system.
    #
    # The closure splits into two kinds of artifact, needing two different
    # treatments:
    #
    #   SEAL   harness code the Assay executes *through*. It has no business
    #          changing underneath a verifier at all.
    #   PIN    derived views that legitimately change, but against which a
    #          given Assay run must be reproducible. These do not need to be
    #          frozen -- they need to be version-stamped, so an Assay run names
    #          the versions it ran against. WAM already does exactly this for
    #          L2 pages and weight versions; it is the same mechanism.
    harness_code = {"retrieval_policy", "dispatch", "tool_schemas", "judge_model"}
    must_seal = sorted((NEVER_EDITABLE | (assay_deps & harness_code) | {"promotion_gate"}))
    must_pin = sorted(assay_deps - harness_code - NEVER_EDITABLE)

    return {
        "assay_dependency_closure": sorted(assay_deps),
        "leaking_artifacts": leaks,
        "leaks_per_tier": per_tier,
        "gate_closure_leaks": gate_leaks,
        "must_seal": must_seal,
        "must_pin": must_pin,
        "S1_seal_holds": len(leaks) == 0,
    }


def injection(seed: int) -> dict:
    """Move only permitted thresholds; hold every tier requirement fixed.

    The gate is Part II section E's, reduced to the three signed conditions:

        Delta_target > eps + noise_leeway
        min(Delta_off) > -eps_reg
        net transfer   > 0
    """
    rng = np.random.default_rng(seed)
    n = 20000
    d_target = rng.normal(0.02, 0.05, size=n)
    d_off = rng.normal(0.0, 0.04, size=(n, 5))
    net = d_off.sum(axis=1)

    def promote_rate(eps: float, noise_leeway: float, eps_reg: float) -> float:
        ok = (d_target > eps + noise_leeway)
        ok &= (d_off.min(axis=1) > -eps_reg)
        ok &= (net > 0)
        return float(ok.mean())

    # Incumbent settings, and settings an L9 archive would drift toward: every
    # one of these is a "gate threshold", which Part I section 4 permits it to
    # edit. No tier requirement is touched in either configuration.
    baseline = promote_rate(eps=0.01, noise_leeway=0.10, eps_reg=0.02)
    relaxed = promote_rate(eps=0.0, noise_leeway=0.0, eps_reg=0.50)

    sweep = [
        {
            "noise_leeway": round(nl, 3),
            "promotion_rate": round(promote_rate(0.01, nl, 0.02), 4),
        }
        for nl in (0.0, 0.02, 0.05, 0.10, 0.20)
    ]

    delta = relaxed - baseline
    return {
        "baseline_promotion_rate": round(baseline, 4),
        "relaxed_promotion_rate": round(relaxed, 4),
        "delta": round(delta, 4),
        "noise_leeway_sweep": sweep,
        "tier_requirements_changed": False,
        "S2_thresholds_inert": bool(abs(delta) <= PROMOTION_RATE_LIMIT),
    }


def main() -> int:
    stat = static_analysis()
    inj = injection(SEED)

    print("\nE4.2  Does the blast-radius fixed point seal the Assay?\n")
    print("Part 1 - static reachability")
    print("-" * 62)
    print(f"  Assay depends on {len(stat['assay_dependency_closure'])} artifacts transitively")
    print(f"  of which L9 may edit: {stat['leaking_artifacts'] or 'none'}")
    for tier, leaks in stat["leaks_per_tier"].items():
        print(f"    {tier:<12} reachable-and-editable: {leaks or '-'}")
    print(f"  S1 seal holds: {'ok' if stat['S1_seal_holds'] else 'NO'}")

    print("\nPart 2 - injection, tier requirements held fixed")
    print("-" * 62)
    print(f"  baseline promotion rate  {inj['baseline_promotion_rate']:.3f}"
          f"   (eps=0.01, noise_leeway=0.10, eps_reg=0.02)")
    print(f"  relaxed  promotion rate  {inj['relaxed_promotion_rate']:.3f}"
          f"   (eps=0.00, noise_leeway=0.00, eps_reg=0.50)")
    print(f"  delta                    {inj['delta']:+.3f}"
          f"   with zero tier requirements changed")
    print("\n  noise_leeway alone:")
    for s in inj["noise_leeway_sweep"]:
        print(f"    {s['noise_leeway']:>5.2f}  ->  promotion rate {s['promotion_rate']:.3f}")
    print(f"\n  S2 thresholds inert: {'ok' if inj['S2_thresholds_inert'] else 'NO'}"
          f"   (needs |delta| <= {PROMOTION_RATE_LIMIT})")

    verdict = "PASS" if (stat["S1_seal_holds"] and inj["S2_thresholds_inert"]) else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    print("\nDeliverable - the corrected boundary, split two ways.")
    print("Sealing the whole dependency closure would freeze l1_entries and")
    print("l3_views, which the system exists to keep updating. Two treatments:\n")
    print("  SEAL (harness code a verifier executes through - must not move):")
    for a in stat["must_seal"]:
        print(f"    {'+' if a not in NEVER_EDITABLE else ' '} {a}")
    print("\n  PIN (derived views that may change, but an Assay run must name")
    print("       the versions it ran against - same stamping L2 pages use):")
    for a in stat["must_pin"]:
        print(f"      {a}")
    print("\n  ('+' marks artifacts Part I section 4 currently leaves editable)\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e4_2_blast_radius_seal.json"
    out.write_text(json.dumps({"seed": SEED, "static": stat, "injection": inj,
                               "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
