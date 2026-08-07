"""E0.2 -- Does the tombstone cascade actually reach the weights?

CLAIM UNDER TEST (L7 card, "Deletion finally reaches weights"):

    "A tombstone cascades to every adapter whose provenance contains the entry,
     forcing recompilation. Unlearning by construction rather than by
     approximation."

    I4: "Every active adapter carries the ledger entry set it was compiled
     from, and can be regenerated from it."

This is the strongest safety claim in the whole design, and the one thing L7
offers that EWC and LwF cannot. It holds only if recorded provenance covers
every path by which an entry reached the weights.

Part I section 7 describes a path that plausibly is not covered:

    "the trajectories the solver produces *while reading the card* are the
     on-policy training data the adapter learns from... the card is what
     *generates the adapter's training set*."

So influence flows  entry -> card -> card-conditioned rollout -> adapter,
while the adapter record stores "the ledger entry IDs it was compiled from" --
and compilation sees rollouts, not cards. If provenance is recorded at the
point of compilation, entries that shaped the card never appear in it, the
cascade never fires for them, and their contribution stays baked into cached
weights.

ARMS:
    direct      provenance = entries the trajectories cited outright
    transitive  provenance = closure through the conditioning card as well

KILL CRITERIA (pre-registered):
    U1 fails if any adapter's weights, after the cascade, differ from what they
       should be with the entry genuinely gone. Measured as staleness: compare
       the weights the system holds against a clean recompile from survivors.
    U2 fails if recorded provenance is a proper subset of true influence for
       any adapter -- I4 says the adapter "can be regenerated from" its
       recorded set, which is false if that set is incomplete.

Both are absolute, not statistical. "Unlearning by construction" admits no
error bar; a single leaking adapter reduces it to unlearning by approximation,
which is the property L7 exists to beat.

NOTE ON U1's MEASUREMENT. The first version of this experiment tested whether
weights *changed* when the deleted value was substituted for another, and
reported zero leakage for a policy with 186 missing influence paths. That test
cannot work: cached weights are constants. They do not vary with the deleted
entry -- they are frozen with it already averaged in, and the leak is that they
were never recomputed. Only comparison against the correct post-deletion value
detects it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.ledger import LedgerWorld  # noqa: E402

DIM = 16
N_ENTRIES = 60
N_CARDS = 8
N_ROLLOUTS = 40
N_ADAPTERS = 6
SEED = 20260806


def build(policy: str, rng: np.random.Generator) -> LedgerWorld:
    """A ledger where adapters are trained on card-conditioned rollouts.

    This is the promotion path the design actually describes: L1 entry ->
    L3 skill card -> rollouts produced while reading the card -> L7 adapter.
    """
    w = LedgerWorld(dim=DIM)
    for e in range(N_ENTRIES):
        w.add_entry(e, rng.normal(size=DIM))

    # Cards distil disjoint slices of the ledger.
    per_card = N_ENTRIES // N_CARDS
    for c in range(N_CARDS):
        w.add_card(c, list(range(c * per_card, (c + 1) * per_card)))

    # Most rollouts are card-conditioned and cite nothing directly -- the
    # solver is following the card's guidance, which is the point of a card.
    for r in range(N_ROLLOUTS):
        card = int(rng.integers(0, N_CARDS))
        if rng.random() < 0.30:
            direct = [int(rng.integers(0, N_ENTRIES))]
        else:
            direct = []
        w.add_rollout(r, card=card, direct_entries=direct)

    per_adapter = N_ROLLOUTS // N_ADAPTERS
    for a in range(N_ADAPTERS):
        w.add_adapter(a, list(range(a * per_adapter, (a + 1) * per_adapter)), policy=policy)
    return w


def run_arm(policy: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    w = build(policy, rng)

    # U2: is recorded provenance complete?
    missing_total = 0
    incomplete_adapters = 0
    for aid in w.adapters:
        missing = w.true_influencers(aid) - w.adapters[aid].provenance
        missing_total += len(missing)
        incomplete_adapters += bool(missing)

    # U1: tombstone every entry in turn, cascade, measure what stayed stale.
    leaks = 0
    trials = 0
    worst = 0.0
    cascade_sizes = []
    for eid in range(N_ENTRIES):
        probe = build(policy, np.random.default_rng(seed))   # fresh world per trial
        invalidated = probe.tombstone(eid)
        cascade_sizes.append(len(invalidated))
        for aid, resid in probe.leakage_after_cascade().items():
            if eid not in probe.true_influencers(aid):
                continue      # never influenced this adapter; nothing to scrub
            trials += 1
            if resid > 1e-12:
                leaks += 1
                worst = max(worst, resid)

    u1 = leaks == 0
    u2 = incomplete_adapters == 0

    return {
        "policy": policy,
        "adapters_with_incomplete_provenance": incomplete_adapters,
        "total_missing_provenance_entries": missing_total,
        "mean_cascade_size": round(float(np.mean(cascade_sizes)), 2),
        "leaking_adapter_entry_pairs": leaks,
        "leak_trials": trials,
        "leak_rate": round(leaks / max(trials, 1), 4),
        "worst_residual_norm": round(worst, 6),
        "U1_no_residual": bool(u1),
        "U2_provenance_complete": bool(u2),
        "verdict": "PASS" if (u1 and u2) else "FAIL",
    }


def main() -> int:
    rows = [run_arm(p, SEED) for p in ("direct", "transitive")]

    hdr = (
        f"{'provenance policy':<20}{'incomplete':>12}{'missing':>9}{'cascade':>9}"
        f"{'leaks':>7}{'rate':>8}{'worst':>9}{'U1':>5}{'U2':>5}  verdict"
    )
    print(
        f"\nE0.2  Does the tombstone cascade reach the weights?"
        f"   ({N_ENTRIES} entries, {N_CARDS} cards, {N_ADAPTERS} adapters)\n"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['policy']:<20}{r['adapters_with_incomplete_provenance']:>12}"
            f"{r['total_missing_provenance_entries']:>9}{r['mean_cascade_size']:>9.2f}"
            f"{r['leaking_adapter_entry_pairs']:>7}{r['leak_rate']:>8.3f}"
            f"{r['worst_residual_norm']:>9.4f}"
            f"{'ok' if r['U1_no_residual'] else 'no':>5}"
            f"{'ok' if r['U2_provenance_complete'] else 'no':>5}"
            f"  {r['verdict']}"
        )
    print(
        f"\n  incomplete  adapters of {N_ADAPTERS} whose recorded provenance is a proper"
        f" subset of true influence"
        f"\n  missing     total (adapter, entry) influence relationships absent from"
        f" the record"
        f"\n  cascade     mean adapters invalidated per tombstone"
        f"\n  leaks/rate  (adapter, entry) pairs whose weights stayed stale after the"
        f" cascade, of those the entry genuinely influenced"
        f"\n  worst       largest gap between held weights and correct post-deletion weights"
        f"\n\n  U1 no residual dependence after cascade.  U2 recorded provenance"
        f" complete (I4 regenerability).\n"
    )

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e0_2_transitive_unlearning.json"
    out.write_text(json.dumps({"seed": SEED, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
