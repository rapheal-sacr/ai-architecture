"""Influence computed functionally, so ground truth cannot share the mechanism's model.

The first E0.2 established that a two-hop influence model is closed under
two-hop closure, which is a fact about the code and not about the design. Its
`true_influencers` enumerated the same two hop types that `record_provenance`
enumerated, so the "correct" answer was the mechanism's own definition wearing a
different name, and no world could have produced a different verdict.

This module fixes that structurally. Ground truth is never enumerated. The world
is a computation from entry values to adapter weights:

    adapter_weights = f(entry_values, alive_set)

and an entry truly influences an adapter iff deleting it moves that adapter's
weights. That is the exact operation a tombstone performs, it is computed by
running the world twice, and it is independent of every provenance policy --
including ones nobody has thought of.

It also adds an influence path that no set-based provenance closure can capture.
Rollouts do not arrive pre-assigned to cards; the system RETRIEVES a card per
query, and which card wins depends on the card's value, which depends on its
source entries. So deleting an entry can flip which card a rollout used, and the
rollout's whole content changes discontinuously. Provenance records the card
that *was* selected. It cannot record the card that *would have been* selected,
because that counterfactual was never run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POLICIES = ("direct", "transitive")


@dataclass
class InfluenceWorld:
    """A ledger -> cards -> retrieved-card-conditioned rollouts -> adapters pipeline."""

    dim: int
    n_entries: int
    n_cards: int
    n_rollouts: int
    n_adapters: int
    rng: field(default=None)

    def __post_init__(self) -> None:
        r = self.rng
        self.values = r.normal(size=(self.n_entries, self.dim))

        # Each card distils an overlapping slice of the ledger.
        per = max(self.n_entries // self.n_cards, 2)
        self.card_sources = [
            sorted(r.choice(self.n_entries, size=per, replace=False).tolist())
            for _ in range(self.n_cards)
        ]

        # Each rollout carries a query. Which card conditions it is decided at
        # generation time by retrieval, not fixed in advance.
        self.queries = r.normal(size=(self.n_rollouts, self.dim))
        self.rollout_direct = [
            ([int(r.integers(0, self.n_entries))] if r.random() < 0.30 else [])
            for _ in range(self.n_rollouts)
        ]

        per_a = max(self.n_rollouts // self.n_adapters, 1)
        self.adapter_rollouts = [
            list(range(a * per_a, min((a + 1) * per_a, self.n_rollouts)))
            for a in range(self.n_adapters)
        ]

    # -- the world as a pure function of (values, alive) --------------------

    def card_values(self, alive: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_cards, self.dim))
        for c, srcs in enumerate(self.card_sources):
            live = [e for e in srcs if alive[e]]
            if live:
                out[c] = self.values[live].mean(axis=0)
        return out

    def selected_cards(self, alive: np.ndarray) -> np.ndarray:
        """Retrieval: each rollout takes the card its query matches best.

        THIS is the third path. The argmax depends on every entry feeding every
        card, so deleting one entry can hand a rollout to a different card
        entirely -- and no record of the losing card exists anywhere.
        """
        cv = self.card_values(alive)
        norms = np.linalg.norm(cv, axis=1, keepdims=True)
        cv_n = cv / np.maximum(norms, 1e-12)
        sims = self.queries @ cv_n.T                    # (n_rollouts, n_cards)
        return np.argmax(sims, axis=1)

    def rollout_contents(self, alive: np.ndarray) -> np.ndarray:
        cv = self.card_values(alive)
        sel = self.selected_cards(alive)
        out = np.zeros((self.n_rollouts, self.dim))
        for r_i in range(self.n_rollouts):
            parts = [cv[sel[r_i]]]
            parts.extend(self.values[e] for e in self.rollout_direct[r_i] if alive[e])
            out[r_i] = np.mean(parts, axis=0)
        return out

    def adapter_weights(self, alive: np.ndarray) -> np.ndarray:
        rc = self.rollout_contents(alive)
        out = np.zeros((self.n_adapters, self.dim))
        for a, rolls in enumerate(self.adapter_rollouts):
            if rolls:
                out[a] = rc[rolls].mean(axis=0)
        return out

    # -- ground truth -------------------------------------------------------

    def true_influenced_adapters(self, eid: int, tol: float = 1e-12) -> set[int]:
        """Adapters whose weights actually move when this entry is deleted.

        Two runs of the world. No enumeration, no hop types, no provenance.
        """
        alive = np.ones(self.n_entries, dtype=bool)
        before = self.adapter_weights(alive)
        alive[eid] = False
        after = self.adapter_weights(alive)
        moved = np.linalg.norm(before - after, axis=1)
        return {int(a) for a in np.where(moved > tol)[0]}

    def selection_flips(self, eid: int) -> int:
        """How many rollouts switch to a different card when this entry is deleted."""
        alive = np.ones(self.n_entries, dtype=bool)
        base = self.selected_cards(alive)
        alive[eid] = False
        return int(np.sum(base != self.selected_cards(alive)))

    # -- what each provenance policy would invalidate -----------------------

    def policy_invalidates(self, eid: int, policy: str) -> set[int]:
        """Adapters the cascade fires on, given what the system recorded.

        Provenance is recorded against the world as it actually ran -- the card
        each rollout did select. That is the only thing observable at compile
        time.
        """
        alive = np.ones(self.n_entries, dtype=bool)
        sel = self.selected_cards(alive)

        hit = set()
        for a, rolls in enumerate(self.adapter_rollouts):
            prov: set[int] = set()
            for r_i in rolls:
                prov.update(self.rollout_direct[r_i])
                if policy == "transitive":
                    prov.update(self.card_sources[sel[r_i]])
            if eid in prov:
                hit.add(a)
        return hit
