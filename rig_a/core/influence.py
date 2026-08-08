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
    card_overlap: float = 1.0
    content_rotation: float = 0.0
    card_sources_override: list = None      # supply an externally-built bank

    def __post_init__(self) -> None:
        r = self.rng
        self.values = r.normal(size=(self.n_entries, self.dim))

        # `card_overlap` is a MULTIPLICITY: how many cards each participating
        # entry feeds. That is the knob L3's admission control actually turns --
        # SESA admits a candidate card only at cosine <= 0.93 against the bank,
        # which bounds how much cards may duplicate one another, which bounds
        # how many cards one entry can feed, which is cascade breadth.
        #
        # Participation is held FIXED as multiplicity varies. An earlier version
        # drew shared sources from a pool whose size shrank as overlap rose, so
        # raising "overlap" mostly reduced how many entries fed any card at all,
        # and cascade breadth moved the wrong way for a reason that had nothing
        # to do with card duplication.
        m = max(int(round(self.card_overlap)), 1)
        base = max(self.n_entries // self.n_cards, 2)
        participants = r.choice(
            self.n_entries, size=min(base * self.n_cards, self.n_entries), replace=False
        )
        # Each participant appears in exactly m cards; card size grows with m so
        # the participating set stays the same.
        slots: list[list[int]] = [[] for _ in range(self.n_cards)]
        for i, e in enumerate(participants):
            for j in range(m):
                slots[(i + j * 1) % self.n_cards].append(int(e))
        self.card_sources = [sorted(set(s)) if s else [int(participants[0])] for s in slots]
        # Allow an externally-constructed bank so the SAME object can be measured
        # by both this module's functional influence graph and another
        # experiment's synthetic draw -- which is the only way to tell whether a
        # disagreement between them is the world or the instrument.
        if self.card_sources_override is not None:
            self.card_sources = [sorted(s) for s in self.card_sources_override]
            self.n_cards = len(self.card_sources)
        self.entry_multiplicity = m

        # Per-card distillation projection. A skill card is not the mean of its
        # source entries -- it is a distillation that captures one PATTERN in
        # them, and two cards distilled from the same entries can capture
        # different patterns and come out near-orthogonal in content while
        # sharing every source.
        #
        # `content_rotation` interpolates between those regimes:
        #   0  content is the raw source mean, so content cosine tracks
        #      provenance overlap exactly -- which is an artifact of the
        #      construction, not a property of card banks
        #   1  content is an independently rotated view of the same information,
        #      so content cosine is decoupled from provenance overlap
        #
        # This exists so that "does admission control bound cascade breadth" can
        # be asked as an intervention on content cosine, rather than inferred
        # from a sweep that moves content and provenance together.
        self.rotations = []
        for _ in range(self.n_cards):
            q, _ = np.linalg.qr(r.normal(size=(self.dim, self.dim)))
            self.rotations.append(q)

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
        """Card content: what retrieval matches against and admission control scores.

        Still a pure function of the live source entries -- deleting a source
        still moves the card, so influence still propagates -- but the
        distillation is a per-card projection rather than a raw mean, so content
        similarity need not track provenance overlap.
        """
        t = self.content_rotation
        out = np.zeros((self.n_cards, self.dim))
        for c, srcs in enumerate(self.card_sources):
            live = [e for e in srcs if alive[e]]
            if live:
                mean = self.values[live].mean(axis=0)
                out[c] = (1.0 - t) * mean + t * (self.rotations[c] @ mean)
        return out

    def provenance_overlap(self) -> float:
        """Mean Jaccard overlap of source sets -- the quantity breadth actually tracks.

        This is what a provenance-aware admission rule would have to bound. It
        is a different object from `mean_card_cosine`, which scores content.
        """
        sets = [set(s) for s in self.card_sources]
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                vals.append(len(sets[i] & sets[j]) / u if u else 0.0)
        return float(np.mean(vals)) if vals else 0.0

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

    def mean_card_cosine(self) -> float:
        """Mean pairwise cosine between card values -- what admission control bounds.

        SESA's rule is stated as a cosine cap on the card bank. This is the
        observable side of `card_overlap`, so a cascade-breadth result can be
        quoted against the threshold a practitioner would actually set.
        """
        alive = np.ones(self.n_entries, dtype=bool)
        cv = self.card_values(alive)
        n = np.linalg.norm(cv, axis=1, keepdims=True)
        cvn = cv / np.maximum(n, 1e-12)
        sims = cvn @ cvn.T
        iu = np.triu_indices(self.n_cards, k=1)
        return float(np.mean(sims[iu]))

    def union_cascade(self, eids: list[int]) -> set[int]:
        """Adapters truly moved by deleting a BATCH of entries together.

        The eager policy pays cascade breadth once per tombstone. A batching
        policy pays it once per window, over the union -- and when cascades
        overlap heavily the union saturates after a handful of deletions, so the
        two costs diverge sharply.
        """
        alive = np.ones(self.n_entries, dtype=bool)
        before = self.adapter_weights(alive)
        for e in eids:
            alive[e] = False
        after = self.adapter_weights(alive)
        moved = np.linalg.norm(before - after, axis=1)
        return {int(a) for a in np.where(moved > 1e-12)[0]}

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
