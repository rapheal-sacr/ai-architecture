"""L1 ledger, L3 skill cards, and L7 adapters, with explicit provenance.

Enough of the stack to test whether the tombstone cascade actually reaches
everything a deleted entry influenced.

The L7 card says a tombstone "cascades to every adapter whose provenance
contains the entry, forcing recompilation -- unlearning by construction rather
than by approximation." That holds only if recorded provenance covers every
path by which an entry reached the weights. Part I section 7 describes a path
that may not be covered:

    "the trajectories the solver produces *while reading the card* are the
     on-policy training data the adapter learns from. The card does not get
     replaced by the adapter -- the card is what *generates the adapter's
     training set*."

So the influence path is  entry -> card -> card-conditioned rollout -> adapter,
while the L7 adapter record stores "the ledger entry IDs it was compiled from."
Whether those are the same set is the question.

Everything here is linear and additive so influence can be computed exactly
rather than estimated. A tombstone that leaves any dependence on the deleted
value is leakage, and we can prove it by substitution rather than by measuring
a norm and arguing about the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROVENANCE_POLICIES = ("direct", "transitive")


@dataclass
class Entry:
    """An L1 ledger entry."""

    eid: int
    value: np.ndarray
    tombstoned: bool = False


@dataclass
class SkillCard:
    """An L3 skill card, distilled from a set of L1 entries."""

    cid: int
    source_entries: list[int]


@dataclass
class Rollout:
    """A trajectory. May be conditioned on a skill card, and may cite entries directly."""

    rid: int
    card: int | None
    direct_entries: list[int]


@dataclass
class Adapter:
    """An L7 adapter, compiled from a set of rollouts.

    `provenance` is what the *system records* -- which is the whole point. The
    cascade fires on recorded provenance, not on true influence, and the gap
    between the two is what this module exists to measure.
    """

    aid: int
    rollouts: list[int]
    provenance: set[int] = field(default_factory=set)
    weights: np.ndarray | None = None


class LedgerWorld:
    """A small ledger plus the derived views compiled from it."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.entries: dict[int, Entry] = {}
        self.cards: dict[int, SkillCard] = {}
        self.rollouts: dict[int, Rollout] = {}
        self.adapters: dict[int, Adapter] = {}

    # -- construction ------------------------------------------------------

    def add_entry(self, eid: int, value: np.ndarray) -> None:
        self.entries[eid] = Entry(eid=eid, value=value)

    def add_card(self, cid: int, source_entries: list[int]) -> None:
        self.cards[cid] = SkillCard(cid=cid, source_entries=list(source_entries))

    def add_rollout(self, rid: int, card: int | None, direct_entries: list[int]) -> None:
        self.rollouts[rid] = Rollout(rid=rid, card=card, direct_entries=list(direct_entries))

    def add_adapter(self, aid: int, rollouts: list[int], policy: str) -> None:
        a = Adapter(aid=aid, rollouts=list(rollouts))
        a.provenance = self.record_provenance(a, policy)
        a.weights = self.compile_adapter(a)
        self.adapters[aid] = a

    # -- the two provenance policies ---------------------------------------

    def record_provenance(self, adapter: Adapter, policy: str) -> set[int]:
        """What the system writes into the adapter record as its ledger provenance.

        `direct`      only entries the trajectories cited outright. This is the
                      literal reading of "the ledger entry IDs it was compiled
                      from" when compilation sees rollouts, not cards.
        `transitive`  closure through the conditioning card as well, so an entry
                      that shaped the card that shaped the rollout is recorded.
        """
        if policy not in PROVENANCE_POLICIES:
            raise ValueError(f"unknown provenance policy: {policy}")

        prov: set[int] = set()
        for rid in adapter.rollouts:
            r = self.rollouts[rid]
            prov.update(r.direct_entries)
            if policy == "transitive" and r.card is not None:
                prov.update(self.cards[r.card].source_entries)
        return prov

    # -- compilation -------------------------------------------------------

    def card_value(self, cid: int) -> np.ndarray:
        """A card is the distillation of its (live) source entries."""
        live = [self.entries[e].value for e in self.cards[cid].source_entries
                if not self.entries[e].tombstoned]
        if not live:
            return np.zeros(self.dim)
        return np.mean(live, axis=0)

    def rollout_content(self, rid: int) -> np.ndarray:
        """A trajectory carries the card that conditioned it, plus what it cited."""
        r = self.rollouts[rid]
        parts = []
        if r.card is not None:
            parts.append(self.card_value(r.card))
        parts.extend(
            self.entries[e].value for e in r.direct_entries
            if not self.entries[e].tombstoned
        )
        if not parts:
            return np.zeros(self.dim)
        return np.mean(parts, axis=0)

    def compile_adapter(self, adapter: Adapter) -> np.ndarray:
        """Weights are the mean of the training trajectories' content."""
        if not adapter.rollouts:
            return np.zeros(self.dim)
        return np.mean([self.rollout_content(r) for r in adapter.rollouts], axis=0)

    # -- the cascade -------------------------------------------------------

    def tombstone(self, eid: int) -> list[int]:
        """Delete an entry and cascade to every adapter whose *recorded* provenance holds it.

        Returns the adapter ids that were invalidated and recompiled. Adapters
        the cascade misses keep their cached weights -- with the deleted value
        still baked in, which is exactly the leak we are looking for.
        """
        self.entries[eid].tombstoned = True
        invalidated = []
        for a in self.adapters.values():
            if eid in a.provenance:
                a.weights = self.compile_adapter(a)   # recompile from survivors
                invalidated.append(a.aid)
        return invalidated

    # -- measurement -------------------------------------------------------

    def leakage_after_cascade(self) -> dict[int, float]:
        """How far each adapter's weights sit from what deletion should have made them.

        Call after `tombstone`. For every adapter, `compile_adapter` now returns
        the correct post-deletion weights, because the tombstoned entry is
        excluded from every card and trajectory it fed. `adapter.weights` is
        what the system actually holds -- recompiled if the cascade fired, and
        otherwise the cached vector from before the deletion. The difference is
        the leak.

        Measuring staleness rather than sensitivity matters. An earlier version
        of this checked whether weights *changed* when the deleted value was
        substituted, and reported zero leakage for a policy with 186 missing
        influence paths. Cached weights are constants: they do not vary with
        the deleted entry, they are frozen with it already averaged in. The leak
        is that they never got recomputed, and only a comparison against the
        correct value can see it.
        """
        return {
            a.aid: float(np.linalg.norm(a.weights - self.compile_adapter(a)))
            for a in self.adapters.values()
        }

    def true_influencers(self, aid: int) -> set[int]:
        """Every entry that genuinely reaches this adapter, by any path.

        Ground truth. The system never computes this -- Rig A does, so we can
        compare it against what the system recorded.
        """
        a = self.adapters[aid]
        influencers: set[int] = set()
        for rid in a.rollouts:
            r = self.rollouts[rid]
            influencers.update(r.direct_entries)
            if r.card is not None:
                influencers.update(self.cards[r.card].source_entries)
        return influencers
