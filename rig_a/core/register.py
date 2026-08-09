"""The recording layer: what the system writes down instead of estimating later.

Worklist Phase 1. Two objects, both pure recording -- no loop changes, no gate
changes, nothing here decides anything. Each one turns a quantity that is
currently inferred or invisible into a record.

    SelectionJournal   (1.1)  what retrieval actually chose, and by how much
    cover / residual   (1.3)  which provenance no live owner defends

WHY A JOURNAL AND NOT A CLOSURE. E0.2b established that transitive provenance
recalls 0.913 and that no set-based closure can reach 1.0, because the missing
dependency is on a retrieval that was never executed. Provenance records the card
that WAS selected; it cannot record the card that WOULD have been. The escape is
not a better closure -- it is to stop needing the counterfactual. Whether a
selection COULD have flipped is decidable from the scores that were already
computed, and those are free to record at the moment they exist.

A NOTE ON WHAT IS RECORDABLE. Everything the journal stores is available to the
system at the moment it retrieves: the query, the candidate scores, the card
norms, the projection norm, and each source entry's distance from its card's
mean. Nothing here reads the world twice or consults a value the system would not
have. That constraint is the whole point -- a journal that needs privileged
access is not a journal, it is an oracle, and it would fail I9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 1.1 -- the selection journal
# ---------------------------------------------------------------------------


@dataclass
class CardBounds:
    """Per-card constants a certificate needs, all known at compile time.

    `src_dev[e]` is ||v_e - mean_c||: how far one source pulls its card's mean.
    That is the per-entry upper bound the rev 2 schema calls for, and it is a
    bound rather than a contribution because the score is normalised and
    therefore not linear in the card value.
    """

    k: int                                  # number of live sources
    norm: float                             # ||cv_c||
    proj_norm: float                        # ||M_c||_2, the distillation projection
    src_dev: dict                           # entry id -> ||v_e - mean_c||


@dataclass
class Selection:
    """One recorded retrieval. This is the whole schema from rev 2 section 2."""

    query_id: int
    scores: np.ndarray                      # (n_cards,) -- candidates+scores
    chosen: int
    query_norm: float

    @property
    def margin(self) -> float:
        """Score gap to the runner-up. Recorded for reporting; the certificate
        uses the per-candidate gaps, because the runner-up is not necessarily
        the candidate a given deletion moves most."""
        s = np.sort(self.scores)
        return float(s[-1] - s[-2]) if len(s) > 1 else float("inf")


@dataclass
class SelectionJournal:
    """Records every retrieval, and certifies which ones a deletion cannot flip.

    THE CERTIFICATE. After deleting a set E, card c's value moves from cv to cv'.
    Writing S for c's sources, m = |S n E|, and mean for c's source mean:

        mean - mean'  =  [ sum_{e in S n E} v_e  -  m * mean ] / (k - m)

    so  ||mean - mean'||  <=  sum_{e in S n E} ||v_e - mean|| / (k - m)  and,
    since cv = M_c @ mean, the perturbed value cv' lies in a ball of radius

        rho_c  =  ||M_c||_2 * sum_dev / (k - m)

    around cv. The score is q . cv/||cv||, so what matters is how far the ball
    lets the card's DIRECTION rotate. For rho < ||cv|| the reachable unit vectors
    are exactly a spherical cap of half-angle phi = arcsin(rho/||cv||) about
    cv/||cv||, which gives a two-sided bound rather than a symmetric one:

        upper_c  =  ||q|| cos(max(theta - phi, 0))
        lower_c  =  ||q|| cos(min(theta + phi, pi))       cos(theta) = score/||q||

    The chosen card survives if, for EVERY other candidate c,

        lower_chosen  >  upper_c

    Every term is recorded. No recompile, no counterfactual, no world re-run.

    WHY THE CAP AND NOT THE OBVIOUS NORM BOUND. The one-line alternative is
    |delta score| <= 2||q|| rho / ||cv||, and it was what this first used. It is
    sound, and it is loose by roughly 2/sin(theta) -- which does not matter for a
    yes/no on one selection and matters enormously for a THRESHOLD, because the
    experiment below reports the support level k at which certification becomes
    viable. A lazy bound would have moved that threshold by a factor of two and
    the number would have been an artifact of the instrument. Cheap here, so
    there is no excuse for the loose form.

    If every source of a card is deleted the card value is exactly zero and its
    score is exactly zero -- not unbounded. That case is handled exactly rather
    than conservatively, for the same reason.

    SOUND BY CONSTRUCTION, AND THE CONSTRUCTION IS CHECKED. Each inequality above
    is an upper bound, so a certified selection provably cannot flip. `audit`
    tests that claim against a world that actually performs the deletion -- not
    because the algebra is in doubt, but because a bound that is sound on paper
    and mis-implemented in code is the exact shape of B3, which reported zero
    leakage for a policy with 186 missing paths. A postcondition some possible
    input would violate.
    """

    bounds: list = field(default_factory=list)      # per card
    log: list = field(default_factory=list)         # per selection

    # -- recording ---------------------------------------------------------

    @classmethod
    def record(cls, world, alive=None) -> "SelectionJournal":
        """Run retrieval once and write down what it saw."""
        if alive is None:
            alive = np.ones(world.n_entries, dtype=bool)

        t = world.content_rotation
        eye = np.eye(world.dim)
        cv = world.card_values(alive)
        norms = np.linalg.norm(cv, axis=1)

        bounds = []
        for c, srcs in enumerate(world.card_sources):
            live = [e for e in srcs if alive[e]]
            mean = world.values[live].mean(axis=0) if live else np.zeros(world.dim)
            m_c = (1.0 - t) * eye + t * world.rotations[c]
            bounds.append(CardBounds(
                k=len(live),
                norm=float(norms[c]),
                proj_norm=float(np.linalg.svd(m_c, compute_uv=False)[0]),
                src_dev={int(e): float(np.linalg.norm(world.values[e] - mean))
                         for e in live},
            ))

        cvn = cv / np.maximum(norms[:, None], 1e-12)
        sims = world.queries @ cvn.T
        qn = np.linalg.norm(world.queries, axis=1)
        log = [Selection(query_id=i, scores=sims[i].copy(),
                         chosen=int(np.argmax(sims[i])), query_norm=float(qn[i]))
               for i in range(world.n_rollouts)]
        return cls(bounds=bounds, log=log)

    # -- the certificate ---------------------------------------------------

    def _score_interval(self, card: int, deleted: set, sel: "Selection") -> tuple:
        """(lower, upper) on card c's score after this deletion. Sound both sides."""
        b = self.bounds[card]
        score = float(sel.scores[card])
        hit = deleted & set(b.src_dev)
        if not hit:
            return score, score
        if len(hit) >= b.k:                     # every source gone: value is exactly 0
            return 0.0, 0.0

        rho = b.proj_norm * sum(b.src_dev[e] for e in hit) / (b.k - len(hit))
        qn = max(sel.query_norm, 1e-12)
        if rho >= b.norm:                       # the ball contains the origin
            return -qn, qn
        phi = np.arcsin(rho / b.norm)
        theta = np.arccos(np.clip(score / qn, -1.0, 1.0))
        return (qn * np.cos(min(theta + phi, np.pi)),
                qn * np.cos(max(theta - phi, 0.0)))

    def certified(self, deleted) -> np.ndarray:
        """Boolean per selection: provably could not have flipped under this deletion."""
        deleted = set(int(e) for e in deleted)
        out = np.zeros(len(self.log), dtype=bool)
        touched = {c for c, b in enumerate(self.bounds) if deleted & set(b.src_dev)}
        n_cards = len(self.bounds)
        for i, sel in enumerate(self.log):
            if not touched:
                out[i] = True
                continue
            lo_ch, _ = self._score_interval(sel.chosen, deleted, sel)
            ok = True
            for c in range(n_cards):
                if c == sel.chosen:
                    continue
                _, hi_c = self._score_interval(c, deleted, sel)
                if hi_c >= lo_ch:
                    ok = False
                    break
            out[i] = ok
        return out

    # -- the postcondition -------------------------------------------------

    def audit(self, world, deleted) -> dict:
        """Compare the certificate against a world that really performs the deletion.

        Returns the counts that decide whether the mechanism is sound and whether
        it is useful -- which are different questions and must not be reported as
        one number:

            violations  certified selections that flipped anyway. ANY is fatal.
            certified   the mechanism's answer
            stable      the truth: selections that did not in fact flip

        `stable - certified` is instrument slack: selections the bound was too
        loose to certify, which cost a recompile they did not need. That is a
        tightness problem. `violations` would be a soundness problem, and there
        is no threshold at which it is acceptable.
        """
        alive = np.ones(world.n_entries, dtype=bool)
        base = world.selected_cards(alive)
        for e in deleted:
            alive[int(e)] = False

        after = world.selected_cards(alive)
        flipped = base != after
        cert = self.certified(deleted)
        return {
            "n": len(self.log),
            "certified": int(cert.sum()),
            "stable": int((~flipped).sum()),
            "flipped": int(flipped.sum()),
            "violations": int((cert & flipped).sum()),
        }


# ---------------------------------------------------------------------------
# 1.3 -- the cover, and what it misses (R-c)
# ---------------------------------------------------------------------------


@dataclass
class ProbeRegistry:
    """R-a and R-d: probes are write-once, and deduplicated at draw time.

    R-d is one line -- "draw from the existing pool first, create only what is
    missing" -- and section 1.3's entire pricing argument rests on it. Without a
    provenance-keyed pool, two overlapping owners each author their own probes
    and the oracle cost multiplies by fleet size, which is exactly the second
    kill criterion rev 2 pre-registers against its own register.

    THE ASSUMPTION R-d MAKES, WHICH THE DOCUMENT DOES NOT STATE. Sharing works
    only if a probe is a function of PROVENANCE ALONE. If a probe tests entry e
    the way owner A uses it, it may test nothing meaningful for owner B, and then
    the key is (entry, owner) rather than (entry,) and the pool never shares.
    `context_bound` is that fraction, swept rather than assumed -- because "the
    sharing is structural" is a claim about probe semantics, and the design has
    not said what a probe is.

    Two prices, two accounts, and they must never be reported as one number:

        distinct probes         ORACLE.  Paid once, at creation. Scarce.
        probe-evaluations/cycle COMPUTE. Paid every cycle, forever. Abundant.
    """

    pool: dict = field(default_factory=dict)        # key -> probe id
    owner_sets: list = field(default_factory=list)  # owner -> list of probe ids
    created: int = 0
    reused: int = 0

    def draw_for(self, owner: int, provenance, n: int, rng,
                 context_bound: float = 0.0) -> list:
        """Equal-N draw from this owner's own provenance. I8 by construction.

        Write-once (R-a): an owner's set is drawn here and never redrawn. The
        pool is consulted first (R-d), so an entry already carrying a probe
        contributes no new oracle cost.
        """
        src = sorted(provenance)
        if not src:
            self.owner_sets.append([])
            return []
        take = rng.choice(src, size=min(n, len(src)), replace=False)
        ids = []
        for e in take:
            bound = rng.random() < context_bound
            key = (int(e), owner) if bound else (int(e),)
            if key in self.pool:
                self.reused += 1
            else:
                self.pool[key] = len(self.pool)
                self.created += 1
            ids.append(self.pool[key])
        self.owner_sets.append(ids)
        return ids

    @property
    def distinct(self) -> int:
        """The ORACLE line: probes that had to be authored or harvested."""
        return len(self.pool)

    @property
    def evaluations(self) -> int:
        """The COMPUTE line: probe-evaluations for one full fleet cycle."""
        return sum(len(s) for s in self.owner_sets)


def owner_provenance(world, alive=None, policy: str = "transitive") -> list:
    """What each owner records as its provenance -- the cover's building block.

    An owner here is an adapter. This is the same recording rule E0.2b measures
    against, deliberately: R-c's question is what a cover built from THIS rule
    fails to reach, not what a better rule would reach.
    """
    if alive is None:
        alive = np.ones(world.n_entries, dtype=bool)
    sel = world.selected_cards(alive)
    out = []
    for rolls in world.adapter_rollouts:
        prov = set()
        for r_i in rolls:
            prov.update(e for e in world.rollout_direct[r_i] if alive[e])
            if policy == "transitive":
                prov.update(e for e in world.card_sources[sel[r_i]] if alive[e])
        out.append(prov)
    return out


def cover_report(world, live_owners=None, retired_owners=None, alive=None) -> dict:
    """R-c's measurement: how much live provenance nothing defends.

    THE DISTINCTION R-c EXISTS FOR. A retirement-only residual set holds the
    probes of owners that left. A TOTAL one holds probes for all provenance no
    live owner covers. Those differ by exactly the provenance that never had an
    owner at all -- entries whose cards no query ever selected, or which no card
    cites. That surface has no owner, therefore no probe set, therefore no
    protection statistic: it is ABSENT rather than under-weighted, and nothing
    reports its loss. Both numbers are returned so the gap is visible.
    """
    if alive is None:
        alive = np.ones(world.n_entries, dtype=bool)
    prov = owner_provenance(world, alive)
    n_owners = len(prov)
    live_owners = set(range(n_owners)) if live_owners is None else set(live_owners)
    retired_owners = set() if retired_owners is None else set(retired_owners)

    live_entries = {int(e) for e in np.where(alive)[0]}
    covered_live = set().union(*[prov[a] for a in live_owners]) if live_owners else set()
    orphaned = set().union(*[prov[a] for a in retired_owners]) if retired_owners else set()

    unowned_total = live_entries - covered_live
    unowned_orphan_only = unowned_total - orphaned       # what R-c adds over R-b
    denom = max(len(live_entries), 1)
    return {
        "live_entries": len(live_entries),
        "live_owners": len(live_owners),
        "covered": len(covered_live),
        "unowned_fraction": len(unowned_total) / denom,
        "never_owned_fraction": len(unowned_orphan_only) / denom,
        "orphan_fraction": len(orphaned & unowned_total) / denom,
    }
