"""A synthetic practice world with known ground truth.

Rig A's job is to test the *dynamics* of WAM's loops without a language model
in the way. To do that we need a world where we know the answers in advance:
which regions are learnable, which are irreducibly noisy, and which skills
genuinely transfer. Then we can ask whether the architecture's detectors and
gates recover what we planted.

The key distinction this module makes, and that the design does not, is
between two sources of predictive variance:

    epistemic   parameter uncertainty. Shrinks with practice. A real gap.
    aleatoric   irreducible outcome noise. Never shrinks. Not a gap.

Part I lists "high posterior variance turns" as a gap source, reading L4's RLS
error covariance as "a free map of what the system does not know." Whether
that map is free depends entirely on which of the two quantities it measures,
and the design never says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Region:
    """One domain signature in the practice world."""

    name: str
    ceiling: float          # best pass rate achievable here, ever
    learn_rate: float       # how fast competence approaches the ceiling
    competence: float = 0.05
    visits: int = 0

    @property
    def is_noise(self) -> bool:
        """A region whose ceiling sits at chance is unlearnable by construction."""
        return self.ceiling <= 0.55

    def practice(self, n: int, rng: np.random.Generator) -> int:
        """Run n practice tasks. Returns how many the solver passed.

        Competence climbs toward the ceiling, and the climb scales with the
        number of tasks actually allocated -- each task closes `learn_rate` of
        the remaining gap, so n of them close 1 - (1-learn_rate)^n. Making the
        gain per *cycle* instead of per *task* would mean a region starved to a
        single task learned as fast as one given a hundred, and the whole
        question of what misallocated budget costs would be invisible.

        In a noise region the ceiling is chance, so competence climbs to 0.5
        and stops there permanently no matter how much practice it receives.
        """
        passes = int(rng.binomial(n, min(max(self.competence, 0.0), 1.0)))
        self.visits += n
        self.competence += (1.0 - (1.0 - self.learn_rate) ** n) * (
            self.ceiling - self.competence
        )
        return passes

    # -- the two variance readings -----------------------------------------

    def epistemic_var(self) -> float:
        """Parameter uncertainty. This is phi^T P phi -- it shrinks with data."""
        return 1.0 / (1.0 + self.visits)

    def aleatoric_var(self) -> float:
        """Irreducible outcome variance at the current competence."""
        p = min(max(self.competence, 0.0), 1.0)
        return p * (1.0 - p)

    def predictive_var(self) -> float:
        """What a calibrated gate needs: total uncertainty about the next outcome."""
        return self.epistemic_var() + self.aleatoric_var()

    def realized_error(self) -> float:
        """Ground-truth error rate. Rig A knows this; the system does not."""
        return 1.0 - min(max(self.competence, 0.0), 1.0)


def make_world(
    n_learnable: int, n_noise: int, rng: np.random.Generator
) -> list[Region]:
    """A world of mostly-learnable regions plus a few that are pure coin flips."""
    regions = []
    # learn_rate is per practice task, not per cycle, so it is small: with a
    # few tens of tasks per region per cycle these close a single-digit
    # percentage of the remaining gap each cycle.
    for i in range(n_learnable):
        regions.append(
            Region(
                name=f"learnable_{i:02d}",
                ceiling=float(rng.uniform(0.85, 0.98)),
                learn_rate=float(rng.uniform(0.002, 0.006)),
            )
        )
    for i in range(n_noise):
        regions.append(
            Region(name=f"NOISE_{i:02d}", ceiling=0.5, learn_rate=0.008)
        )
    return regions


# -- the challenger --------------------------------------------------------


def frontier_reward(p_hat: float, lo: float = 0.0, hi: float = 1.0,
                    endpoint_penalty: float = 1.0, tol: float = 1e-3) -> float:
    """SESA's bell-shaped, endpoint-penalised proposer reward.

        r_p(x) = -lambda                  if p_hat in {0, 1}
                 4(l + p_hat)(h - p_hat)  otherwise

    Note where the maximum sits. With l=0 and h=1 this is 4*p(1-p), which peaks
    at p_hat = 0.5. A region the solver passes exactly half the time earns the
    highest possible proposer reward -- and a pure coin flip sits at p = 0.5
    forever, by definition. Frontier shaping cannot tell "productively hard"
    from "random".
    """
    if p_hat <= tol or p_hat >= 1.0 - tol:
        return -endpoint_penalty
    return 4.0 * (lo + p_hat) * (hi - p_hat)


GAP_READINGS = ("predictive", "epistemic", "reducible")


@dataclass
class Challenger:
    """Allocates a practice budget across regions from a gap signal.

    `reading` selects which quantity is used as the gap score:

        predictive  epistemic + aleatoric. What a calibrated gate must use.
        epistemic   parameter variance only. Shrinks with visits regardless
                    of whether anything was actually learned.
        reducible   the candidate fix: variance *decline* attributable to
                    practice -- a derivative, not a level. A region that does
                    not improve when practised stops asking for budget.
    """

    reading: str = "predictive"
    per_source_cap: float | None = None   # Part III section 6.3
    _prev_var: dict[str, float] = field(default_factory=dict)

    def gap_scores(self, regions: list[Region]) -> np.ndarray:
        scores = []
        for r in regions:
            if self.reading == "predictive":
                s = r.predictive_var()
            elif self.reading == "epistemic":
                s = r.epistemic_var()
            elif self.reading == "reducible":
                cur = r.predictive_var()
                prev = self._prev_var.get(r.name, cur)
                # Only variance that actually fell in response to practice
                # counts as evidence that more practice would help.
                s = max(0.0, prev - cur)
            else:
                raise ValueError(f"unknown gap reading: {self.reading}")
            scores.append(s)
        return np.asarray(scores, dtype=float)

    def allocate(self, regions: list[Region], budget: int) -> np.ndarray:
        """Split `budget` practice tasks across regions.

        An RL-trained proposer maximising the frontier reward, sampling from
        the gap set, converges to allocating in proportion to the product of
        the two signals. That product is what we model.
        """
        gaps = self.gap_scores(regions)
        rewards = np.array([max(0.0, frontier_reward(r.competence)) for r in regions])
        weights = gaps * rewards

        if self.per_source_cap is not None:
            cap = self.per_source_cap * weights.sum() if weights.sum() > 0 else 0.0
            if cap > 0:
                weights = np.minimum(weights, cap)

        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(regions))
            total = weights.sum()
        share = weights / total
        return np.floor(share * budget).astype(int)

    def observe(self, regions: list[Region]) -> None:
        """Record this cycle's variance so `reducible` can difference it next cycle."""
        for r in regions:
            self._prev_var[r.name] = r.predictive_var()
