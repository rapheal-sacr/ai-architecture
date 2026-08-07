"""R_t -- the one estimator, three readings.

Part II section A claims that L4's RLS error covariance and L7's null-space
projector are the same operator, and that eigendecomposing R_t once answers
three questions the stack currently tracks in three places:

    gate    (L4)  posterior variance along the query direction  -> g
    gap set (L8)  integrated variance per domain signature
    budget  (L7)  free basis = eigenvectors with sigma_k <= eps

The budget reading is the load-bearing one, because it is the only one that
turns a continuous spectrum into a *counted* resource:

    occupied rank = #{ sigma_k > eps }

This module builds R_t from a feature stream and exposes all three readings,
so the experiments can test whether that count is well posed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RtEstimator:
    """Exponentially-weighted feature correlation matrix.

        R_t = sum_i lambda^(t-i) phi_i phi_i^T + delta I

    Rank-one updated, exactly as RLS maintains it. `lam` is the forgetting
    factor: lam = 1.0 never forgets, lam < 1 lets old commitments decay out
    of the spectrum and release rank.
    """

    dim: int
    lam: float = 1.0
    delta: float = 1e-6
    R: np.ndarray = field(init=False)
    n_updates: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.R = self.delta * np.eye(self.dim)

    def update(self, phi: np.ndarray) -> None:
        """Absorb one feature vector (or a batch, one row per observation)."""
        phi = np.atleast_2d(phi)
        for row in phi:
            self.R = self.lam * self.R + np.outer(row, row)
            self.n_updates += 1

    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Descending eigendecomposition. Returns (sigma, U) with R = U diag(sigma) U^T."""
        sigma, U = np.linalg.eigh(self.R)
        order = np.argsort(sigma)[::-1]
        return sigma[order], U[:, order]

    # -- the three readings ------------------------------------------------

    def occupied_rank(self, eps: float) -> int:
        """Budget reading (L7): how many directions count as committed."""
        sigma, _ = self.eig()
        return int(np.sum(sigma > eps))

    def free_basis(self, eps: float) -> np.ndarray:
        """Budget reading (L7): the columns an adapter is allowed to write into."""
        sigma, U = self.eig()
        return U[:, sigma <= eps]

    def posterior_variance(self, q: np.ndarray) -> float:
        """Gate reading (L4): q^T R^-1 q, the RLS posterior variance along q."""
        return float(q @ np.linalg.solve(self.R, q))


# -- feature-stream generators ---------------------------------------------
#
# The design assumes a spectrum shaped like the diagram in
# `wam_one_estimator_three_readings.png`: a block of large sigma on the left,
# a noise floor on the right, and a threshold eps you can draw between them.
# Whether real features produce that shape is the empirical question, so each
# generator below emits a *feature stream*, and R_t is accumulated from it in
# the ordinary way. We never posit a spectrum directly.


def stream_bimodal(
    n: int, dim: int, n_committed: int, floor: float, rng: np.random.Generator
) -> np.ndarray:
    """The shape the design assumes: signal in a k-dim subspace, isotropic noise elsewhere."""
    signal = rng.normal(size=(n, n_committed))
    padded = np.zeros((n, dim))
    padded[:, :n_committed] = signal
    noise = floor * rng.normal(size=(n, dim))
    return padded + noise


def stream_power_law(
    n: int, dim: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """The shape transformer features actually tend to have: sigma_k ~ k^-alpha, no knee."""
    scale = np.arange(1, dim + 1, dtype=float) ** (-alpha / 2.0)
    return rng.normal(size=(n, dim)) * scale


def stream_exponential(
    n: int, dim: int, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Intermediate case: geometric decay, a soft knee whose location depends on `rate`."""
    scale = np.exp(-rate * np.arange(dim) / 2.0)
    return rng.normal(size=(n, dim)) * scale


class DomainMixture:
    """The realistic case, and the one WAM actually runs on.

    Traffic is a mixture of domain signatures. Each domain occupies its own
    random low-rank subspace with power-law structure *within* the subspace,
    and domains are visited at unequal rates -- which is what real traffic
    looks like.

    The domain subspaces are drawn ONCE, at construction, and every `sample`
    call draws from the same world. This is not incidental. An earlier version
    was a bare function that re-drew `qr(normal(...))` on every call, so a
    train set and a "held-out" set came from unrelated subspaces -- and every
    measurement taken across them was measuring generalisation to a different
    world rather than to held-out traffic. It reported that 99% of train energy
    needed 37 directions while the same fraction of test energy needed 127 of
    128, which then read as "realistic traffic saturates the space." That
    conclusion was an artifact of the generator.
    """

    def __init__(
        self,
        dim: int,
        n_domains: int,
        domain_rank: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        self.dim = dim
        self.n_domains = n_domains
        self.domain_rank = domain_rank
        self.bases = [
            np.linalg.qr(rng.normal(size=(dim, domain_rank)))[0] for _ in range(n_domains)
        ]
        self.within = np.arange(1, domain_rank + 1, dtype=float) ** (-alpha / 2.0)
        # Zipfian visit rates: a few domains dominate the traffic.
        rates = 1.0 / np.arange(1, n_domains + 1, dtype=float)
        self.rates = rates / rates.sum()

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw n observations from this world. Returns (features, domain labels)."""
        labels = rng.choice(self.n_domains, size=n, p=self.rates)
        feats = np.empty((n, self.dim))
        for i, d in enumerate(labels):
            coeffs = rng.normal(size=self.domain_rank) * self.within
            feats[i] = self.bases[d] @ coeffs
        return feats, labels


# -- diagnostics -----------------------------------------------------------


def rank_elasticity(est: RtEstimator, eps_grid: np.ndarray) -> np.ndarray:
    """|d log(rank) / d log(eps)| across a threshold sweep.

    A *well-posed* threshold needs a plateau: some range of eps at least a
    decade wide where elasticity is near zero, meaning the occupied-rank count
    does not depend on where exactly you drew the line. A power-law spectrum
    has no such plateau, and then "occupied rank" is not a resource -- it is a
    restatement of eps.
    """
    sigma, _ = est.eig()
    ranks = np.array([max(np.sum(sigma > e), 1) for e in eps_grid], dtype=float)
    log_r, log_e = np.log(ranks), np.log(eps_grid)
    return np.abs(np.gradient(log_r, log_e))


def free_subspace_leakage(
    est: RtEstimator, eps: float, queries: np.ndarray
) -> float:
    """Fraction of a held-out query's energy that lies in the 'free' subspace.

    This is the test that matters, and it is not a tautology. Writing into the
    free basis is provably harmless to vectors lying *exactly* in the committed
    span -- that is just linear algebra. But real queries are not confined to
    the committed span; for a heavy-tailed spectrum they carry real mass in the
    tail that eps discarded. That mass is what an adapter allocated to "free"
    rank will disturb.

    Returns mean ||Pi_free q||^2 / ||q||^2 over the held-out queries.
    """
    U_free = est.free_basis(eps)
    if U_free.shape[1] == 0:
        return 0.0
    proj = queries @ U_free
    num = np.sum(proj**2, axis=1)
    den = np.sum(queries**2, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-30)))


def interference_from_free_write(
    est: RtEstimator,
    eps: float,
    queries: np.ndarray,
    readout: np.ndarray,
    rank_request: int,
    rng: np.random.Generator,
    write_norm: float = 0.10,
) -> float:
    """Relative output perturbation on held-out queries from a free-basis write.

    Simulates L7 allocating a rank-`rank_request` adapter inside the free basis
    -- the operation the subspace budget declares safe -- and measures what it
    does to the model's output on traffic it was never supposed to touch.

    The write is normalised to `write_norm` times the readout's Frobenius norm
    so the number means something fixed: "a 10%-magnitude adapter placed
    entirely in free rank moves held-out outputs by X%." Without that
    normalisation the result would just track the arbitrary scale of the write.
    """
    U_free = est.free_basis(eps)
    if U_free.shape[1] == 0:
        return 0.0
    r = min(rank_request, U_free.shape[1])
    basis = U_free[:, :r]
    delta_W = basis @ rng.normal(size=(r, readout.shape[1]))
    delta_W *= write_norm * np.linalg.norm(readout) / np.linalg.norm(delta_W)

    base = queries @ readout
    perturbed = queries @ (readout + delta_W)
    num = np.linalg.norm(perturbed - base, axis=1)
    den = np.linalg.norm(base, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-30)))


def free_basis_by_rank(est: RtEstimator, committed_rank: int) -> np.ndarray:
    """Free basis under the energy/GPM criterion: everything below the top-r directions."""
    _, U = est.eig()
    return U[:, committed_rank:]


def leakage_by_rank(est: RtEstimator, committed_rank: int, queries: np.ndarray) -> float:
    """Fraction of `queries` energy outside the top-`committed_rank` directions.

    Measured on a set the rank was NOT chosen from, this is a real test. Measured
    on the set the rank was chosen from it is circular: picking r to capture 95%
    of a set's energy guarantees 5% leakage on that same set. The gap between the
    two is whether the spectrum estimate generalises.
    """
    U_free = free_basis_by_rank(est, committed_rank)
    if U_free.shape[1] == 0:
        return 0.0
    proj = queries @ U_free
    num = np.sum(proj**2, axis=1)
    den = np.sum(queries**2, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-30)))


def interference_by_rank(
    est: RtEstimator,
    committed_rank: int,
    queries: np.ndarray,
    readout: np.ndarray,
    rank_request: int,
    rng: np.random.Generator,
    write_norm: float = 0.10,
) -> float:
    """Output perturbation from an adapter written into the energy-criterion free basis."""
    U_free = free_basis_by_rank(est, committed_rank)
    if U_free.shape[1] == 0:
        return 0.0
    r = min(rank_request, U_free.shape[1])
    basis = U_free[:, :r]
    delta_W = basis @ rng.normal(size=(r, readout.shape[1]))
    delta_W *= write_norm * np.linalg.norm(readout) / np.linalg.norm(delta_W)

    base = queries @ readout
    perturbed = queries @ (readout + delta_W)
    num = np.linalg.norm(perturbed - base, axis=1)
    den = np.linalg.norm(base, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-30)))


def rank_for_energy(est: RtEstimator, queries: np.ndarray, frac: float) -> int:
    """Smallest r such that the top-r eigendirections hold `frac` of held-out energy.

    This is the parameter-free version of the budget question, and it sidesteps
    eps entirely. The design needs committed rank to be *small* relative to the
    dimension, because free rank is what is left over and free rank is the
    resource adapters spend. If r95 is already most of the dimension, there is
    no budget to allocate no matter where eps is drawn.
    """
    _, U = est.eig()
    proj = queries @ U
    energy = np.sum(proj**2, axis=0)
    total = energy.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(energy) / total
    return int(np.searchsorted(cum, frac) + 1)
