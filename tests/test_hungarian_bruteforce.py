import itertools

import numpy as np
import pytest

from tracking.hungarian import solve


def brute_force_min(C: np.ndarray) -> float:
    n = len(C)
    return min(sum(C[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force(seed):
    """Small random instances: the solver's cost equals the brute-force minimum.

    Note we compare optimal VALUES, never specific matchings: degenerate
    instances have many optimal assignments.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 8))
    C = rng.uniform(0.0, 10.0, size=(n, n))
    sigma, u, v = solve(C)
    assert C[np.arange(n), sigma].sum() == pytest.approx(brute_force_min(C))


def test_degenerate_ties():
    """Many equal costs — the classic stress case for augmenting-path logic."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        C = rng.integers(0, 3, size=(6, 6)).astype(float)
        sigma, u, v = solve(C)
        assert C[np.arange(6), sigma].sum() == pytest.approx(brute_force_min(C))
