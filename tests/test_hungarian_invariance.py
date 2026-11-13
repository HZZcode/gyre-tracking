import numpy as np
import pytest

from tracking.certificates import verify_optimal
from tracking.hungarian import solve


def test_permutation_invariance():
    """Permuting rows and columns (consistently) must not change the value."""
    rng = np.random.default_rng(42)
    C = rng.uniform(0.0, 10.0, size=(7, 7))
    sigma, u, v = solve(C)
    base_cost = C[np.arange(7), sigma].sum()

    perm = rng.permutation(7)
    C2 = C[np.ix_(perm, perm)]
    sigma2, u2, v2 = solve(C2)
    assert C2[np.arange(7), sigma2].sum() == pytest.approx(base_cost)
    verify_optimal(C2, sigma2, u2, v2)


def test_scaled_costs():
    """Scaling all costs by a constant scales the optimal value exactly."""
    rng = np.random.default_rng(5)
    C = rng.uniform(0.0, 10.0, size=(6, 6))
    s1, u1, v1 = solve(C)
    C3 = 3.0 * C
    s2, u2, v2 = solve(C3)
    assert C3[np.arange(6), s2].sum() == pytest.approx(3.0 * C[np.arange(6), s1].sum())
    verify_optimal(C3, s2, u2, v2)
