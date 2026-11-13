import numpy as np
import pytest

from tracking.certificates import verify_optimal
from tracking.hungarian import solve


@pytest.mark.parametrize("seed", range(20))
def test_certificate_on_random_matrices(seed):
    """The dual potentials returned with the matching must certify optimality.

    This is the check that scales: unlike brute force it works at any n.
    """
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 15))
    C = rng.uniform(0.0, 50.0, size=(n, n))
    sigma, u, v = solve(C)
    verify_optimal(C, sigma, u, v)


def test_certificate_integer_matrix():
    C = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    sigma, u, v = solve(C)
    verify_optimal(C, sigma, u, v)


def test_certificate_negative_costs():
    """Negative costs are legal; potentials may then go negative too."""
    rng = np.random.default_rng(3)
    C = rng.normal(0.0, 5.0, size=(8, 8))
    sigma, u, v = solve(C)
    verify_optimal(C, sigma, u, v)
