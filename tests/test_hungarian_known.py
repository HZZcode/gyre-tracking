import numpy as np
import pytest

from tracking.certificates import verify_optimal
from tracking.hungarian import solve

C = np.array(
    [
        [1.0, 2.0, 9.0],
        [2.0, 4.0, 5.0],
        [3.0, 6.0, 8.0],
    ]
)


def test_known_example():
    """Worked example: unique optimum 10 = r1->c2 + r2->c3 + r3->c1.

    Greedy picks 1 + 4 + 8 = 13 and is wrong: it grabs C[0,0] = 1 first.
    The solver must find the global optimum AND a valid dual certificate.
    """
    sigma, u, v = solve(C)
    assert set(enumerate(sigma.tolist())) == {(0, 1), (1, 2), (2, 0)}
    assert C[np.arange(3), sigma].sum() == pytest.approx(10.0)
    verify_optimal(C, sigma, u, v)


def test_rejects_non_square():
    with pytest.raises(ValueError):
        solve(np.zeros((2, 3)))


def test_rejects_non_finite():
    with pytest.raises(ValueError):
        solve(np.array([[0.0, np.inf], [1.0, 2.0]]))
