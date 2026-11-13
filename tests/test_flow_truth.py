import numpy as np
import pytest

from tracking.flow import DoubleGyre
from tracking.truth import LEFT_FC, RIGHT_FC, center_x, true_centers

GYRE = DoubleGyre()


def test_incompressible():
    """div(u) = du/dx + dv/dy must vanish identically (central differences)."""
    rng = np.random.default_rng(0)
    h = 1e-6
    for t in (0.0, 1.7, 4.2, 7.9):
        x = rng.uniform(0.05, 1.95, 200)
        y = rng.uniform(0.05, 0.95, 200)
        dudx = (GYRE.velocity(x + h, y, t)[0] - GYRE.velocity(x - h, y, t)[0]) / (2 * h)
        dvdy = (GYRE.velocity(x, y + h, t)[1] - GYRE.velocity(x, y - h, t)[1]) / (2 * h)
        assert np.allclose(dudx + dvdy, 0.0, atol=1e-6)


def test_centers_at_t0():
    """t = 0: a = 0, b = 1  =>  centers at exactly x = 0.5 and x = 1.5."""
    assert center_x(GYRE, LEFT_FC, 0.0) == pytest.approx(0.5)
    assert center_x(GYRE, RIGHT_FC, 0.0) == pytest.approx(1.5)


def test_velocity_zero_at_centers():
    """The closed-form centers really are stagnation points of the flow."""
    times = np.linspace(0.0, 10.0, 21)
    centers = true_centers(GYRE, times)
    for k in range(2):
        for i, t in enumerate(times):
            c = centers[k, i]
            u, v = GYRE.velocity(c[0], c[1], float(t))
            assert abs(u) < 1e-9 and abs(v) < 1e-9


def test_centers_stay_in_domain():
    times = np.linspace(0.0, 10.0, 401)  # one full forcing period
    centers = true_centers(GYRE, times)
    xs = centers[..., 0]
    assert np.all((xs >= 0.0) & (xs <= 2.0))
    assert np.allclose(centers[..., 1], 0.5)
