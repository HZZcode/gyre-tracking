"""Closed-form ground-truth vortex-center trajectories.

Interior stagnation points of the double gyre satisfy

    cos(pi * y) = 0   =>  y = 1/2
    cos(pi * f) = 0   =>  f(x, t) = k + 1/2

for the k-th gyre cell (k = 0: left cell 0 < f < 1; k = 1: right cell
1 < f < 2). Solving a(t) x^2 + b(t) x = f_c gives

    x(t) = (-b(t) +/- sqrt(b(t)^2 + 4 a(t) f_c)) / (2 a(t)),   a(t) != 0
    x(t) = f_c / b(t),                                         a(t) = 0

For the default parameter regime exactly one root lies in the domain [0, 2]
for each f_c, so the center trajectories are exact closed formulas: every
tracking error is measurable against them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .flow import DoubleGyre

FloatArray = NDArray[np.float64]

LEFT_FC = 0.5   # f-value of the left gyre center
RIGHT_FC = 1.5  # f-value of the right gyre center


def center_x(gyre: DoubleGyre, f_c: float, t: float) -> float:
    """x-coordinate of the gyre center with f = f_c at time t (NaN if none)."""
    a, b = gyre.a(t), gyre.b(t)
    if abs(a) < 1e-12:
        roots = [f_c / b]
    else:
        disc = b * b + 4.0 * a * f_c
        if disc < 0.0:
            return float("nan")
        sq = float(np.sqrt(disc))
        roots = [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]
    inside = [r for r in roots if 0.0 <= r <= 2.0]
    if len(inside) != 1:
        return float("nan")
    return float(inside[0])


def true_centers(
    gyre: DoubleGyre,
    times: NDArray | list[float],
    f_c_values: tuple[float, ...] = (LEFT_FC, RIGHT_FC),
) -> FloatArray:
    """Centers per object and frame: shape (len(f_c_values), K, 2).

    Column 0 is x, column 1 is y (= 1/2 always). NaN marks a center that
    does not exist at that time (does not happen for the default regime).
    """
    times = np.asarray(times, dtype=float)
    out = np.full((len(f_c_values), len(times), 2), np.nan)
    for k, f_c in enumerate(f_c_values):
        for i, t in enumerate(times):
            out[k, i, 0] = center_x(gyre, f_c, float(t))
            out[k, i, 1] = 0.5
    return out
