"""Optimality certificates for the linear assignment problem.

Given a claimed optimal assignment `sigma` with dual potentials (u, v),
three checks PROVE optimality via LP strong duality:

1. dual feasibility        u[i] + v[j] <= C[i, j]              (all i, j)
2. complementary slackness u[i] + v[sigma[i]] == C[i, sigma[i]]
3. strong duality          sum(u) + sum(v) == cost(sigma)

(1) alone implies every assignment costs >= sum(u) + sum(v); (2) + (3) say
sigma attains that bound. Verification costs O(n^2) — cheaper than solving,
and valid at any scale (brute force is only for small-instance testing).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def check_dual_feasibility(
    C: FloatArray, u: FloatArray, v: FloatArray, atol: float = 1e-9
) -> bool:
    """u[i] + v[j] <= C[i, j] for every pair (i, j)."""
    return bool(np.all(u[:, None] + v[None, :] <= C + atol))


def check_complementary_slackness(
    C: FloatArray, sigma: IntArray, u: FloatArray, v: FloatArray, atol: float = 1e-9
) -> bool:
    """Matched pairs are tight: u[i] + v[sigma[i]] == C[i, sigma[i]]."""
    idx = np.arange(len(sigma))
    return bool(np.all(np.abs(u + v[sigma] - C[idx, sigma]) <= atol))


def check_strong_duality(
    C: FloatArray, sigma: IntArray, u: FloatArray, v: FloatArray, atol: float = 1e-9
) -> bool:
    """The dual objective equals the primal cost."""
    cost = float(C[np.arange(len(sigma)), sigma].sum())
    return abs(float(u.sum() + v.sum()) - cost) <= atol


def verify_optimal(
    C: FloatArray, sigma: IntArray, u: FloatArray, v: FloatArray, atol: float = 1e-9
) -> None:
    """Assert all three conditions; raise AssertionError listing the failures."""
    failures: list[str] = []
    if not check_dual_feasibility(C, u, v, atol):
        violations = u[:, None] + v[None, :] - C
        failures.append(f"dual feasibility violated at {int((violations > atol).sum())} pair(s)")
    if not check_complementary_slackness(C, sigma, u, v, atol):
        failures.append("complementary slackness violated on matched pairs")
    if not check_strong_duality(C, sigma, u, v, atol):
        failures.append("strong duality: dual objective != primal cost")
    if failures:
        raise AssertionError("; ".join(failures))
