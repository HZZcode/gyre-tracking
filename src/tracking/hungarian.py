"""From-scratch linear assignment solver — THE piece to implement.

Contract
--------
`solve(C)` takes a finite (n, n) float cost matrix and returns (sigma, u, v):

- sigma : (n,) int64 array; sigma[i] = column assigned to row i;
- u, v  : (n,) float64 arrays of dual potentials satisfying

    dual feasibility        u[i] + v[j] <= C[i, j]                    (all i, j)
    complementary slackness u[i] + v[sigma[i]] == C[i, sigma[i]]
    strong duality          sum(u) + sum(v) == sum_i C[i, sigma[i]]

within float tolerance. Together the three conditions PROVE that sigma is
optimal; `certificates.verify_optimal` checks them, and the brute-force test
suite compares optimal values on small instances. Rectangular inputs are NOT
handled here: `associate.build_cost_matrix` pads to square with dummy
columns before calling.

Algorithm: shortest augmenting path with dual potentials (JV variant; the
same scheme scipy uses internally). One phase per row; in each phase an
alternating tree of tight edges grows from the new row, potentials move by
the minimal slack delta until an unmatched column becomes reachable, and the
matching is augmented along the alternating path. O(n^3) overall.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def solve(C: NDArray) -> tuple[IntArray, FloatArray, FloatArray]:
    """Solve the square assignment problem; see the module docstring."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square; pad with dummy columns first (see associate)")
    if not np.all(np.isfinite(C)):
        raise ValueError("C must be finite; encode gates via dummy costs, not inf")

    n = C.shape[0]
    # Implementation sketch (shortest augmenting path with potentials):
    #
    #   u = zeros(n); v = zeros(n)          # dual potentials
    #   row_of_col = full(n, -1)            # row_of_col[j] = row matched to column j
    #   for i in range(n):                  # phase i: match row i
    #       tree_rows = {i}; tree_cols = set()
    #       minv = C[i] - u[i] - v          # slack from the newest tree row
    #       parent = full(n, -1)            # for path reconstruction
    #       repeat:
    #           j* = argmin over not-yet-used columns of minv
    #           if minv[j*] > 0:            # no tight column reachable yet:
    #               delta = minv[j*]        #   move potentials by the minimal slack
    #               u[tree_rows] += delta; v[tree_cols] -= delta; minv[unused] -= delta
    #           mark j* used
    #           if row_of_col[j*] == -1:    # unmatched column: augment
    #               flip matching along parent pointers; break
    #           else:                       # matched column: its row joins the tree
    #               tree_rows.add(row_of_col[j*]); tree_cols.add(j*)
    #               minv = minimum(minv, C[row_of_col[j*]] - u[row_of_col[j*]] - v)
    #   ...
    #
    # On return, all three certificate conditions in certificates.py must hold.
    raise NotImplementedError("implement the JV shortest-augmenting-path solver")


def greedy_solve(C: NDArray) -> IntArray:
    """Baseline matcher for the experiments: repeatedly take the globally
    cheapest unassigned (row, column) pair. Not optimal — the point of the
    sweep is to quantify how badly this fails as noise grows."""
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    sigma = -np.ones(n, dtype=np.int64)
    rows_used: set[int] = set()
    cols_used: set[int] = set()
    for flat in np.argsort(C, axis=None):
        i, j = divmod(int(flat), m)
        if i in rows_used or j in cols_used:
            continue
        rows_used.add(i)
        cols_used.add(j)
        sigma[i] = j
        if len(rows_used) == min(n, m):
            break
    return sigma
