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
rows/columns before calling.

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
        raise ValueError(
            "C must be square; pad with dummy columns first (see associate)"
        )
    if not np.all(np.isfinite(C)):
        raise ValueError("C must be finite; encode gates via dummy costs, not inf")

    n = C.shape[0]
    u = np.zeros(n)
    v = np.zeros(n)
    row_of_col = np.full(n, -1, dtype=np.int64)    # col -> matched row
    match_of_row = np.full(n, -1, dtype=np.int64)  # row -> matched col (inverse)
    for i in range(n):
        tree_rows = {i}
        tree_cols = set()
        minv = C[i] - u[i] - v
        parent = np.full(n, i, dtype=np.int64)     # col -> row that discovered it
        used = np.zeros(n, dtype=bool)             # per phase
        while True:
            j = int(np.argmin(np.where(used, np.inf, minv)))
            used[j] = True
            delta = minv[j]
            if delta != 0:
                # Potentials move by the minimal slack. delta < 0 is possible
                # with negative costs and self-corrects the new row's dual
                # feasibility (its u drops to min_j C[i, j] - v[j]).
                u[list(tree_rows)] += delta
                v[list(tree_cols)] -= delta
                minv[~used] -= delta
            if row_of_col[j] == -1:
                # Unmatched column reached: flip the alternating path.
                jfree = j
                while True:
                    r = parent[jfree]              # row that discovered this column
                    j_old = match_of_row[r]        # its previous column (-1 at the root)
                    match_of_row[r] = jfree
                    row_of_col[jfree] = r
                    if j_old == -1:
                        break
                    jfree = j_old                  # freed column goes to its discoverer
                break
            # Matched column: its row joins the tree; rescan from that row.
            r = row_of_col[j]
            tree_rows.add(r)
            tree_cols.add(j)
            cur = C[r] - u[r] - v
            better = (~used) & (cur < minv)
            minv[better] = cur[better]
            parent[better] = r
    sigma = np.empty(n, dtype=np.int64)
    sigma[row_of_col] = np.arange(n)
    return sigma, u, v


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
