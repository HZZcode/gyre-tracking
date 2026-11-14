"""Fragment stitching: the same verified solver, one level up.

Per-frame association breaks trajectories wherever a detection is missed.
Stitching links the resulting fragments: nodes are fragments; the cost
between fragment i (its last position, extrapolated) and fragment j (its
first position, starting later in time) is the squared extrapolation
distance, gated as usual. One more assignment problem — reuse
`hungarian.solve` and verify the certificate again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import certificates, hungarian

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrackSpan:
    first_frame: int
    last_frame: int
    first_pos: FloatArray
    last_pos: FloatArray
    last_vel: FloatArray


def stitch_plan(info: dict[int, TrackSpan], gate_sq: float, dt: float,
                max_gap: int = 3) -> dict[int, int]:
    tids = sorted(info)

    n = len(tids)
    C = np.full((n, n), gate_sq)
    for i, f in enumerate(tids):
        sf = info[f]
        for j, y in enumerate(tids):
            gap = info[y].first_frame - sf.last_frame
            if y == f or gap <= 0 or gap > max_gap:
                continue
            pred = sf.last_pos + sf.last_vel * dt * gap
            d2 = float(((info[y].first_pos - pred) ** 2).sum())
            C[i, j] = d2

    sigma, u, v = hungarian.solve(C)
    certificates.verify_optimal(C, sigma, u, v)

    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in enumerate(sigma):
        if j < n:
            union(i, j)

    # Total map: every tid -> its component root (the earliest id in the
    # chain, since union keeps the min). Roots map to themselves.
    return {tids[a]: tids[find(a)] for a in range(n)}
