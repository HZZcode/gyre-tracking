"""Fragment stitching: the same verified solver, one level up.

Per-frame association breaks trajectories wherever a detection is missed.
Stitching links the resulting fragments: nodes are fragments; the cost
between fragment i (its last position, extrapolated) and fragment j (its
first position, starting later in time) is the squared extrapolation
distance, gated as usual. One more assignment problem — reuse
`hungarian.solve` and verify the certificate again.
"""

from __future__ import annotations


def stitch(fragments: list, gate_sq: float, dt: float) -> list:
    """Merge track fragments into longer trajectories. TODO"""
    raise NotImplementedError
