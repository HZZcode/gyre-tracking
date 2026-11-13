"""Per-frame association and track lifecycle.

The tracking-specific intelligence lives in the cost matrix; the solver is
generic. Minimal recipe (see README for the probabilistic reading):

- prediction:  pred = pos + vel * dt, velocity from the last two positions;
- gate/dummy:  one dummy column per track with cost = gate_sq, so "pair
  beyond the gate" and "matched to dummy" coincide — a single parameter;
- lifecycle:   orphan detection -> provisional track; a track may coast for
  at most `max_coast` consecutive misses, then terminates; tracks shorter
  than `min_length` positions are discarded as clutter.

Deliberately out of scope (README "Extensions"): Kalman filtering, anisotropic
Mahalanobis gating, multi-frame min-cost-flow association.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import certificates, hungarian

FloatArray = NDArray[np.float64]


@dataclass
class Track:
    track_id: int
    positions: list  # history of (x, y), oldest first
    misses: int = 0
    hits: int = 0

    @property
    def pos(self) -> FloatArray:
        return np.asarray(self.positions[-1], dtype=float)

    def velocity(self, dt: float) -> FloatArray:
        """Finite-difference velocity; zero vector if fewer than two fixes."""
        if len(self.positions) < 2:
            return np.zeros(2)
        return (np.asarray(self.positions[-1], dtype=float) - np.asarray(self.positions[-2], dtype=float)) / dt

    def predicted(self, dt: float) -> FloatArray:
        """Constant-velocity prediction for the next frame."""
        return self.pos + self.velocity(dt) * dt


def build_cost_matrix(
    predicted: FloatArray, detections: FloatArray, gate_sq: float
) -> FloatArray:
    """Squared distances plus one dummy column per track.

    Returns an (n, m + n) matrix: columns [0, m) hold the squared distance
    from each predicted position to each detection; columns [m, m + n) are
    dummies with cost = gate_sq. A track matched to a dummy column is
    unmatched this frame.
    """
    raise NotImplementedError


@dataclass
class TrackManager:
    """Owns the live tracks; call `update` once per frame."""

    gate_sq: float
    dt: float
    max_coast: int = 5
    min_hits: int = 3
    min_length: int = 3
    next_id: int = 0
    tracks: list = field(default_factory=list)     # live tracks
    finished: list = field(default_factory=list)   # terminated tracks

    def update(self, detections: FloatArray) -> None:
        """One frame: predict -> build cost matrix -> solve (and verify the
        certificate!) -> apply assignments -> lifecycle rules.

        Steps:
        1. predicted positions for all live tracks;
        2. padded cost matrix (detections + one dummy column per track);
        3. sigma, u, v = hungarian.solve(C);
           certificates.verify_optimal(C, sigma, u, v);
        4. sigma[i] < m: append the detection to track i, reset misses,
           hits += 1;  sigma[i] >= m: coast (misses += 1);
        5. detections not used by any real assignment start new tracks;
        6. tracks with misses > max_coast terminate; terminated tracks with
           fewer than min_length positions are dropped as clutter.
        """
        raise NotImplementedError
