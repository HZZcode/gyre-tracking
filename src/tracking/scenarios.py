"""Scenario ground truths: S1 double-gyre centers, S2 crossing objects.

Both return a (n_objects, K, 2) array of true positions per frame, which
`detector_sim.simulate_tracks` turns into noisy detections + clutter.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .flow import DoubleGyre
from .truth import true_centers

FloatArray = NDArray[np.float64]


def s1_gyre(gyre: DoubleGyre, times: NDArray | list[float]) -> FloatArray:
    """S1: the two closed-form gyre-vortex centers, shape (2, K, 2)."""
    return true_centers(gyre, times)


def s2_crossing(times: NDArray | list[float]) -> FloatArray:
    """S2: two objects on constant-velocity lines that cross mid-domain.

    A: (0.25, 0.25) -> (1.75, 0.75);  B: (1.75, 0.25) -> (0.25, 0.75).
    They cross at (1.0, 0.5) at the midpoint of the time span. Around the
    crossing, all four points (two predictions + two detections) cluster
    within a few sigma, and per-frame association can swap identities —
    the structural failure mode this scenario isolates.
    """
    times = np.asarray(times, dtype=float)
    t0, t1 = times[0], times[-1]
    s = ((times - t0) / (t1 - t0))[:, None]          # (K, 1) in [0, 1]
    a = np.array([0.25, 0.25]) + s * np.array([1.50, 0.50])
    b = np.array([1.75, 0.25]) + s * np.array([-1.50, 0.50])
    return np.stack([a, b], axis=0)                  # (2, K, 2)
