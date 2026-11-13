"""Synthetic detector: turns ground-truth centers into per-frame detections.

The association stage is deliberately isolated from detection: each true
center is found independently with probability `p_detect`, perturbed by
isotropic Gaussian noise with standard deviation `sigma`, and every frame
additionally contains Poisson(`clutter_rate`) uniformly distributed false
positives. The simulator also records which true object (or clutter) each
detection came from — that is what the metrics are scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .flow import DoubleGyre
from .truth import true_centers

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DOMAIN = (0.0, 2.0, 0.0, 1.0)  # x0, x1, y0, y1


@dataclass(frozen=True)
class FrameDetections:
    points: FloatArray  # (m_k, 2) detection positions
    source: IntArray    # (m_k,) true-object index, -1 = clutter


def simulate(
    gyre: DoubleGyre,
    times: NDArray | list[float],
    sigma: float,
    p_detect: float,
    clutter_rate: float,
    rng: np.random.Generator,
) -> list[FrameDetections]:
    """One detection sequence over `times` (K frames)."""
    times = np.asarray(times, dtype=float)
    centers = true_centers(gyre, times)  # (n_objects, K, 2)
    frames: list[FrameDetections] = []
    for k in range(len(times)):
        pts: list[NDArray] = []
        src: list[int] = []
        for obj in range(centers.shape[0]):
            c = centers[obj, k]
            if np.isnan(c[0]):
                continue
            if rng.random() < p_detect:
                pts.append(c + rng.normal(0.0, sigma, size=2))
                src.append(obj)
        for _ in range(int(rng.poisson(clutter_rate))):
            pts.append(rng.uniform((DOMAIN[0], DOMAIN[2]), (DOMAIN[1], DOMAIN[3])))
            src.append(-1)
        frames.append(
            FrameDetections(
                points=np.array(pts, dtype=float).reshape(-1, 2),
                source=np.array(src, dtype=np.int64),
            )
        )
    return frames
