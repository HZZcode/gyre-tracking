"""Per-frame association and track lifecycle.

The tracking-specific intelligence lives in the cost matrix and the lifecycle
rules; the solver is generic. Minimal recipe (see README for the probabilistic
reading):

- prediction:  pred = pos + vel * dt, velocity from the last two positions,
  step clipped to |v*dt| <= max_step (poor man's process-noise bound);
- gate/dummy:  padded S x S matrix with base cost = gate_sq, so "pair beyond
  the gate" and "unmatched" coincide — a single parameter;
- confirmation (standard MTT heuristic): only tracks with hits >= min_hits
  compete for detections in the LAP; provisional tracks may only claim
  leftovers, and die on their first miss — clutter never steals detections
  from established tracks;
- birth gating: an unclaimed detection inside some confirmed track's gate is
  NOT promoted to a new track (it is clutter or a duplicate) — only gates
  that no confirmed track could explain birth a track;
- lifecycle:   orphan detection -> provisional track; established tracks may
  coast up to `max_coast` consecutive misses; tracks shorter than
  `min_length` positions are discarded as clutter.

Deliberately out of scope (README "Extensions"): Kalman filtering, anisotropic
Mahalanobis gating, multi-frame min-cost-flow association.
"""

from __future__ import annotations

from collections.abc import Callable
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

    def predicted(self, dt: float, max_step: float = np.inf) -> FloatArray:
        """Constant-velocity prediction, step clipped to |v*dt| <= max_step.

        The finite-difference velocity has std ~ sqrt(2)*sigma/dt (both
        endpoints noisy), so one corrupted match can otherwise fling the
        prediction across the domain. The cap is the poor man's process-noise
        bound: vortex centers advect slowly by physical prior.
        """
        step = self.velocity(dt) * dt
        norm = float(np.hypot(step[0], step[1]))
        if norm > max_step:
            step = step * (max_step / norm)
        return self.pos + step


def build_cost_matrix(
    predicted: FloatArray, detections: FloatArray, gate_sq: float
) -> FloatArray:
    """Pad the (n_tracks, m) distance block into an S x S matrix, S = n + m.

    Four blocks, base fill = gate_sq:
    - rows [0, n) x cols [0, m): squared distances (real pairs);
    - rows [0, n) x cols [m, m+n): dummy columns — track matched here = coast;
    - rows [n, n+m) x cols [0, m): dummy rows — detection matched here =
      unclaimed (will spawn a new track);
    - rows [n, n+m) x cols [m, m+n): filler.

    S = n + m guarantees EVERY track a dummy column and EVERY detection a
    dummy row, so no pair is ever forced: with k real pairs the total cost is
    gate_sq*(n+m) + sum(d2 - gate_sq over pairs), hence the per-pair decision
    threshold is exactly d2 < gate_sq, independent of the track/detection
    counts. (Padding only to max(n, m) would leave frames with m == n fully
    forced — a distant clutter point could then hijack a track.)
    """
    n_tr = len(predicted)
    m = len(detections)
    S = n_tr + m
    C = np.full((S, S), gate_sq)
    C[:n_tr, :m] = ((predicted[:, None, :] - detections[None, :, :]) ** 2).sum(-1)
    return C


@dataclass
class TrackManager:
    """Owns the live tracks; call `update` once per frame.

    Two-pass confirmation heuristic: confirmed tracks (hits >= min_hits)
    compete for detections in a verified LAP; provisional tracks get a second
    LAP over the leftovers only, and die on their first miss. Clutter tracks
    therefore never take detections away from established tracks.
    """

    gate_sq: float
    dt: float
    max_step: float = 0.05     # cap on |prediction step| per frame (domain units)
    max_coast: int = 5
    min_hits: int = 3
    min_length: int = 3
    solver: Callable = hungarian.solve   # (C) -> (sigma, u, v); swap for greedy
    verify: bool = True                  # verify the dual certificate each LAP
    next_id: int = 0
    tracks: list = field(default_factory=list)     # live tracks
    finished: list = field(default_factory=list)   # terminated tracks

    def _match_pass(self, tracks: list, detections: FloatArray, idx: np.ndarray,
                    claimed: np.ndarray, frame_match: np.ndarray) -> np.ndarray:
        """One verified LAP: `tracks` compete for detections[idx].

        `idx` holds the ORIGINAL detection indices this pass competes for
        (pass 2 sees only the leftovers, so subset bookkeeping must map back
        through idx). A track matched to a dummy column coasts (misses += 1);
        a detection matched to a dummy row stays unclaimed. Successful claims
        update the track in place, set claimed[orig] = True and record the
        track id in frame_match[orig]. Returns `explainable`, a mask over idx
        marking detections that were within some track's gate.
        """
        m = len(idx)
        subset = detections[idx]
        predicted = np.array([tr.predicted(self.dt, self.max_step) for tr in tracks])
        C = build_cost_matrix(predicted, subset, self.gate_sq)
        explainable = (C[:len(tracks), :m] <= self.gate_sq).any(axis=0)
        sigma, u, v = self.solver(C)
        if self.verify:
            certificates.verify_optimal(C, sigma, u, v)
        for i, tr in enumerate(tracks):
            j = int(sigma[i])
            if j < m:
                orig = int(idx[j])
                tr.positions.append(subset[j])
                tr.misses = 0
                tr.hits += 1
                claimed[orig] = True
                frame_match[orig] = tr.track_id
            else:
                tr.misses += 1
        return explainable

    def update(self, detections: FloatArray):
        """Advance the tracker by one frame.

        Returns frame_match, an (m,) array with frame_match[j] = track id
        assigned to detection j (-1 if unclaimed).
        """
        m = len(detections)
        frame_match = np.full(m, -1, dtype=np.int64)
        claimed = np.zeros(m, dtype=bool)
        explainable = np.zeros(m, dtype=bool)

        confirmed = [tr for tr in self.tracks if tr.hits >= self.min_hits]
        provisional = [tr for tr in self.tracks if tr.hits < self.min_hits]

        # Pass 1: confirmed tracks compete for all detections.
        if confirmed and m > 0:
            explainable = self._match_pass(confirmed, detections,
                                           np.arange(m), claimed, frame_match)
        else:
            for tr in confirmed:
                tr.misses += 1

        # Pass 2: provisional tracks may only claim leftovers (same LAP).
        free = np.flatnonzero(~claimed)
        if provisional and len(free) > 0:
            self._match_pass(provisional, detections, free, claimed, frame_match)
        else:
            for tr in provisional:
                tr.misses += 1

        # Births: unclaimed detections start new provisional tracks — but
        # only detections that NO confirmed track's gate could explain. A
        # detection inside a confirmed gate that still went unclaimed lost
        # the LAP: it is clutter or a duplicate, not evidence of a new
        # object. Without this birth gate, clutter chains confirm into
        # track doppelgangers that then steal real detections.
        for j in np.flatnonzero(~claimed & ~explainable):
            self.tracks.append(Track(track_id=self.next_id,
                                     positions=[detections[int(j)]],
                                     misses=0, hits=1))
            frame_match[int(j)] = self.next_id
            self.next_id += 1

        # Deaths: provisional tracks die on their FIRST miss; established
        # tracks may coast up to max_coast. Short tracks were clutter.
        alive = []
        for tr in self.tracks:
            dead = (tr.misses > 0 and tr.hits < self.min_hits) or tr.misses > self.max_coast
            if dead:
                if len(tr.positions) >= self.min_length:
                    self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return frame_match
