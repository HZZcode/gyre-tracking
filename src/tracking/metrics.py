"""Trajectory-quality metrics against synthetic ground truth.

Conventions: for frame k, `gt_source[k][j]` is the true object that produced
detection j (-1 = clutter) and `matches[k][j]` is the track id assigned to
detection j (-1 = unmatched). The mapping between track ids and true objects
is resolved by majority vote (purity) over the whole sequence.
"""

from __future__ import annotations


def fragmentation_counts(gt_source: list, matches: list) -> dict[int, int]:
    """True-object index -> number of DISTINCT track ids its detections were
    matched to (clutter ignored). A perfect tracker returns {obj: 1} for
    every object; larger values mean a trajectory was broken into fragments."""
    raise NotImplementedError


def association_accuracy(gt_source: list, matches: list) -> float:
    """Fraction of real-object detections (gt_source >= 0) matched to the
    track id that owns the majority of that object's detections."""
    raise NotImplementedError


def id_switches(gt_source: list, matches: list) -> int:
    """Number of frames in which an object's assigned track id differs from
    the previous frame's (both frames matched; clutter ignored)."""
    raise NotImplementedError
