"""Animations and summary plots. Implement after the pipeline runs."""

from __future__ import annotations


def animate(frames, tracks_by_frame, true_centers, path: str) -> None:
    """Per-frame scatter: detections (dots), active tracks (colored polylines),
    ground-truth centers (crosses). Write mp4/gif to `path`."""
    raise NotImplementedError


def plot_cost_matrix(C, assignments=None, path: str | None = None) -> None:
    """Heatmap of one frame's padded cost matrix; mark the chosen assignment."""
    raise NotImplementedError


def plot_sweep(results, metric: str, path: str) -> None:
    """Line plot of a sweep metric vs noise, one curve per algorithm."""
    raise NotImplementedError
