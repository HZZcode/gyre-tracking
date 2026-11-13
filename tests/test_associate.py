"""Unit tests for TrackManager: lifecycle, gating, solver swap, and the
no-forced-matching guarantee of the S = n + m padded assignment.

The last one is a regression test: padding only to max(n, m) leaves frames
with m == n_tracks fully forced, so a distant clutter point can hijack a
confirmed track (observed as domain-spanning jumps in the trajectory plot).
"""

import numpy as np
import pytest

from tracking.associate import TrackManager
from tracking.hungarian import greedy_solve

DT = 0.1


def greedy_solver(C):
    """Same (sigma, u, v) triple as hungarian.solve, with fake duals."""
    return greedy_solve(C), np.zeros(len(C)), np.zeros(len(C))


def test_two_objects_get_two_stable_tracks():
    tm = TrackManager(gate_sq=0.1**2, dt=DT)
    for k in range(20):
        pts = np.array([[0.5 + 0.01 * k, 0.5], [1.2 + 0.01 * k, 0.5]])
        fm = tm.update(pts)
        assert list(fm) == [0, 1]          # same ids every frame, both claimed
    assert [tr.hits for tr in tm.tracks] == [20, 20]
    assert all(tr.misses == 0 for tr in tm.tracks)


def test_provisional_track_dies_on_first_miss():
    tm = TrackManager(gate_sq=0.1**2, dt=DT)
    tm.update(np.array([[0.5, 0.5]]))      # hits = 1 < min_hits: provisional
    fm = tm.update(np.zeros((0, 2)))       # empty frame: no crash (regression)
    assert fm.shape == (0,)
    assert len(tm.tracks) == 0             # died unconfirmed, dropped as clutter


def test_confirmed_track_coasts_through_empty_frame():
    tm = TrackManager(gate_sq=0.1**2, dt=DT)
    for _ in range(5):
        tm.update(np.array([[0.5, 0.5]]))
    fm = tm.update(np.zeros((0, 2)))
    assert len(tm.tracks) == 1
    assert tm.tracks[0].misses == 1 and tm.tracks[0].hits == 5


def test_no_forced_matching_at_equal_counts():
    """m == n_tracks must still allow both sides to stay unmatched."""
    tm = TrackManager(gate_sq=0.05**2, dt=DT)
    for _ in range(5):
        tm.update(np.array([[0.5, 0.5]]))  # confirmed track at (0.5, 0.5)
    # single detection == single track, but it sits at distance 0.4 >> gate
    fm = tm.update(np.array([[0.9, 0.5]]))
    assert fm[0] != 0                      # NOT claimed by the far track
    assert tm.tracks[0].misses == 1        # the track coasted instead
    assert len(tm.tracks) == 2             # the far detection started its own


def test_far_detection_never_joins_across_the_domain():
    """Even with zero alternatives, a pair beyond the gate is not matched."""
    tm = TrackManager(gate_sq=0.05**2, dt=DT)
    for _ in range(5):
        tm.update(np.array([[0.5, 0.5]]))
    before = tm.tracks[0].pos.copy()
    for k in range(10):
        fm = tm.update(np.array([[0.5, 0.9 + 0.01 * k]]))   # far below
        assert fm[0] != 0
        live0 = [tr for tr in tm.tracks if tr.track_id == 0]
        if live0:
            assert np.allclose(live0[0].pos, before)         # coasting: frozen
    assert all(tr.track_id != 0 for tr in tm.tracks)          # coasted out
    assert any(tr.track_id == 0 for tr in tm.finished)        # archived


def test_in_gate_unclaimed_detection_does_not_spawn_a_track():
    """Birth gating: an unclaimed detection that a confirmed track's gate
    already explains is clutter — it must not become a new track."""
    tm = TrackManager(gate_sq=0.05**2, dt=DT)
    for _ in range(5):
        tm.update(np.array([[0.5, 0.5]]))
    # second detection is in-gate (d2 = 0.0016 < 0.0025) but loses the LAP
    fm = tm.update(np.array([[0.5, 0.5], [0.54, 0.5]]))
    assert list(fm[:1]) == [0]             # own detection claimed as usual
    assert fm[1] == -1                     # clutter unclaimed...
    assert len(tm.tracks) == 1             # ...and NOT promoted to a track


def test_two_pass_bookkeeping_uses_original_indices():
    """Regression: pass 2 (provisional, leftover-only) used subset-relative
    indices, corrupting claimed/frame_match — spurious births and identity
    churn in every real run. Claims must record ORIGINAL detection indices.
    """
    tm = TrackManager(gate_sq=0.1**2, dt=DT)
    for _ in range(3):
        tm.update(np.array([[0.5, 0.5]]))          # track 0 confirmed
    fm = tm.update(np.array([[1.5, 0.5]]))         # object 1 appears -> id 1
    assert list(fm) == [1]
    fm = tm.update(np.array([[0.5, 0.5], [1.5, 0.5]]))   # mixed frame:
    # pass 1: confirmed track 0 claims det 0; pass 2: provisional track 1
    # claims det 1 — each under its ORIGINAL index.
    assert list(fm) == [0, 1]
    assert len(tm.tracks) == 2                     # no spurious birth


def test_greedy_solver_option_runs_and_tracks():
    tm = TrackManager(gate_sq=0.1**2, dt=DT, solver=greedy_solver, verify=False)
    for k in range(20):
        pts = np.array([[0.5 + 0.01 * k, 0.5], [1.2 + 0.01 * k, 0.5]])
        tm.update(pts)
    confirmed = [tr for tr in tm.tracks if tr.hits >= tm.min_hits]
    assert len(confirmed) == 2
