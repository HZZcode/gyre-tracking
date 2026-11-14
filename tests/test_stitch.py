"""Stitching spec, as hand-built scenarios.

stitch_plan takes per-tid TrackSpans (first/last frame, first/last claimed
position, last constant-velocity estimate in units per TIME) and returns a
total remap old_tid -> canonical_tid, canonical = earliest id in the chain.
"""

import numpy as np

from tracking.stitch import TrackSpan, stitch_plan

DT = 0.1
GATE_SQ = 0.1 ** 2


def mk(first_frame, last_frame, first_pos, last_pos, vel):
    """TrackSpan constructor with numpy coercion; vel in units per time."""
    return TrackSpan(first_frame, last_frame,
                     np.asarray(first_pos, dtype=float),
                     np.asarray(last_pos, dtype=float),
                     np.asarray(vel, dtype=float))


def test_basic_stitch_links_successor_to_earlier_id():
    # Object moves +0.01 / frame (vel 0.1 per unit time). Track 7 claims
    # frames 0-4, track 8 claims frames 5-7 continuing the same motion.
    v = [0.1, 0.0]
    info = {7: mk(0, 4, [0.50, 0.5], [0.54, 0.5], v),
            8: mk(5, 7, [0.55, 0.5], [0.57, 0.5], v)}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[8] == id_map[7] == 7


def test_gap_longer_than_max_gap_never_links():
    v = [0.1, 0.0]
    info = {7: mk(0, 4, [0.50, 0.5], [0.54, 0.5], v),
            8: mk(15, 17, [0.65, 0.5], [0.67, 0.5], v)}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[8] != id_map[7]


def test_direction_mismatch_never_links():
    # Track 7 ends heading +x; track 8 starts 0.2 LATERALLY off the
    # extrapolation — outside the gate even though the frame gap is 1.
    info = {7: mk(0, 4, [0.50, 0.5], [0.54, 0.5], [0.1, 0.0]),
            8: mk(5, 7, [0.55, 0.7], [0.55, 0.9], [0.0, 0.1])}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[8] != id_map[7]


def test_chain_of_three_fragments_merges_to_earliest():
    v = [0.1, 0.0]
    info = {1: mk(0, 2, [0.50, 0.5], [0.52, 0.5], v),
            2: mk(3, 5, [0.53, 0.5], [0.55, 0.5], v),
            3: mk(6, 8, [0.56, 0.5], [0.58, 0.5], v)}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[1] == id_map[2] == id_map[3] == 1


def test_competing_enders_lap_picks_the_closer_prediction():
    # Two dead stubs both predict into the same successor's start; the
    # certified LAP must award the link to the closer one (stub 2) and
    # leave stub 1 as its own trajectory.
    info = {1: mk(0, 4, [0.30, 0.5], [0.50, 0.5], [0.0, 0.0]),
            2: mk(0, 4, [0.38, 0.5], [0.58, 0.5], [0.0, 0.0]),
            3: mk(5, 7, [0.55, 0.5], [0.57, 0.5], [0.0, 0.0])}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[3] == id_map[2]      # closer stub wins the link
    assert id_map[1] == 1              # loser stays separate


def test_single_track_maps_to_itself():
    info = {5: mk(0, 9, [0.5, 0.5], [0.59, 0.5], [0.1, 0.0])}
    id_map = stitch_plan(info, GATE_SQ, DT)
    assert id_map[5] == 5
