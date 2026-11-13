import numpy as np
import pytest

from tracking.metrics import association_accuracy, fragmentation_counts, id_switches

# Hand-built micro-scenario, 3 frames, 2 true objects:
# - object 0 detected every frame, matched to track 7 twice then track 9;
# - object 1 detected in frames 0-1 only (missed in frame 2), track 8;
# - frame 2 also contains one clutter detection (source -1, unmatched).
GT = [
    np.array([0, 1]),
    np.array([0, 1]),
    np.array([0, -1]),
]
MATCHES = [
    np.array([7, 8]),
    np.array([7, 8]),
    np.array([9, -1]),
]


def test_fragmentation_counts():
    # object 0 was split across tracks {7, 9}; object 1 stayed on track 8.
    assert fragmentation_counts(GT, MATCHES) == {0: 2, 1: 1}


def test_association_accuracy():
    # object 0: 2 of 3 detections on its majority track; object 1: 1 of 1.
    assert association_accuracy(GT, MATCHES) == pytest.approx(3 / 4)


def test_id_switches():
    # object 0 switches track 7 -> 9 between frames 1 and 2; object 1 never.
    assert id_switches(GT, MATCHES) == 1
