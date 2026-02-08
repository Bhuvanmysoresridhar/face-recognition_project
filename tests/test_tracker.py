"""Tests for face tracker."""

import pytest
from utils.config import Config
from recognition.tracker import FaceTracker


@pytest.fixture
def tracker():
    config = Config("/nonexistent")
    return FaceTracker(config)


def test_register_new_faces(tracker):
    detections = [(10, 110, 110, 10), (200, 300, 300, 200)]
    names = ["alice", "bob"]
    confs = [0.9, 0.8]
    result = tracker.update(detections, names, confs)
    assert len(result) == 2


def test_track_across_frames(tracker):
    # Frame 1: detect face at position
    det1 = [(100, 200, 200, 100)]
    tracker.update(det1, ["alice"], [0.9])

    # Frame 2: face moved slightly
    det2 = [(105, 205, 205, 105)]
    result = tracker.update(det2, ["alice"], [0.9])

    # Should still be 1 tracked object (same face)
    assert len(result) == 1


def test_disappearing_face(tracker):
    tracker.max_disappeared = 2

    det = [(100, 200, 200, 100)]
    tracker.update(det, ["alice"], [0.9])

    # Face disappears for 3 frames
    for _ in range(3):
        result = tracker.update([], [], [])

    # Should be deregistered
    assert len(result) == 0


def test_new_face_registration(tracker):
    det1 = [(100, 200, 200, 100)]
    tracker.update(det1, ["alice"], [0.9])

    # New face far away
    det2 = [(100, 200, 200, 100), (500, 600, 600, 500)]
    result = tracker.update(det2, ["alice", "bob"], [0.9, 0.8])
    assert len(result) == 2


def test_reset(tracker):
    tracker.update([(10, 110, 110, 10)], ["a"], [0.9])
    tracker.reset()
    assert len(tracker.objects) == 0


def test_disabled_tracker():
    config = Config("/nonexistent")
    tracker = FaceTracker(config)
    tracker.enabled = False

    det = [(10, 110, 110, 10), (200, 300, 300, 200)]
    result = tracker.update(det, ["a", "b"], [0.9, 0.8])
    assert len(result) == 2
