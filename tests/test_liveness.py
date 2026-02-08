"""Tests for liveness detection module."""

import numpy as np
import pytest
from utils.config import Config
from recognition.liveness import LivenessDetector


@pytest.fixture
def detector():
    config = Config("/nonexistent")
    return LivenessDetector(config)


def test_disabled_liveness():
    config = Config("/nonexistent")
    det = LivenessDetector(config)
    det.enabled = False
    # Create a dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = det.check_liveness(frame, (100, 300, 300, 100))
    assert result["is_live"] is True
    assert result["confidence"] == 1.0


def test_texture_score(detector):
    # High-texture image (real-looking)
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    score = detector._texture_score(frame)
    assert score > 0

    # Flat image (spoof-like)
    flat = np.ones((200, 200, 3), dtype=np.uint8) * 128
    flat_score = detector._texture_score(flat)
    assert flat_score < score


def test_color_analysis(detector):
    # Varied color image
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    score = detector._color_analysis(frame)
    assert score > 0


def test_eye_aspect_ratio(detector):
    # Simulated open eye (wider)
    open_eye = [(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)]
    ear = detector._eye_aspect_ratio(open_eye)
    assert ear > 0

    # Simulated closed eye (very narrow)
    closed_eye = [(0, 0), (1, 0.1), (2, 0.1), (3, 0), (2, -0.1), (1, -0.1)]
    ear_closed = detector._eye_aspect_ratio(closed_eye)
    assert ear_closed < ear


def test_reset(detector):
    detector._blink_counters["test"] = 5
    detector._ear_history["test"] = [0.3, 0.2, 0.3]
    detector.reset("test")
    assert "test" not in detector._blink_counters

    detector._blink_counters["a"] = 1
    detector._blink_counters["b"] = 2
    detector.reset()
    assert len(detector._blink_counters) == 0
