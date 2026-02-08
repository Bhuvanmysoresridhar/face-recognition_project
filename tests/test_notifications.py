"""Tests for notification system."""

import pytest
from utils.config import Config
from utils.notifications import NotificationManager


@pytest.fixture
def notifier():
    config = Config("/nonexistent")
    nm = NotificationManager(config)
    return nm


def test_disabled_by_default(notifier):
    assert notifier.enabled is False


def test_unconfigured_email(notifier):
    notifier.enabled = True
    result = notifier.send_email("test", "body")
    assert result is False  # no credentials configured


def test_cooldown(notifier):
    notifier.enabled = True
    notifier._configured = False  # prevent actual sends
    # Simulate a sent alert
    from datetime import datetime
    notifier._last_alert["unknown_face_cam0"] = datetime.now()
    assert notifier._can_send("unknown_face_cam0") is False
    assert notifier._can_send("unknown_face_cam1") is True


def test_alert_unknown_when_disabled(notifier):
    assert notifier.alert_unknown_face() is False
