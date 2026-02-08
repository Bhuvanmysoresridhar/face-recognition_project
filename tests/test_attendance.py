"""Tests for attendance system."""

import os
import pytest
from datetime import datetime

from utils.config import Config
from utils.database import Database
from utils.attendance import AttendanceManager


@pytest.fixture
def attendance(tmp_path):
    cfg_data = {
        "attendance": {
            "enabled": True,
            "cooldown_minutes": 30,
            "export_format": "csv",
            "auto_export": False,
        },
        "paths": {
            "attendance_dir": str(tmp_path / "attendance"),
            "known_faces_dir": str(tmp_path / "faces"),
            "encoding_cache": str(tmp_path / "data" / "enc.pkl"),
            "database": str(tmp_path / "data" / "test.db"),
            "logs_dir": str(tmp_path / "data" / "logs"),
        },
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_data, f)

    config = Config(str(cfg_path))
    db = Database(str(tmp_path / "test.db"))
    db.add_person("alice")
    att = AttendanceManager(config, db)
    yield att
    db.close()


def test_mark_attendance(attendance):
    assert attendance.mark_attendance("alice", 0.9) is True


def test_cooldown(attendance):
    attendance.mark_attendance("alice", 0.9)
    # Second call within cooldown should be rejected
    assert attendance.mark_attendance("alice", 0.9) is False


def test_disabled(attendance):
    attendance.enabled = False
    assert attendance.mark_attendance("alice", 0.9) is False


def test_export_csv(attendance):
    attendance.mark_attendance("alice", 0.9)
    path = attendance.export_attendance()
    assert path is not None
    assert path.endswith(".csv")
    assert os.path.exists(path)


def test_get_today(attendance):
    attendance.mark_attendance("alice", 0.9)
    records = attendance.get_today_attendance()
    assert len(records) >= 1
