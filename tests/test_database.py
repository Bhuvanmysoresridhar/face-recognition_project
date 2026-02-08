"""Tests for SQLite database module."""

import os
import pytest
from datetime import datetime

from utils.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    yield d
    d.close()


def test_add_person(db):
    assert db.add_person("alice") is True
    assert db.add_person("alice") is False  # duplicate


def test_get_person(db):
    db.add_person("bob")
    p = db.get_person("bob")
    assert p is not None
    assert p["name"] == "bob"


def test_get_all_persons(db):
    db.add_person("charlie")
    db.add_person("diana")
    persons = db.get_all_persons()
    names = [p["name"] for p in persons]
    assert "charlie" in names
    assert "diana" in names


def test_remove_person(db):
    db.add_person("eve")
    db.remove_person("eve")
    assert db.get_person("eve") is None


def test_update_image_count(db):
    db.add_person("frank")
    db.update_image_count("frank", 5)
    p = db.get_person("frank")
    assert p["image_count"] == 5


def test_log_detection(db):
    db.add_person("grace")
    db.log_detection("grace", 0.85, distance=0.15, camera_index=0)
    dets = db.get_detections(name="grace")
    assert len(dets) == 1
    assert dets[0]["confidence"] == 0.85


def test_detection_stats(db):
    db.add_person("henry")
    db.log_detection("henry", 0.9)
    db.log_detection("henry", 0.8)
    stats = db.get_detection_stats()
    henry = [s for s in stats if s["name"] == "henry"][0]
    assert henry["count"] == 2
    assert abs(henry["avg_confidence"] - 0.85) < 0.01


def test_attendance(db):
    db.add_person("ivy")
    assert db.check_in("ivy") is True
    assert db.check_in("ivy") is False  # already checked in

    records = db.get_attendance()
    assert len(records) == 1
    assert records[0]["name"] == "ivy"
    assert records[0]["check_out"] is None

    db.check_out("ivy")
    records = db.get_attendance()
    assert records[0]["check_out"] is not None


def test_attendance_range(db):
    db.add_person("jack")
    db.check_in("jack")
    today = datetime.now().strftime("%Y-%m-%d")
    records = db.get_attendance_range(today, today)
    assert len(records) >= 1
