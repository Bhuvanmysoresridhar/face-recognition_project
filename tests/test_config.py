"""Tests for configuration system."""

import os
import tempfile
import pytest
import yaml

from utils.config import Config


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config file."""
    cfg = {
        "recognition": {"threshold": 0.5, "min_face_size": 30},
        "camera": {"index": 2},
        "paths": {
            "known_faces_dir": str(tmp_path / "faces"),
            "encoding_cache": str(tmp_path / "data" / "enc.pkl"),
            "database": str(tmp_path / "data" / "test.db"),
            "attendance_dir": str(tmp_path / "data" / "attendance"),
            "logs_dir": str(tmp_path / "data" / "logs"),
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


def test_default_config():
    """Config with no file uses defaults."""
    cfg = Config("/nonexistent/config.yaml")
    assert cfg.get("recognition", "threshold") == 0.6
    assert cfg.get("camera", "index") == 0


def test_custom_config(tmp_config):
    """Config merges user values over defaults."""
    cfg = Config(tmp_config)
    assert cfg.get("recognition", "threshold") == 0.5
    assert cfg.get("recognition", "min_face_size") == 30
    assert cfg.get("camera", "index") == 2
    # Defaults still present for unset keys
    assert cfg.get("recognition", "model") == "hog"
    assert cfg.get("liveness", "enabled") is True


def test_get_with_default():
    cfg = Config("/nonexistent/config.yaml")
    assert cfg.get("nonexistent", "key", default="fallback") == "fallback"


def test_section():
    cfg = Config("/nonexistent/config.yaml")
    rec = cfg.section("recognition")
    assert isinstance(rec, dict)
    assert "threshold" in rec


def test_data_dirs_created(tmp_config):
    """Config creates data directories on init."""
    cfg = Config(tmp_config)
    assert os.path.isdir(cfg.get("paths", "attendance_dir"))
    assert os.path.isdir(cfg.get("paths", "logs_dir"))
