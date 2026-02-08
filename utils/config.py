"""
Configuration loader - reads config.yaml and provides typed access.
"""

import os
import yaml


class Config:
    """Loads and provides access to configuration values."""

    _defaults = {
        "recognition": {
            "threshold": 0.6,
            "min_face_size": 50,
            "model": "hog",
            "frame_scale": 0.25,
            "skip_frames": 2,
        },
        "camera": {"index": 0, "width": 640, "height": 480},
        "paths": {
            "known_faces_dir": "known_faces",
            "encoding_cache": "data/encodings.pkl",
            "database": "data/face_recognition.db",
            "attendance_dir": "data/attendance",
            "logs_dir": "data/logs",
        },
        "liveness": {
            "enabled": True,
            "blink_threshold": 0.25,
            "texture_threshold": 80.0,
            "min_blinks": 1,
            "check_interval": 30,
        },
        "attendance": {
            "enabled": True,
            "cooldown_minutes": 30,
            "export_format": "csv",
            "auto_export": True,
        },
        "notifications": {
            "enabled": False,
            "unknown_face_alert": True,
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "",
                "password": "",
                "recipients": [],
            },
            "cooldown_minutes": 5,
        },
        "web": {
            "enabled": False,
            "host": "0.0.0.0",
            "port": 5000,
            "secret_key": "change-this-to-a-random-secret-key",
            "max_cameras": 4,
        },
        "tracker": {
            "enabled": True,
            "max_disappeared": 15,
            "max_distance": 75,
        },
    }

    def __init__(self, config_path="config.yaml"):
        self._data = self._deep_copy(self._defaults)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_cfg = yaml.safe_load(f) or {}
            self._deep_merge(self._data, user_cfg)
        # Ensure data directories exist
        for key in ("encoding_cache", "database", "attendance_dir", "logs_dir"):
            path = self._data["paths"][key]
            dir_path = os.path.dirname(path) if "." in os.path.basename(path) else path
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    # ---- public helpers ----

    def get(self, *keys, default=None):
        """Dot-path access: config.get('recognition', 'threshold')"""
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def section(self, name):
        return self._data.get(name, {})

    @property
    def data(self):
        return self._data

    # ---- internal ----

    @staticmethod
    def _deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = v

    @staticmethod
    def _deep_copy(d):
        import copy
        return copy.deepcopy(d)
