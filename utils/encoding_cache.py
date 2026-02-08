"""
Persistent face encoding cache - avoids recomputing encodings on every startup.
"""

import os
import pickle
import hashlib
import numpy as np


class EncodingCache:
    """Caches face encodings to disk so they persist across restarts."""

    def __init__(self, cache_path="data/encodings.pkl"):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.cache_path = cache_path
        self._cache = self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                return {}
        return {}

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    @staticmethod
    def _file_hash(filepath):
        """SHA256 hash of file contents for change detection."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_encodings(self, name, image_paths):
        """
        Get cached encodings for a person.
        Returns (encodings_list, needs_update) where needs_update is True
        if any images changed or are new.
        """
        hashes = {p: self._file_hash(p) for p in image_paths}
        cached = self._cache.get(name)

        if cached and cached.get("hashes") == hashes:
            return cached["encodings"], False

        return None, True

    def store_encodings(self, name, image_paths, encodings):
        """Store encodings with file hashes for invalidation."""
        hashes = {p: self._file_hash(p) for p in image_paths}
        self._cache[name] = {
            "hashes": hashes,
            "encodings": encodings,
        }
        self.save()

    def remove_person(self, name):
        if name in self._cache:
            del self._cache[name]
            self.save()

    def get_all_names(self):
        return list(self._cache.keys())

    def clear(self):
        self._cache = {}
        self.save()
