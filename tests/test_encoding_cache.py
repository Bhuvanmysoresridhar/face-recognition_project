"""Tests for encoding cache."""

import os
import numpy as np
import pytest

from utils.encoding_cache import EncodingCache


@pytest.fixture
def cache(tmp_path):
    return EncodingCache(str(tmp_path / "enc.pkl"))


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal test image file."""
    path = tmp_path / "test.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return str(path)


def test_empty_cache(cache):
    assert cache.get_all_names() == []


def test_store_and_retrieve(cache, sample_image):
    encodings = [np.random.rand(128)]
    cache.store_encodings("test_person", [sample_image], encodings)

    cached, needs_update = cache.get_encodings("test_person", [sample_image])
    assert not needs_update
    assert len(cached) == 1
    np.testing.assert_array_almost_equal(cached[0], encodings[0])


def test_cache_invalidation(cache, tmp_path):
    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"\xff\xd8" + b"\x00" * 50)

    encodings = [np.random.rand(128)]
    cache.store_encodings("person", [str(img1)], encodings)

    # Modify file
    img1.write_bytes(b"\xff\xd8" + b"\x01" * 50)

    cached, needs_update = cache.get_encodings("person", [str(img1)])
    assert needs_update


def test_remove_person(cache, sample_image):
    cache.store_encodings("removable", [sample_image], [np.random.rand(128)])
    assert "removable" in cache.get_all_names()

    cache.remove_person("removable")
    assert "removable" not in cache.get_all_names()


def test_clear(cache, sample_image):
    cache.store_encodings("a", [sample_image], [np.random.rand(128)])
    cache.clear()
    assert cache.get_all_names() == []


def test_persistence(tmp_path, sample_image):
    path = str(tmp_path / "persist.pkl")
    c1 = EncodingCache(path)
    c1.store_encodings("saved", [sample_image], [np.random.rand(128)])

    c2 = EncodingCache(path)
    assert "saved" in c2.get_all_names()
