"""Tests for ConditioningCache — LRU cache keyed by (path, mtime)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestConditioningCache:
    @pytest.fixture(autouse=True)
    def setup(self):
        from chatterstream.conditioning_cache import ConditioningCache
        self.mock_model = MagicMock()
        self.mock_model.prepare_conditionals = MagicMock(return_value="conditionals_result")
        self.cache = ConditioningCache(self.mock_model, maxsize=3)

    def test_cache_miss_calls_extraction(self, tmp_path):
        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"fake audio data")
        result = self.cache.get(str(voice_file))
        assert result == "conditionals_result"
        self.mock_model.prepare_conditionals.assert_called_once_with(str(voice_file))

    def test_cache_hit_skips_extraction(self, tmp_path):
        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"fake audio data")
        path = str(voice_file)
        self.cache.get(path)
        self.cache.get(path)
        # Should only call extraction once
        assert self.mock_model.prepare_conditionals.call_count == 1

    def test_mtime_change_invalidates(self, tmp_path):
        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"original data")
        path = str(voice_file)
        self.cache.get(path)

        # Modify file to change mtime
        voice_file.write_bytes(b"modified data")
        # Force a different mtime
        os.utime(path, (os.path.getatime(path), os.path.getmtime(path) + 10))

        self.cache.get(path)
        assert self.mock_model.prepare_conditionals.call_count == 2

    def test_lru_eviction(self, tmp_path):
        # Cache maxsize=3, add 4 entries
        paths = []
        for i in range(4):
            f = tmp_path / f"voice_{i}.wav"
            f.write_bytes(f"data_{i}".encode())
            paths.append(str(f))
            self.cache.get(paths[-1])

        # First entry should have been evicted
        self.mock_model.prepare_conditionals.reset_mock()
        self.cache.get(paths[0])  # Should miss (evicted)
        self.mock_model.prepare_conditionals.assert_called_once()

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            self.cache.get("/nonexistent/path/voice.wav")

    def test_clear_empties_cache(self, tmp_path):
        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"data")
        path = str(voice_file)
        self.cache.get(path)
        self.cache.clear()

        self.mock_model.prepare_conditionals.reset_mock()
        self.cache.get(path)
        self.mock_model.prepare_conditionals.assert_called_once()

    def test_resolved_path_dedup(self, tmp_path):
        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"data")
        # Use both relative-ish and absolute path
        path1 = str(voice_file)
        path2 = str(voice_file.resolve())
        self.cache.get(path1)
        self.cache.get(path2)
        # Same resolved path, same mtime → only 1 call
        assert self.mock_model.prepare_conditionals.call_count == 1
