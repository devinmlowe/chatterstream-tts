"""LRU cache for voice conditioning data, keyed by (resolved_path, mtime)."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

from .interfaces import ConditioningCacheBase


class ConditioningCache(ConditioningCacheBase):
    """Caches Conditionals objects to avoid repeated extraction.

    Cache key: (resolved_path, file_mtime).  Invalidates automatically when
    the voice file is modified.  Uses an LRU eviction policy.
    """

    def __init__(self, model, maxsize: int = 8):
        self._model = model
        self._maxsize = maxsize
        # OrderedDict for LRU: most-recently-used items are moved to end
        self._cache: OrderedDict[tuple[str, float], object] = OrderedDict()

    def get(self, voice_path: str) -> object:
        resolved = str(Path(voice_path).resolve())
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        mtime = os.path.getmtime(resolved)
        key = (resolved, mtime)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # Evict stale entries for same path (different mtime)
        stale = [k for k in self._cache if k[0] == resolved]
        for k in stale:
            del self._cache[k]

        # Evict LRU if at capacity
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        result = self._model.prepare_conditionals(resolved)
        self._cache[key] = result
        return result

    def clear(self) -> None:
        self._cache.clear()
