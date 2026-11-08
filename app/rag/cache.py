"""Task 16 — in-memory response cache for the grounded-generation step.

Key = (collection, normalized masked query). Only the stateless retrieve+generate step is cached,
never the full crew response: with session memory, identical text can mean different things.
Normalization runs on the MASKED query, so raw PII never becomes a cache key.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from typing import Callable

from app.rag.generator import GroundedAnswer

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


class GenerationCache:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], GroundedAnswer] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self, collection: str, query: str, compute: Callable[[], GroundedAnswer]
    ) -> tuple[GroundedAnswer, bool]:
        key = (collection, normalize_query(query))
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                self.hits += 1
                return cached, True
        result = compute()  # computed outside the lock; a concurrent duplicate compute is harmless
        with self._lock:
            self._store[key] = result
            self.misses += 1
        return result, False

    def clear(self) -> int:
        """Invalidate everything (called when the knowledge base changes). Returns entries removed."""
        with self._lock:
            removed = len(self._store)
            self._store.clear()
            return removed

    def __len__(self) -> int:
        return len(self._store)
