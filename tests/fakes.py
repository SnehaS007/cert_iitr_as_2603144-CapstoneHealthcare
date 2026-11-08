"""Test-only embedder: deterministic hashed bag-of-words (lexical, not semantic).

Used ONLY by tests, injected explicitly, so the RAG pipeline logic can be exercised without the
SentenceTransformers model. It is never used by application code. Numbers it produces are not
representative of MiniLM similarities.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what", "when", "will",
    "with", "who", "you", "your", "this", "that", "there", "he", "him", "our",
}


class HashingEmbedder:
    name = "test-hashing-embedder"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in _STOPWORDS]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in self._tokens(text):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
