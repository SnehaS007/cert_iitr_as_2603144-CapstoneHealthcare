"""Embedding interface. Production uses a local SentenceTransformers model.

The embedder is passed explicitly into indexing/retrieval (dependency injection). There is no
fallback: if the model is not cached and the Hub is unreachable, SentenceTransformer raises.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import app.config as config


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()


@lru_cache(maxsize=1)
def get_default_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
