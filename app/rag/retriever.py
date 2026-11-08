"""Top-k retrieval from one Chroma collection. Similarity = 1 - cosine distance."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

import app.config as config
from app.rag.embeddings import Embedder
from app.rag.indexer import get_client


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float


class Retriever:
    def __init__(self, collection_name: str, embedder: Embedder, client: chromadb.ClientAPI | None = None):
        if collection_name not in config.COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection_name}'; expected one of {config.COLLECTIONS}")
        self.collection_name = collection_name
        self.embedder = embedder
        self._collection = (client or get_client()).get_collection(collection_name, embedding_function=None)
        indexed_with = (self._collection.metadata or {}).get("embedder")
        if indexed_with != embedder.name:
            raise RuntimeError(
                f"Collection '{collection_name}' was indexed with '{indexed_with}' but the query "
                f"embedder is '{embedder.name}'. Re-run the indexer."
            )

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        result = self._collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                title=meta["title"],
                text=text,
                similarity=1.0 - distance,
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
