"""Task 3 — two chunking strategies over the same knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    FIXED_CHUNK_WORDS,
    FIXED_OVERLAP_WORDS,
    SENTENCES_PER_CHUNK,
)
from app.rag.kb import KBDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    strategy: str
    index: int

    @property
    def embed_text(self) -> str:
        """Text sent to the embedder. The document title is prepended for BOTH strategies so the
        comparison stays fair and short chunks keep their topic context."""
        return f"{self.title}: {self.text}"

    @property
    def metadata(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "strategy": self.strategy, "index": self.index}


def fixed_size_chunks(
    doc: KBDocument, size: int = FIXED_CHUNK_WORDS, overlap: int = FIXED_OVERLAP_WORDS
) -> list[Chunk]:
    """Sliding word windows of `size` words; consecutive windows share `overlap` words.

    Word-based (not character-based) windows avoid cutting words in half, which degrades embeddings.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"invalid window: size={size}, overlap={overlap}")
    words = doc.text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::fixed::{index}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=" ".join(words[start : start + size]),
                strategy=COLLECTION_FIXED,
                index=index,
            )
        )
        if start + size >= len(words):
            break
    return chunks


def sentence_chunks(doc: KBDocument, per_chunk: int = SENTENCES_PER_CHUNK) -> list[Chunk]:
    """Group consecutive sentences, `per_chunk` sentences per chunk, no overlap."""
    if per_chunk <= 0:
        raise ValueError(f"per_chunk must be positive, got {per_chunk}")
    sentences = doc.sentences
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::sentence::{index}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=" ".join(sentences[start : start + per_chunk]),
            strategy=COLLECTION_SENTENCE,
            index=index,
        )
        for index, start in enumerate(range(0, len(sentences), per_chunk))
    ]


def chunk_corpus(docs: list[KBDocument]) -> dict[str, list[Chunk]]:
    """Return {collection_name: chunks} for both strategies."""
    return {
        COLLECTION_FIXED: [c for d in docs for c in fixed_size_chunks(d)],
        COLLECTION_SENTENCE: [c for d in docs for c in sentence_chunks(d)],
    }
