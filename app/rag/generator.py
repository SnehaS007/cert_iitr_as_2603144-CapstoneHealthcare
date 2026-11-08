"""Task 4 — grounded generation: retrieve top-k, answer ONLY from retrieved context, or fall back.

Under MOCK_LLM, retrieval similarity is the only groundedness signal: if the top-1 cosine
similarity is below the calibrated threshold, the LLM is not called and the fallback is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import app.config as config
from app.llm.mock_llm import MockGroundedLLM
from app.rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    answer: str
    grounded: bool
    top1_similarity: float
    threshold: float
    collection: str
    source_doc_ids: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    used_sentences: list[str] = field(default_factory=list)       # answer body, sentence by sentence
    supporting_chunks: list[RetrievedChunk] = field(default_factory=list)  # context given to the LLM


class GroundedGenerator:
    def __init__(self, retriever: Retriever, llm: MockGroundedLLM, threshold: float, k: int = config.TOP_K):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold
        self.k = k

    def answer(self, query: str) -> GroundedAnswer:
        retrieved = self.retriever.retrieve(query, k=self.k)
        top1 = retrieved[0].similarity
        common = dict(
            query=query,
            top1_similarity=top1,
            threshold=self.threshold,
            collection=self.retriever.collection_name,
            retrieved=retrieved,
        )
        if top1 < self.threshold:
            return GroundedAnswer(answer=config.FALLBACK_ANSWER, grounded=False, **common)
        # Only chunks that individually clear the threshold are given to the LLM as context.
        supporting = [c for c in retrieved if c.similarity >= self.threshold]
        generated = self.llm.generate(query, supporting)
        return GroundedAnswer(
            answer=generated.text,
            grounded=True,
            source_doc_ids=generated.source_doc_ids,
            used_sentences=generated.used_sentences,
            supporting_chunks=supporting,
            **common,
        )
