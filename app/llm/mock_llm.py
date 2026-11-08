"""Deterministic MOCK_LLM used for grounded generation (zero keys, zero network).

It composes the answer ONLY from sentences of the retrieved context: each candidate sentence is
scored by cosine similarity to the query and the best ones are returned. Extractive output means
every answer sentence is, by construction, present in the retrieved context.
"""

from __future__ import annotations

from dataclasses import dataclass

import app.config as config
from app.rag.embeddings import Embedder
from app.rag.kb import split_sentences
from app.rag.retriever import RetrievedChunk

MIN_SENTENCE_WORDS = 4  # shorter pieces are chunk-boundary fragments, not answerable sentences


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized, so dot == cosine


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    used_sentences: list[str]
    source_doc_ids: list[str]


class MockGroundedLLM:
    def __init__(self, embedder: Embedder, max_sentences: int = config.MAX_ANSWER_SENTENCES):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.calls = 0  # call counter; also used later as cache-hit evidence (Task 16)

    def generate(self, query: str, context: list[RetrievedChunk]) -> GeneratedAnswer:
        if not context:
            raise ValueError("MockGroundedLLM.generate requires a non-empty context")
        self.calls += 1

        candidates: list[tuple[str, str, str]] = []  # (sentence, doc_id, title), de-duplicated
        seen: set[str] = set()
        for chunk in context:
            for sentence in split_sentences(chunk.text):
                if sentence not in seen:
                    seen.add(sentence)
                    candidates.append((sentence, chunk.doc_id, chunk.title))

        # Fixed-size windows cut sentences at their edges. Drop a fragment when a longer candidate
        # contains it (the overlap usually carries the full sentence) and drop tiny fragments.
        texts = [s for s, _, _ in candidates]
        candidates = [
            c for c in candidates
            if len(c[0].split()) >= MIN_SENTENCE_WORDS
            and not any(c[0] != other and c[0] in other for other in texts)
        ]
        if not candidates:
            raise ValueError("No usable sentences in the retrieved context")

        query_vec = self.embedder.encode([query])[0]
        sentence_vecs = self.embedder.encode([s for s, _, _ in candidates])
        ranked = sorted(
            zip(candidates, sentence_vecs), key=lambda pair: _dot(query_vec, pair[1]), reverse=True
        )[: self.max_sentences]

        used = [cand for cand, _ in ranked]
        titles = list(dict.fromkeys(title for _, _, title in used))
        text = " ".join(s for s, _, _ in used) + f" (Source: {'; '.join(titles)})"
        return GeneratedAnswer(
            text=text,
            used_sentences=[s for s, _, _ in used],
            source_doc_ids=list(dict.fromkeys(doc_id for _, doc_id, _ in used)),
        )


def get_generation_llm(embedder: Embedder) -> MockGroundedLLM:
    if not config.MOCK_LLM:
        raise NotImplementedError(
            "MOCK_LLM=false: a real LLM backend is not wired in yet. Set MOCK_LLM=true."
        )
    return MockGroundedLLM(embedder)
