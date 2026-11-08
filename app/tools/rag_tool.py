"""rag_lookup — CrewAI tool for the Retrieval Agent: cached grounded generation (Tasks 3-5, 16)."""

from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from app.rag.cache import GenerationCache
from app.rag.generator import GroundedAnswer, GroundedGenerator


class RagLookupArgs(BaseModel):
    query: str = Field(..., min_length=1, description="The patient's policy question")


def rag_result_payload(result: GroundedAnswer, cache_hit: bool) -> dict:
    return {
        "grounded": result.grounded,
        "top1_similarity": round(result.top1_similarity, 4),
        "threshold": result.threshold,
        "collection": result.collection,
        "answer_sentences": result.used_sentences,
        "sources": result.source_doc_ids,
        "context": [c.text for c in result.supporting_chunks],
        "cache_hit": cache_hit,
    }


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Answer a clinic-policy question using only the Practo policy knowledge base."
    args_schema: type[BaseModel] = RagLookupArgs
    _generator: GroundedGenerator = PrivateAttr()
    _cache: GenerationCache = PrivateAttr()
    _invocations: list[str] = PrivateAttr(default_factory=list)
    _last_result: dict | None = PrivateAttr(default=None)

    def __init__(self, generator: GroundedGenerator, cache: GenerationCache, **kwargs):
        super().__init__(**kwargs)
        self._generator = generator
        self._cache = cache

    @property
    def invocations(self) -> list[str]:
        return list(self._invocations)

    @property
    def last_result(self) -> dict | None:
        """Ground truth for the output guard: taken from the tool, not from what an LLM repeated."""
        return self._last_result

    def _run(self, query: str) -> str:
        self._invocations.append(query)
        result, hit = self._cache.get_or_compute(
            self._generator.retriever.collection_name, query, lambda: self._generator.answer(query)
        )
        self._last_result = rag_result_payload(result, hit)
        return json.dumps(self._last_result)
