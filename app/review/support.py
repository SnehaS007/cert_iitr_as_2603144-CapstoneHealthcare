"""Lexical claim-support check used by the AutoGen Policy-Compliance Reviewer.

Deliberately independent of the embedding similarity used by the output guard: the reviewer uses
token overlap against the evidence, so the two checks can disagree and catch each other's misses.
"""

from __future__ import annotations

import re

from app.rag.kb import split_sentences

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "the", "to", "was", "were", "will", "with", "your", "you", "this",
    "that", "our", "any", "all", "only", "same", "each",
}
SUPPORT_THRESHOLD = 0.85


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold().replace(",", "")) if t not in _STOP}


def support_score(sentence: str, evidence: list[str]) -> float:
    """Best fraction of the sentence's content tokens covered by any single evidence sentence."""
    tokens = content_tokens(sentence)
    if not tokens:
        return 1.0
    evidence_sentences = [s for block in evidence for s in split_sentences(block)] or [""]
    return max(len(tokens & content_tokens(e)) / len(tokens) for e in evidence_sentences)


def unsupported_sentences(answer: str, evidence: list[str]) -> list[str]:
    return [s for s in split_sentences(answer) if support_score(s, evidence) < SUPPORT_THRESHOLD]
