import pytest

import app.config as config
from app.llm.mock_llm import MockGroundedLLM
from app.rag.generator import GroundedGenerator
from app.rag.retriever import RetrievedChunk
from tests.fakes import HashingEmbedder

CONTEXT = [
    RetrievedChunk("fee::0", "consultation_fee_structure", "Fees",
                   "Cardiology consultations range from INR 800 to INR 2,000. Fees vary by doctor.", 0.8),
    RetrievedChunk("lab::0", "lab_test_turnaround", "Labs", "Routine blood tests are reported within 24 hours.", 0.4),
]


class _StubRetriever:
    collection_name = config.COLLECTION_SENTENCE

    def __init__(self, top1: float):
        self.top1 = top1

    def retrieve(self, query, k=3):
        return [RetrievedChunk(c.chunk_id, c.doc_id, c.title, c.text, self.top1 if i == 0 else c.similarity)
                for i, c in enumerate(CONTEXT)]


def _gen(top1: float) -> GroundedGenerator:
    return GroundedGenerator(_StubRetriever(top1), MockGroundedLLM(HashingEmbedder()), threshold=0.5)


def test_below_threshold_falls_back_without_llm_call():
    g = _gen(0.49)
    result = g.answer("How much is cardiology?")
    assert not result.grounded and result.answer == config.FALLBACK_ANSWER
    assert g.llm.calls == 0


def test_answer_uses_only_retrieved_sentences():
    g = _gen(0.8)
    result = g.answer("How much does a cardiology consultation cost?")
    context_text = " ".join(c.text for c in CONTEXT)
    llm_out = g.llm.generate("How much does a cardiology consultation cost?", CONTEXT)
    assert result.grounded and g.llm.calls == 2
    assert all(s in context_text for s in llm_out.used_sentences)
    assert llm_out.used_sentences[0].startswith("Cardiology")


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        GroundedGenerator(_StubRetriever(0.8), MockGroundedLLM(HashingEmbedder()), threshold=0.0)


def test_chunks_below_threshold_are_not_given_to_llm():
    result = _gen(0.8).answer("How much does a cardiology consultation cost?")
    assert result.source_doc_ids == ["consultation_fee_structure"]  # lab chunk (0.4 < 0.5) excluded


def test_fixed_chunk_fragments_are_dropped():
    llm = MockGroundedLLM(HashingEmbedder())
    context = [
        RetrievedChunk("h::0", "home", "Home", "Home visits need 24 hours notice. A home visit costs the doctor's consultation fee", 0.9),
        RetrievedChunk("h::1", "home", "Home", "A home visit costs the doctor's consultation fee plus a travel charge of INR 300. Each visit", 0.8),
    ]
    out = llm.generate("How much does a home visit cost?", context)
    assert "A home visit costs the doctor's consultation fee" not in out.used_sentences
    assert "Each visit" not in out.used_sentences
    assert "A home visit costs the doctor's consultation fee plus a travel charge of INR 300." in out.used_sentences
