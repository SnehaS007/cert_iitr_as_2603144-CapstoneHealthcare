"""Task 10 — output-side groundedness check.

Runs on the Composer's draft BEFORE it can reach the user. It does not trust the draft's own
`grounded` flag: it re-derives what is supported from the tools' own results (RagLookupTool.last_result,
AppointmentLookupTool.last_result), then:

- policy sentences are allowed only if the RAG tool's top-1 similarity cleared the calibrated threshold
  and the sentence is one the RAG tool actually returned;
- appointment sentences are allowed only if they are facts the lookup tool actually returned.

If nothing in the draft is supported, the answer is refused with the fallback message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import CrewDraft


@dataclass
class OutputGuardDecision:
    allowed: bool
    reason: str
    supported_sentences: list[str] = field(default_factory=list)
    removed_sentences: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # context handed to the review stage


def check_groundedness(draft: CrewDraft, rag_result: dict | None, lookup_result: dict | None) -> OutputGuardDecision:
    policy_ok = bool(rag_result) and rag_result["top1_similarity"] >= rag_result["threshold"]
    allowed_policy = set(rag_result["answer_sentences"]) if policy_ok else set()
    allowed_facts = set(lookup_result.get("facts", [])) if lookup_result else set()

    supported = [s for s in draft.answer_sentences if s in allowed_policy or s in allowed_facts]
    removed = [s for s in draft.answer_sentences if s not in supported]
    evidence = (list(rag_result["context"]) if policy_ok else []) + sorted(allowed_facts)

    if not supported:
        if rag_result and not policy_ok:
            reason = (f"low_retrieval_similarity: top-1 {rag_result['top1_similarity']:.4f} < "
                      f"threshold {rag_result['threshold']:.4f} and no appointment facts")
        else:
            reason = "no_supported_content"
        return OutputGuardDecision(False, reason, [], removed, evidence)
    reason = "supported" if not removed else f"removed_{len(removed)}_unsupported_sentence(s)"
    return OutputGuardDecision(True, reason, supported, removed, evidence)
