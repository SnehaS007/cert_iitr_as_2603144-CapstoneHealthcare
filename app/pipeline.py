"""End-to-end request pipeline.

  input guard (PII mask, injection) -> runtime budget -> session memory -> CrewAI crew
  -> output groundedness guard -> AutoGen review -> SupportResponse (validated) -> JSONL log

Every request writes exactly one log entry, including rejected and failed requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import app.config as config
from app.crew.crew import run_crew
from app.governance.budget import BudgetExceededError, TokenBudget
from app.guardrails.input_guard import InputRejectedError, guard_input
from app.guardrails.output_guard import check_groundedness
from app.llm.mock_llm import get_generation_llm
from app.logging_utils import JsonlRequestLogger, StageTimer, new_trace_id
from app.memory.session import SessionStore, resolve_record_id, with_session_memory
from app.rag.cache import GenerationCache
from app.rag.chunking import fixed_size_chunks, sentence_chunks
from app.rag.embeddings import Embedder
from app.rag.generator import GroundedGenerator
from app.rag.indexer import get_client, upsert_chunks
from app.rag.kb import KBDocument
from app.rag.retriever import Retriever
from app.review.autogen_review import review
from app.schemas import AddDocumentResponse, SupportResponse


class DocumentConflictError(ValueError):
    pass


@dataclass
class TurnTrace:
    """Diagnostics for demos and tests; not part of the API response."""

    record_id: str | None = None
    record_id_source: str = "none"
    rag_invocations: list[str] = field(default_factory=list)
    lookup_invocations: list[str] = field(default_factory=list)
    lookup_facts: list[str] = field(default_factory=list)
    crew_events: list[dict] = field(default_factory=list)
    output_guard: str = ""
    review_messages: list[tuple[str, str]] = field(default_factory=list)


class SupportPipeline:
    def __init__(
        self,
        embedder: Embedder,
        client=None,
        collection: str | None = None,
        threshold: float | None = None,
        budget: TokenBudget | None = None,
        logger: JsonlRequestLogger | None = None,
        cache: GenerationCache | None = None,
        store: SessionStore | None = None,
    ):
        self.embedder = embedder
        self.client = client or get_client()
        self.collection = collection or config.get_recommended_collection()
        self.threshold = threshold if threshold is not None else config.get_grounding_threshold(self.collection)
        self.generation_llm = get_generation_llm(embedder)
        self.generator = GroundedGenerator(
            Retriever(self.collection, embedder, self.client), self.generation_llm, self.threshold
        )
        self.budget = budget or TokenBudget()
        self.logger = logger or JsonlRequestLogger()
        self.cache = cache or GenerationCache()
        self.store = store or SessionStore()
        self._runnable = with_session_memory(self._core, self.store)
        self.last_trace: TurnTrace | None = None

    # --- one conversational turn ----------------------------------------------------------------
    def ask(self, query: str, session_id: str, endpoint: str = "/ask", trace_id: str | None = None) -> SupportResponse:
        trace_id, timer = trace_id or new_trace_id(), StageTimer()
        guard = guard_input(query)
        timer.lap("input_guard")
        base = dict(trace_id=trace_id, endpoint=endpoint, session_id=session_id, masked_query=guard.masked_text)

        if guard.injection_detected:
            self.logger.log_request(**base, outcome="rejected_injection", guardrail_events=guard.events,
                                    stage_timings_ms=timer.stages, total_ms=timer.total_ms)
            raise InputRejectedError(guard)

        # Session history is used only to resolve references (record IDs); it is not sent to the
        # agents, so the budget covers what the crew actually receives.
        try:
            estimate = self.budget.enforce_preflight(guard.masked_text)
        except BudgetExceededError as exc:
            timer.lap("budget")
            self.logger.log_request(**base, outcome="rejected_budget", guardrail_events=guard.events,
                                    budget_stage=exc.estimate.stage, tokens=exc.estimate.tokens,
                                    token_cap=exc.estimate.cap, stage_timings_ms=timer.stages,
                                    total_ms=timer.total_ms)
            raise
        timer.lap("budget")

        try:
            out = self._runnable.invoke(
                {"input": guard.masked_text, "trace_id": trace_id, "session_id": session_id,
                 "guard_events": guard.events},
                config={"configurable": {"session_id": session_id}},
            )
        except BudgetExceededError as exc:
            self.logger.log_request(**base, outcome="rejected_budget", guardrail_events=guard.events,
                                    budget_stage=exc.estimate.stage, tokens=exc.estimate.tokens,
                                    token_cap=exc.estimate.cap, stage_timings_ms=timer.stages,
                                    total_ms=timer.total_ms)
            raise
        except Exception as exc:
            self.logger.log_request(**base, outcome="error", error_type=type(exc).__name__,
                                    guardrail_events=guard.events, stage_timings_ms=timer.stages,
                                    total_ms=timer.total_ms)
            raise
        response: SupportResponse = out["response"]
        timer.stages.update(out["stage_timings_ms"])
        self.logger.log_request(
            **base,
            outcome="refused" if response.refusal_reason else "answered",
            guardrail_events=response.guardrail_events,
            record_id_source=out["trace"].record_id_source,
            cache_hit=response.cache_hit,
            grounded=response.grounded,
            refusal_reason=response.refusal_reason,
            review_approved=response.review.approved if response.review else None,
            escalation_recommended=response.escalation_recommended,
            tokens_projected=estimate.tokens,
            cost_projected_usd=estimate.cost_usd,
            tokens_consumed=out["crew_llm_input_tokens"],
            cost_consumed_usd=self.budget.cost(out["crew_llm_input_tokens"]),
            token_cap=self.budget.cap,
            stage_timings_ms=timer.stages,
            total_ms=timer.total_ms,
        )
        self.last_trace = out["trace"]
        return response

    def _core(self, inputs: dict) -> dict:
        timer = StageTimer()
        masked, history = inputs["input"], inputs.get("history", [])
        trace = TurnTrace()
        trace.record_id, trace.record_id_source = resolve_record_id(masked, history)
        timer.lap("memory")

        run = run_crew(masked, trace.record_id, self.generator, self.cache, self.budget)
        timer.lap("crew")
        trace.rag_invocations, trace.lookup_invocations = run.rag_invocations, run.lookup_invocations
        trace.crew_events = run.llm_events
        trace.lookup_facts = list(run.lookup_result.get("facts", [])) if run.lookup_result else []

        decision = check_groundedness(run.draft, run.rag_result, run.lookup_result)
        trace.output_guard = decision.reason
        timer.lap("output_guard")
        events = list(inputs.get("guard_events", []))
        if decision.removed_sentences:
            events.append(f"output_guard_removed:{len(decision.removed_sentences)}")

        appointment = run.draft.appointment
        if not decision.allowed:
            events.append("output_guard_refused")
            response = SupportResponse(
                trace_id=inputs["trace_id"], session_id=inputs["session_id"], answer=config.FALLBACK_ANSWER, grounded=False,
                sources=[], appointment=None, escalation_recommended=False,
                refusal_reason=decision.reason, review=None,
                cache_hit=bool(run.rag_result and run.rag_result["cache_hit"]), guardrail_events=events,
            )
        else:
            outcome = review(" ".join(decision.supported_sentences), decision.evidence)
            trace.review_messages = outcome.messages
            timer.lap("review")
            if not outcome.verdict.approved:
                events.append("review_revised")
            response = SupportResponse(
                trace_id=inputs["trace_id"], session_id=inputs["session_id"], answer=outcome.verdict.final_answer, grounded=True,
                sources=run.draft.sources, appointment=appointment,
                escalation_recommended=bool(appointment and appointment.escalation_recommended),
                refusal_reason=None, review=outcome.verdict,
                cache_hit=bool(run.rag_result and run.rag_result["cache_hit"]), guardrail_events=events,
            )
        return {
            "answer": response.answer,
            "response": response,
            "trace": trace,
            "stage_timings_ms": timer.stages,
            "crew_llm_input_tokens": run.llm_input_tokens,
        }

    # --- knowledge-base updates -----------------------------------------------------------------
    def add_document(self, doc_id: str, title: str, text: str, trace_id: str | None = None) -> AddDocumentResponse:
        trace_id, timer = trace_id or new_trace_id(), StageTimer()
        title_guard, text_guard = guard_input(title), guard_input(text)
        events = title_guard.events + text_guard.events
        base = dict(trace_id=trace_id, endpoint="/add-document", doc_id=doc_id)
        if title_guard.injection_detected or text_guard.injection_detected:
            # Indirect prompt injection: documents become model context, so they get the same check.
            self.logger.log_request(**base, outcome="rejected_injection", guardrail_events=events,
                                    total_ms=timer.total_ms)
            raise InputRejectedError(text_guard if text_guard.injection_detected else title_guard)

        doc = KBDocument(doc_id=doc_id, title=title_guard.masked_text, text=text_guard.masked_text)
        if not 2 <= len(doc.sentences) <= 5:
            raise ValueError(f"Document must have 2-5 sentences, got {len(doc.sentences)}")
        plan = [(self.client.get_collection(name, embedding_function=None), chunks)
                for name, chunks in ((config.COLLECTION_FIXED, fixed_size_chunks(doc)),
                                     (config.COLLECTION_SENTENCE, sentence_chunks(doc)))]
        for collection, _ in plan:  # check both collections before writing to either
            if collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
                raise DocumentConflictError(f"Document '{doc_id}' already exists in '{collection.name}'")
        counts: dict[str, int] = {}
        for collection, chunks in plan:
            upsert_chunks(collection, chunks, self.embedder)
            counts[collection.name] = len(chunks)
        cleared = self.cache.clear()  # cached answers may now be stale
        self.logger.log_request(**base, outcome="indexed", guardrail_events=events, chunks_indexed=counts,
                                cache_entries_cleared=cleared, total_ms=timer.total_ms)
        return AddDocumentResponse(doc_id=doc_id, chunks_indexed=counts, cache_entries_cleared=cleared)
