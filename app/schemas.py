"""Pydantic schemas.

- CrewDraft:      Task 9 structured output every CrewAI crew run must conform to (output_pydantic).
- ReviewVerdict:  Task 14 structured output of the AutoGen Final-Editor.
- SupportResponse and API request/response models: Task 11.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import app.config as config

RECORD_ID_PATTERN = r"^APT-\d{4}$"


class AppointmentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=RECORD_ID_PATTERN)
    category: str
    status: str
    consultation_fee_inr: int = Field(ge=0)
    days_since_created: int = Field(ge=0, le=30)
    follow_up_required: bool
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_threshold: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool


class CrewDraft(BaseModel):
    """Structured output of the Response Composer (the crew's final task)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    answer_sentences: list[str]
    grounded: bool
    policy_used: bool
    sources: list[str]
    appointment: AppointmentStatus | None = None
    escalation_recommended: bool = False


class ReviewVerdict(BaseModel):
    """Structured output of the AutoGen Final-Editor (brief: approved, final_answer, reason)."""

    approved: bool
    final_answer: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SupportResponse(BaseModel):
    """Final validated response returned by the pipeline and the API."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    session_id: str
    answer: str = Field(min_length=1)
    grounded: bool
    sources: list[str]
    appointment: AppointmentStatus | None = None
    escalation_recommended: bool = False
    refusal_reason: str | None = None
    review: ReviewVerdict | None = None
    cache_hit: bool = False
    guardrail_events: list[str] = Field(default_factory=list)


# --- API models ---------------------------------------------------------------------------------
class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)


class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=3, max_length=64, pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=20, max_length=3000)


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_indexed: dict[str, int]
    cache_entries_cleared: int


class ErrorResponse(BaseModel):
    trace_id: str
    error: Literal["input_rejected", "budget_exceeded", "validation_error", "conflict", "internal_error"]
    detail: str
