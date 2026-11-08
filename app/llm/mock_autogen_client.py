"""MockReviewClient — deterministic autogen_core ChatCompletionClient (zero keys, zero network).

One instance per review agent, with a fixed `mode`:
- "reviewer": reads the task JSON {draft, evidence} and returns JSON listing unsupported sentences.
- "editor":   reads the task JSON and the reviewer's findings and returns a ReviewVerdict JSON.
              AssistantAgent(output_content_type=ReviewVerdict) passes json_output=ReviewVerdict and
              parses our JSON into StructuredMessage[ReviewVerdict].
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

import app.config as config
from app.governance.budget import estimate_tokens
from app.rag.kb import split_sentences
from app.review.support import unsupported_sentences
from app.schemas import ReviewVerdict

REVIEWER_NAME = "policy_compliance_reviewer"
EDITOR_NAME = "final_editor"


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _find_json(messages: Sequence[LLMMessage], key: str, source: str | None = None) -> dict | None:
    for msg in messages:
        if source is not None and getattr(msg, "source", None) != source:
            continue
        try:
            data = json.loads(_text(getattr(msg, "content", "")))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and key in data:
            return data
    return None


class MockReviewClient(ChatCompletionClient):
    def __init__(self, mode: Literal["reviewer", "editor"]):
        if mode not in ("reviewer", "editor"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.calls = 0
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # --- core -----------------------------------------------------------------------------------
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        self.calls += 1
        task = _find_json(messages, "draft")
        if task is None:
            raise ValueError("Review task JSON with 'draft' and 'evidence' not found in messages")

        if self.mode == "reviewer":
            flagged = unsupported_sentences(task["draft"], task["evidence"])
            content = json.dumps({"unsupported": flagged, "compliant": not flagged})
        else:
            findings = _find_json(messages, "unsupported", source=REVIEWER_NAME)
            if findings is None:
                raise ValueError("Reviewer findings not found; the editor must run after the reviewer")
            content = self._verdict(task["draft"], findings["unsupported"]).model_dump_json()

        prompt_tokens = sum(estimate_tokens(_text(getattr(m, "content", ""))) for m in messages)
        self._last = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=estimate_tokens(content))
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + self._last.prompt_tokens,
            completion_tokens=self._usage.completion_tokens + self._last.completion_tokens,
        )
        return CreateResult(finish_reason="stop", content=content, usage=self._last, cached=False)

    @staticmethod
    def _verdict(draft: str, unsupported: list[str]) -> ReviewVerdict:
        if not unsupported:
            return ReviewVerdict(approved=True, final_answer=draft, reason="All claims are supported by the evidence.")
        kept = [s for s in split_sentences(draft) if s not in unsupported]
        final = " ".join(kept) if kept else config.FALLBACK_ANSWER
        return ReviewVerdict(
            approved=False,
            final_answer=final,
            reason=f"Removed {len(unsupported)} claim(s) not supported by the evidence: " + " | ".join(unsupported),
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, json_output=json_output, cancellation_token=cancellation_token)

    # --- bookkeeping ----------------------------------------------------------------------------
    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return sum(estimate_tokens(_text(getattr(m, "content", ""))) for m in messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))

    @property
    def capabilities(self) -> ModelCapabilities:  # deprecated upstream, still abstract in 0.7.5
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False, function_calling=False, json_output=True,
            family="unknown", structured_output=True, multiple_system_messages=True,
        )
