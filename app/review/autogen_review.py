"""Task 14 — AutoGen review stage after the crew's draft.

RoundRobinGroupChat([Policy-Compliance Reviewer, Final Editor], max_turns=2):
  turn 1: the reviewer checks every draft sentence against the evidence (retrieved context + tool facts);
  turn 2: the Final Editor returns StructuredMessage[ReviewVerdict] (approved, final_answer, reason),
          approving the draft unchanged or revising it.
The team is constructed with custom_message_types=[StructuredMessage[ReviewVerdict]] as the brief
requires (autogen-agentchat 0.7.5 also auto-registers it; passing it explicitly is harmless).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

import app.config as config
from app.llm.mock_autogen_client import EDITOR_NAME, REVIEWER_NAME, MockReviewClient
from app.schemas import ReviewVerdict


class ReviewStageError(RuntimeError):
    pass


@dataclass
class ReviewOutcome:
    verdict: ReviewVerdict
    messages: list[tuple[str, str]]  # (source, content) transcript of the team run
    model_calls: int


def build_review_team() -> tuple[RoundRobinGroupChat, list[MockReviewClient]]:
    if not config.MOCK_LLM:
        raise NotImplementedError("MOCK_LLM=false: no real model client is wired in. Set MOCK_LLM=true.")
    reviewer_client, editor_client = MockReviewClient("reviewer"), MockReviewClient("editor")
    reviewer = AssistantAgent(
        name=REVIEWER_NAME,
        model_client=reviewer_client,
        system_message=(
            "You are a Policy-Compliance Reviewer for Practo patient support. Check every sentence of the "
            "draft against the evidence. List any sentence the evidence does not support. Do not rewrite."
        ),
    )
    editor = AssistantAgent(
        name=EDITOR_NAME,
        model_client=editor_client,
        system_message=(
            "You are the Final Editor. If the reviewer found no unsupported sentences, approve the draft "
            "unchanged. Otherwise remove the unsupported sentences and explain why."
        ),
        output_content_type=ReviewVerdict,
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team, [reviewer_client, editor_client]


async def areview(draft: str, evidence: list[str]) -> ReviewOutcome:
    if not draft.strip():
        raise ValueError("draft must not be empty")
    team, clients = build_review_team()
    task = json.dumps({"draft": draft, "evidence": evidence})
    result = await team.run(task=task)

    verdicts = [
        m for m in result.messages
        if isinstance(m, StructuredMessage) and isinstance(m.content, ReviewVerdict)
    ]
    if len(verdicts) != 1:
        raise ReviewStageError(f"Expected exactly one ReviewVerdict, got {len(verdicts)}")
    transcript = [
        (m.source, m.content.model_dump_json() if isinstance(m, StructuredMessage) else str(m.content))
        for m in result.messages
    ]
    return ReviewOutcome(
        verdict=ReviewVerdict.model_validate(verdicts[0].content.model_dump()),
        messages=transcript,
        model_calls=sum(c.calls for c in clients),
    )


def review(draft: str, evidence: list[str]) -> ReviewOutcome:
    """Sync wrapper. Must not be called from a thread with a running event loop (FastAPI calls the
    pipeline in a worker thread, where this is safe)."""
    return asyncio.run(areview(draft, evidence))
