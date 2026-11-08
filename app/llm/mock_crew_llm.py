"""MockCrewLLM — deterministic CrewAI LLM (subclass of crewai.BaseLLM), zero keys, zero network.

Behaviour verified against crewai 1.15.36 with supports_function_calling() == False (ReAct text path):
  1. CrewAI sends [system, user]; we reply with "Action: <tool>" + "Action Input: {json}".
  2. CrewAI runs the tool and appends "Observation: <result>" to OUR assistant message, then asks again.
  3. We reply "Final Answer: <observation>".

Pitfall 1 (brief): CrewAI's system prompt literally contains "Observation: the result of the action".
  -> Observations are read ONLY from role == "assistant" messages that contain our own "Action Input:"
     line, and only from text that follows that line. System/user prompt text is never parsed for them.
Pitfall 2 (brief): never dispatch by tool name (a tool named "rag_lookup" contains "lookup").
  -> The tool is selected by its declared args_schema: the agent's tool whose required fields are all
     present in the task's TOOL_INPUT_JSON line is called, with exactly those fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import BaseLLM
from pydantic import Field

import app.config as config
from app.governance.budget import TokenBudget, estimate_tokens
from app.schemas import CrewDraft

TOOL_INPUT_MARKER = "TOOL_INPUT_JSON:"
_TOOL_INPUT_RE = re.compile(rf"^{re.escape(TOOL_INPUT_MARKER)}\s*(\{{.*\}})\s*$", re.MULTILINE)
_CONTEXT_HEADER = "This is the context you're working with:"


def _final(payload: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {payload}"


def _own_observation(messages: list[dict]) -> str | None:
    """Return the tool result for OUR most recent action, or None if we have not acted yet."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        marker = content.find("\nAction Input:")
        if marker == -1:
            return None  # our latest assistant turn was not a tool call
        after = content[marker:]
        obs_at = after.find("\nObservation:")
        if obs_at != -1:
            return after[obs_at + len("\nObservation:"):].strip()
        nxt = messages[i + 1] if i + 1 < len(messages) else None  # observation as a separate message
        if nxt and (nxt.get("content") or "").lstrip().startswith("Observation:"):
            return nxt["content"].lstrip()[len("Observation:"):].strip()
        return None
    return None


def _user_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")


def _required_fields(tool: Any) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        raise ValueError(f"Tool {tool!r} has no args_schema; cannot dispatch by schema.")
    return {name for name, f in schema.model_fields.items() if f.is_required()}


def _context_objects(text: str) -> list[dict]:
    if _CONTEXT_HEADER not in text:
        return []
    objects = []
    for line in text.split(_CONTEXT_HEADER, 1)[1].splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # non-JSON context lines are prose, not tool payloads
    return objects


def compose_draft(context: list[dict]) -> CrewDraft:
    """Response Composer behaviour: combine the RAG and lookup payloads into one validated draft."""
    rag = next((o for o in context if "top1_similarity" in o), None)
    lookup = next((o for o in context if "record_found" in o), None)
    sentences: list[str] = []
    sources: list[str] = []
    policy_used = bool(rag and rag.get("grounded") and rag.get("answer_sentences"))
    if policy_used:
        sentences += rag["answer_sentences"]
        sources += rag["sources"]
    appointment = lookup.get("appointment") if lookup and lookup.get("record_found") else None
    if lookup and lookup.get("facts"):
        sentences += lookup["facts"]
        sources.append("appointment_records")
    grounded = bool(sentences)
    return CrewDraft(
        answer=" ".join(sentences) if grounded else config.FALLBACK_ANSWER,
        answer_sentences=sentences,
        grounded=grounded,
        policy_used=policy_used,
        sources=sources,
        appointment=appointment,
        escalation_recommended=bool(appointment and appointment["escalation_recommended"]),
    )


class MockCrewLLM(BaseLLM):
    calls: int = 0
    input_tokens: int = 0
    events: list[dict] = Field(default_factory=list)
    budget: TokenBudget | None = None  # runtime hard stop (Task 15)

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        """CrewAI's override point for token estimation; same heuristic as the runtime budget."""
        if isinstance(messages, str):
            return estimate_tokens(messages)
        return sum(estimate_tokens(str(m.get("content") or "")) for m in messages)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        self.calls += 1
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else [dict(m) for m in messages]
        self.input_tokens += self.get_input_tokens_count(msgs)
        if self.budget is not None:
            self.budget.enforce_runtime(self.input_tokens)  # raises BudgetExceededError; never exceeds silently
        role = getattr(from_agent, "role", "unknown")

        observation = _own_observation(msgs)
        if observation is not None:
            self.events.append({"agent": role, "event": "final_answer_from_tool"})
            return _final(observation)

        text = _user_text(msgs)
        agent_tools = list(getattr(from_agent, "tools", None) or [])
        if not agent_tools:  # Composer: no tools, combines prior task outputs
            draft = compose_draft(_context_objects(text))
            self.events.append({"agent": role, "event": "compose", "grounded": draft.grounded})
            return _final(draft.model_dump_json())

        match = _TOOL_INPUT_RE.search(text)
        provided: dict = json.loads(match.group(1)) if match else {}
        provided = {k: v for k, v in provided.items() if v not in (None, "")}
        for tool in agent_tools:
            required = _required_fields(tool)
            if required and required <= provided.keys():
                args = {k: provided[k] for k in required}
                self.events.append({"agent": role, "event": "tool_call", "tool": tool.name, "args": args})
                return f"Thought: I need to use a tool.\nAction: {tool.name}\nAction Input: {json.dumps(args)}"

        self.events.append({"agent": role, "event": "no_tool_input"})
        missing = sorted({f for t in agent_tools for f in _required_fields(t)} - provided.keys())
        return _final(json.dumps({"record_found": False, "error": "missing_input", "missing": missing, "facts": []}))

    def supports_function_calling(self) -> bool:
        return False  # force CrewAI's ReAct text protocol

    def supports_stop_words(self) -> bool:
        return False  # we never emit "Observation:" ourselves, so no truncation is needed
