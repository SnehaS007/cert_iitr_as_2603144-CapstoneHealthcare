"""Task 7 — the CrewAI crew: Retrieval Agent -> Lookup Agent -> Response Composer (sequential).

Tools and agents are built per request (CrewAI objects hold run state). The Composer's output must
validate against CrewDraft (Task 9): it is declared as output_pydantic AND re-validated in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from crewai import Crew, Process, Task

import app.config as config
from app.governance.budget import TokenBudget
from app.crew.tool_policy import (
    COMPOSER_AGENT,
    LOOKUP_AGENT,
    RETRIEVAL_AGENT,
    build_agent,
    verify_crew,
)
from app.llm.mock_crew_llm import TOOL_INPUT_MARKER, MockCrewLLM
from app.schemas import CrewDraft
from app.tools.appointment_tool import AppointmentLookupTool
from app.rag.cache import GenerationCache
from app.rag.generator import GroundedGenerator
from app.tools.rag_tool import RagLookupTool


class CrewOutputError(ValueError):
    pass


@dataclass
class CrewRun:
    draft: CrewDraft
    rag_result: dict | None
    lookup_result: dict | None
    rag_invocations: list[str]
    lookup_invocations: list[str]
    llm_events: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    llm_input_tokens: int = 0


def make_crew_llm(budget: TokenBudget | None = None) -> MockCrewLLM:
    if not config.MOCK_LLM:
        raise NotImplementedError("MOCK_LLM=false: no real LLM backend is wired in. Set MOCK_LLM=true.")
    return MockCrewLLM(model="mock-crew-llm", budget=budget)


def run_crew(
    query: str,
    record_id: str | None,
    generator: GroundedGenerator,
    cache: GenerationCache,
    budget: TokenBudget | None = None,
) -> CrewRun:
    """Run one crew for a MASKED query. `record_id` comes from the query or session memory."""
    llm = make_crew_llm(budget)
    rag_tool = RagLookupTool(generator=generator, cache=cache)
    lookup_tool = AppointmentLookupTool()

    retrieval_agent = build_agent(
        RETRIEVAL_AGENT,
        goal="Find the clinic-policy answer to the patient's question in the knowledge base.",
        backstory="You answer only from Practo policy documents and never guess.",
        llm=llm,
        tools=[rag_tool],
    )
    lookup_agent = build_agent(
        LOOKUP_AGENT,
        goal="Look up the patient's appointment record when an appointment ID is available.",
        backstory="You are the only agent allowed to read appointment records.",
        llm=llm,
        tools=[lookup_tool],
    )
    composer_agent = build_agent(
        COMPOSER_AGENT,
        goal="Combine the policy answer and the appointment record into one accurate reply.",
        backstory="You state only facts given to you by the other agents.",
        llm=llm,
        tools=[],
    )
    agents = [retrieval_agent, lookup_agent, composer_agent]
    verify_crew(agents)

    retrieval_task = Task(
        description=(
            "Answer the patient's clinic-policy question with the rag_lookup tool and return the tool's "
            f"JSON result unchanged.\n{TOOL_INPUT_MARKER} {json.dumps({'query': query})}"
        ),
        expected_output="The rag_lookup JSON result, unchanged, on one line.",
        agent=retrieval_agent,
    )
    lookup_task = Task(
        description=(
            "If an appointment ID is provided, look it up with check_appointment_status and return the "
            "tool's JSON result unchanged; otherwise report that no ID was provided.\n"
            f"{TOOL_INPUT_MARKER} {json.dumps({'record_id': record_id})}"
        ),
        expected_output="The check_appointment_status JSON result, unchanged, on one line.",
        agent=lookup_agent,
    )
    compose_task = Task(
        description="Combine the policy answer and the appointment information into one draft reply.",
        expected_output="A JSON object matching the CrewDraft schema.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        output_pydantic=CrewDraft,
    )
    crew = Crew(
        agents=agents,
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()

    try:
        draft = CrewDraft.model_validate_json(result.raw)  # Task 9: validate in code, not just declared
    except ValueError as exc:
        raise CrewOutputError(f"Crew output does not match CrewDraft: {exc}") from exc
    if result.pydantic is not None and result.pydantic.model_dump() != draft.model_dump():
        raise CrewOutputError("CrewAI's parsed output and the in-code validation disagree")

    return CrewRun(
        draft=draft,
        rag_result=rag_tool.last_result,
        lookup_result=lookup_tool.last_result,
        rag_invocations=rag_tool.invocations,
        lookup_invocations=lookup_tool.invocations,
        llm_events=list(llm.events),
        llm_calls=llm.calls,
        llm_input_tokens=llm.input_tokens,
    )
