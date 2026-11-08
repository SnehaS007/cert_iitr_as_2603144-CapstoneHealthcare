"""Task 15 (Application layer) — principle of least autonomy for tools.

Every agent is built through `build_agent`, which checks each requested tool against TOOL_POLICY and
raises ToolPolicyViolation before the agent exists. `verify_crew` re-checks a whole crew, and
delegation is disabled on every agent, so no agent can reach another agent's tool indirectly.
"""

from __future__ import annotations

from crewai import Agent

RETRIEVAL_AGENT = "Retrieval Agent"
LOOKUP_AGENT = "Lookup Agent"
COMPOSER_AGENT = "Response Composer"

# tool name -> roles allowed to hold it. Anything not listed is denied.
TOOL_POLICY: dict[str, frozenset[str]] = {
    "rag_lookup": frozenset({RETRIEVAL_AGENT}),
    "check_appointment_status": frozenset({LOOKUP_AGENT}),
}


class ToolPolicyViolation(PermissionError):
    pass


def assert_tools_allowed(role: str, tools: list) -> None:
    for tool in tools:
        allowed = TOOL_POLICY.get(tool.name, frozenset())
        if role not in allowed:
            raise ToolPolicyViolation(
                f"Agent '{role}' may not hold tool '{tool.name}' (allowed: {sorted(allowed) or 'nobody'})"
            )


def build_agent(role: str, goal: str, backstory: str, llm, tools: list | None = None) -> Agent:
    tools = list(tools or [])
    assert_tools_allowed(role, tools)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
        max_retry_limit=0,  # a budget stop or validation failure must not be retried silently
    )


def verify_crew(agents: list[Agent]) -> None:
    for agent in agents:
        if agent.allow_delegation:
            raise ToolPolicyViolation(f"Agent '{agent.role}' has delegation enabled")
        assert_tools_allowed(agent.role, list(agent.tools or []))
