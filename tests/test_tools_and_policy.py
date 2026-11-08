import json

import pytest

import dataset
from app.crew.tool_policy import (
    COMPOSER_AGENT, LOOKUP_AGENT, RETRIEVAL_AGENT, ToolPolicyViolation, build_agent, verify_crew,
)
from app.llm.mock_crew_llm import MockCrewLLM
from app.tools import appointment_tool as at


def test_escalation_formula_and_threshold():
    assert at.escalation_score(True, 30) == 1.0 and at.escalation_score(False, 0) == 0.0
    assert at.escalation_score(True, 15) == pytest.approx(0.8)
    assert at.ESCALATION_THRESHOLD == pytest.approx(0.6 + 0.4 * at.FOLLOW_UP_DAYS_P50 / 30, abs=1e-4)
    assert at.FOLLOW_UP_DAYS_P50 == 14.0  # seed 2027


def test_only_follow_up_records_escalate_and_score_is_not_boolean_or():
    for r in dataset.APPOINTMENTS:
        s = at.check_appointment_status(r["record_id"])
        assert 0.0 <= s["escalation_score"] <= 1.0
        if s["escalation_recommended"]:
            assert r["follow_up_required"] and r["days_since_created"] >= at.FOLLOW_UP_DAYS_P50
    fresh_follow_up = at.escalation_score(True, 0)
    assert fresh_follow_up < at.ESCALATION_THRESHOLD  # follow-up alone does not escalate


def test_unknown_record_raises_and_tool_reports_not_found():
    with pytest.raises(at.RecordNotFoundError):
        at.check_appointment_status("APT-9999")
    out = json.loads(at.AppointmentLookupTool()._run("APT-9999"))
    assert out["record_found"] is False and out["error"] == "record_not_found"


@pytest.mark.parametrize("role", [COMPOSER_AGENT, RETRIEVAL_AGENT, "Some New Agent"])
def test_lookup_tool_blocked_for_every_agent_but_lookup(role):
    with pytest.raises(ToolPolicyViolation):
        build_agent(role, "g", "b", MockCrewLLM(model="m"), tools=[at.AppointmentLookupTool()])


def test_lookup_agent_allowed_and_delegation_checked():
    agent = build_agent(LOOKUP_AGENT, "g", "b", MockCrewLLM(model="m"), tools=[at.AppointmentLookupTool()])
    verify_crew([agent])
    agent.allow_delegation = True
    with pytest.raises(ToolPolicyViolation):
        verify_crew([agent])
