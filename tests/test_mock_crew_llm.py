"""The two silent pitfalls named in the brief."""

from types import SimpleNamespace

from app.llm.mock_crew_llm import MockCrewLLM
from app.tools.appointment_tool import AppointmentLookupTool
from app.tools.rag_tool import RagLookupArgs

SYSTEM = ("Use the format:\nThought: ...\nAction: the action to take\nAction Input: the input\n"
          "Observation: the result of the action\nFinal Answer: ...")


class _RagLike:  # a tool literally named rag_lookup whose schema needs `query`
    name = "rag_lookup"
    args_schema = RagLookupArgs


def _agent(*tools):
    return SimpleNamespace(role="Tester", tools=list(tools))


def test_pitfall1_template_observation_is_not_treated_as_tool_result():
    llm = MockCrewLLM(model="m")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": 'Do it.\nTOOL_INPUT_JSON: {"query": "fees"}'}]
    out = llm.call(msgs, from_agent=_agent(_RagLike()))
    assert out.startswith("Thought: I need to use a tool.") and "Final Answer" not in out


def test_pitfall1_observation_read_from_own_assistant_message():
    llm = MockCrewLLM(model="m")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": 'TOOL_INPUT_JSON: {"query": "fees"}'},
            {"role": "assistant", "content": 'Action: rag_lookup\nAction Input: {"query": "fees"}\nObservation: {"ok": 1}'}]
    assert llm.call(msgs, from_agent=_agent(_RagLike())).endswith('Final Answer: {"ok": 1}')


def test_pitfall2_dispatch_by_schema_not_by_name():
    # "rag_lookup" contains "lookup"; with only a record_id available the schema must pick the record tool.
    llm = MockCrewLLM(model="m")
    msgs = [{"role": "user", "content": 'TOOL_INPUT_JSON: {"record_id": "APT-1001"}'}]
    out = llm.call(msgs, from_agent=_agent(_RagLike(), AppointmentLookupTool()))
    assert "Action: check_appointment_status" in out and '"record_id": "APT-1001"' in out


def test_no_matching_input_means_no_tool_call():
    llm = MockCrewLLM(model="m")
    out = llm.call([{"role": "user", "content": 'TOOL_INPUT_JSON: {"record_id": null}'}],
                   from_agent=_agent(AppointmentLookupTool()))
    assert "Final Answer" in out and "Action:" not in out
