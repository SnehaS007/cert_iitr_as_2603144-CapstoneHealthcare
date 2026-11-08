"""Task 6 — appointment lookup tool with a designed escalation score.

escalation_score = W_FOLLOW_UP * follow_up_required + W_STALENESS * (days_since_created / 30)

- follow_up_required (0/1) carries most weight: escalation is about patients who still need care.
- Staleness (days_since_created / 30, fixed domain max, so the scale does not move with the data)
  ranks them: an older appointment that still needs follow-up is more urgent.
  ASSUMPTION (direction not yet confirmed by the project owner): older = more urgent.
  Set STALENESS_INCREASES_RISK = False to make newer appointments score higher instead.

Threshold = W_FOLLOW_UP + W_STALENESS * (P50 / 30), where P50 is the median days_since_created
among follow-up records in the generated dataset. A non-follow-up record can score at most 0.4, so
only follow-up records escalate, and only those at least as old as the median follow-up record.
For seed 2027 the median is 14 days, which also matches the KB follow-up window: discounted
follow-ups end at 14 days (kb/follow_up_discount_policy.md).
"""

from __future__ import annotations

import json

import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

import dataset
from app.schemas import RECORD_ID_PATTERN, AppointmentStatus

W_FOLLOW_UP = 0.6
W_STALENESS = 0.4
MAX_DAYS = dataset.MAX_DAYS_SINCE_CREATED
STALENESS_INCREASES_RISK = True
THRESHOLD_PERCENTILE = 50

_follow_up_days = [r["days_since_created"] for r in dataset.APPOINTMENTS if r["follow_up_required"]]
if not _follow_up_days:
    raise ValueError("Dataset has no follow-up records; escalation threshold cannot be derived.")
FOLLOW_UP_DAYS_P50: float = float(np.percentile(_follow_up_days, THRESHOLD_PERCENTILE))
ESCALATION_THRESHOLD: float = round(W_FOLLOW_UP + W_STALENESS * (FOLLOW_UP_DAYS_P50 / MAX_DAYS), 4)


class RecordNotFoundError(KeyError):
    pass


def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    if not 0 <= days_since_created <= MAX_DAYS:
        raise ValueError(f"days_since_created must be in [0, {MAX_DAYS}], got {days_since_created}")
    staleness = days_since_created / MAX_DAYS
    recency_signal = staleness if STALENESS_INCREASES_RISK else 1.0 - staleness
    return round(W_FOLLOW_UP * float(follow_up_required) + W_STALENESS * recency_signal, 4)


def check_appointment_status(record_id: str) -> dict:
    """Look up one appointment. Raises RecordNotFoundError for an unknown id (no silent default)."""
    record = dataset.APPOINTMENTS_BY_ID.get(record_id.strip().upper())
    if record is None:
        raise RecordNotFoundError(record_id)
    score = escalation_score(record["follow_up_required"], record["days_since_created"])
    return AppointmentStatus(
        **record,
        escalation_score=score,
        escalation_threshold=ESCALATION_THRESHOLD,
        escalation_recommended=score >= ESCALATION_THRESHOLD,
    ).model_dump()


def appointment_facts(status: dict) -> list[str]:
    """Plain-language facts derived only from the record; the Composer may state only these."""
    facts = [
        f"Appointment {status['record_id']} ({status['category']}) is currently {status['status']}.",
        f"The consultation fee for appointment {status['record_id']} is INR {status['consultation_fee_inr']:,}.",
    ]
    if status["follow_up_required"]:
        facts.append(f"The doctor marked a follow-up visit as required for appointment {status['record_id']}.")
    if status["escalation_recommended"]:
        facts.append("This appointment has been flagged for a call from our patient-care team.")
    return facts


class AppointmentLookupArgs(BaseModel):
    record_id: str = Field(..., pattern=RECORD_ID_PATTERN, description="Appointment ID, format APT-1234")


class AppointmentLookupTool(BaseTool):
    """CrewAI wrapper. Least autonomy: only the Lookup Agent may hold this tool (app/crew/tool_policy.py)."""

    name: str = "check_appointment_status"
    description: str = "Look up one appointment's status, consultation fee and escalation score by record ID."
    args_schema: type[BaseModel] = AppointmentLookupArgs
    _invocations: list[str] = PrivateAttr(default_factory=list)
    _last_result: dict | None = PrivateAttr(default=None)

    @property
    def invocations(self) -> list[str]:
        return list(self._invocations)

    @property
    def last_result(self) -> dict | None:
        return self._last_result

    def _run(self, record_id: str) -> str:
        self._invocations.append(record_id)
        try:
            status = check_appointment_status(record_id)
            result = {"record_found": True, "appointment": status, "facts": appointment_facts(status)}
        except RecordNotFoundError:
            # Surfaced as an explicit, structured "not found" result (not swallowed): the Composer
            # tells the patient the ID was not found.
            result = {
                "record_found": False,
                "error": "record_not_found",
                "record_id": record_id,
                "facts": [f"No appointment was found with ID {record_id}. Please check the ID and try again."],
            }
        self._last_result = result
        return json.dumps(result)
