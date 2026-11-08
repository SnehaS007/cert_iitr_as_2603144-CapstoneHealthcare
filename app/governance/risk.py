"""Task 15 — risk classification (Low / Medium / High scheme from the brief)."""

RISK_LEVEL = "High"

RISK_JUSTIFICATION = (
    "The system reads appointment records tied to identifiable patients (status, specialty, fees, "
    "follow-up need) and answers questions about clinical services such as emergency protocol, "
    "prescription refills and telemedicine eligibility. The brief's scheme lists medical data as High "
    "risk. A wrong or ungrounded answer could delay a patient seeking care, and a data leak would be a "
    "health-privacy breach. High risk is why the system masks PII before any processing or logging, "
    "refuses when retrieval does not support an answer, routes every answer through an independent "
    "review team, restricts record access to one agent, caps per-request spend, and writes an audit log."
)

CONTROLS = {
    "data": "Fixed-format PII (phone numbers) masked before agents, memory, cache and logs",
    "model": "Deterministic MOCK_LLM; extractive answers; groundedness gate; AutoGen review",
    "application": "Least autonomy: only the Lookup Agent holds check_appointment_status; no delegation",
    "runtime": "Per-request token/cost cap; JSONL audit log with trace IDs; response cache",
}
