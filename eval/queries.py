"""Query sets for Part 1.

The calibration set is deliberately DISJOINT from the demo/evaluation set, so the Task 4 demo
tests the threshold on queries it was not tuned on.
"""

# Task 4 calibration: >=3 in-scope and >=2 out-of-scope queries.
CALIBRATION_IN_SCOPE: list[str] = [
    "What is the process for getting a second opinion?",
    "How long does a biopsy report take?",
    "Can I get my prescription refilled without seeing the doctor again?",
    "What should I do if I have severe chest pain?",
    "How far ahead can I book an appointment?",
]
CALIBRATION_OUT_OF_SCOPE: list[str] = [
    "What is the capital of Australia?",
    "Recommend a good gaming laptop under 80000 rupees.",
    "How do I change a flat tyre on my car?",
]

# Task 4 demo + Task 5 evaluation: in-scope queries with gold (relevant) parent documents.
EVAL_QUERIES: list[dict] = [
    {
        "query": "How much does a cardiology consultation cost?",
        "relevant": {"consultation_fee_structure"},
    },
    {
        "query": "Can I reschedule my appointment, and is there a charge if I do it late?",
        "relevant": {"cancellation_rescheduling_window"},
    },
    {
        "query": "When will my lipid profile blood test results be ready?",
        "relevant": {"lab_test_turnaround"},
    },
    {
        "query": "Is my follow-up visit discounted if I go back to the same doctor after 10 days?",
        "relevant": {"follow_up_discount_policy"},
    },
    {
        "query": "My elderly father cannot walk well. Can a doctor visit him at home, or can he use a video consultation?",
        "relevant": {"home_visit_eligibility", "telemedicine_eligibility"},
    },
    {
        "query": "How do I claim insurance for my consultation, and who can see my medical records?",
        "relevant": {"insurance_claim_process", "patient_data_privacy"},
    },
]

# Task 4: deliberately out-of-scope query that must trigger the fallback (not used in calibration).
DEMO_OUT_OF_SCOPE: str = "What is a good recipe for butter chicken?"
