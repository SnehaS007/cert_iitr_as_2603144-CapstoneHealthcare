import re

import dataset
from app.rag.kb import load_kb

REQUIRED_DOC_IDS = {
    "appointment_booking_policy", "cancellation_rescheduling_window", "consultation_fee_structure",
    "insurance_claim_process", "prescription_refill_policy", "lab_test_turnaround",
    "telemedicine_eligibility", "emergency_visit_protocol", "patient_data_privacy",
    "follow_up_discount_policy", "second_opinion_process", "home_visit_eligibility",
}


def test_all_required_topics_present():
    assert REQUIRED_DOC_IDS <= {d.doc_id for d in load_kb()}


def test_each_document_has_2_to_5_sentences():
    for doc in load_kb():
        assert 2 <= len(doc.sentences) <= 5, (doc.doc_id, len(doc.sentences))


def test_no_abbreviations_that_break_sentence_splitting():
    for doc in load_kb():
        assert not re.search(r"\b(e\.g|i\.e|Dr|Mr|Mrs|etc)\.", doc.text), doc.doc_id


def test_fee_document_matches_dataset_bands():
    fee_doc = next(d for d in load_kb() if d.doc_id == "consultation_fee_structure")
    for category, (low, high) in dataset.FEE_BANDS_INR.items():
        sentence = next(s for s in fee_doc.sentences if category in s)
        amounts = [int(a) for a in re.findall(r"INR (\d+)", sentence.replace(",", ""))]
        assert low in amounts and high in amounts, (category, amounts)
