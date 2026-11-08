import pytest

import dataset


def test_meets_structural_thresholds():
    s = dataset.validate(dataset.APPOINTMENTS)
    assert s["n_records"] >= 40
    assert all(c >= 3 for c in s["category_counts"].values())
    assert all(c >= 1 for c in s["status_counts"].values())
    assert 10.0 <= s["follow_up_pct"] <= 30.0


def test_required_fields_and_ranges():
    required = {"record_id", "category", "status", "consultation_fee_inr", "days_since_created", "follow_up_required"}
    for r in dataset.APPOINTMENTS:
        assert required <= r.keys()
        assert 0 <= r["days_since_created"] <= 30
        assert isinstance(r["follow_up_required"], bool)
        low, high = dataset.FEE_BANDS_INR[r["category"]]
        assert low <= r["consultation_fee_inr"] <= high


def test_deterministic_for_seed():
    assert dataset.generate_appointments(dataset.SEED) == dataset.generate_appointments(dataset.SEED)


def test_previous_seed_is_rejected_not_patched():
    # Seed 2026 produced 8% follow-ups; validation must reject it rather than records being edited.
    with pytest.raises(dataset.DatasetValidationError, match="follow_up_required"):
        dataset.validate(dataset.generate_appointments(2026))
