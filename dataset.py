"""Task 1 — Seeded, deterministic appointment dataset for the Practo (Healthcare) track.

All records are fabricated. No patient-identifying fields are generated.

Run:  python dataset.py
"""

from __future__ import annotations

import random
from collections import Counter

# ---------------------------------------------------------------------------
# Design choices (mirrored in README.md section 0)
# ---------------------------------------------------------------------------
SEED = 2027
N_RECORDS = 50

CATEGORY_WEIGHTS: dict[str, float] = {
    "General Medicine": 0.30,
    "Dermatology": 0.20,
    "Pediatrics": 0.20,
    "Cardiology": 0.15,
    "Orthopedics": 0.15,
}

STATUS_WEIGHTS: dict[str, float] = {
    "Completed": 0.35,
    "Scheduled": 0.30,
    "Rescheduled": 0.13,
    "Cancelled": 0.12,
    "No-Show": 0.10,
}

# Consultation fee band per specialty, INR (inclusive). Must match kb/consultation_fee_structure.md.
# Reasoning: typical Indian private outpatient consultations run from a few hundred rupees for a
# general physician up to roughly two thousand rupees for a cardiologist.
FEE_BANDS_INR: dict[str, tuple[int, int]] = {
    "General Medicine": (300, 700),
    "Pediatrics": (400, 900),
    "Dermatology": (500, 1200),
    "Orthopedics": (600, 1500),
    "Cardiology": (800, 2000),
}
FEE_ROUNDING_INR = 50

FOLLOW_UP_PROBABILITY = 0.20
MAX_DAYS_SINCE_CREATED = 30

# Structural thresholds required by the brief
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
FOLLOW_UP_PCT_BAND = (10.0, 30.0)


class DatasetValidationError(Exception):
    """Raised when the generated dataset misses a structural threshold.

    Fix by changing SEED or the weights above and regenerating — never by editing records.
    """


def _round_fee(value: int) -> int:
    return int(round(value / FEE_ROUNDING_INR) * FEE_ROUNDING_INR)


def generate_appointments(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    """Generate `n` appointment records deterministically from `seed`."""
    rng = random.Random(seed)  # local RNG: no global random state is touched
    categories = list(CATEGORY_WEIGHTS)
    statuses = list(STATUS_WEIGHTS)

    records: list[dict] = []
    for i in range(n):
        category = rng.choices(categories, weights=list(CATEGORY_WEIGHTS.values()), k=1)[0]
        status = rng.choices(statuses, weights=list(STATUS_WEIGHTS.values()), k=1)[0]
        low, high = FEE_BANDS_INR[category]
        records.append(
            {
                "record_id": f"APT-{1001 + i}",
                "category": category,
                "status": status,
                "consultation_fee_inr": _round_fee(rng.randint(low, high)),
                "days_since_created": rng.randint(0, MAX_DAYS_SINCE_CREATED),
                "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    category_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    follow_up = sum(r["follow_up_required"] for r in records)
    return {
        "n_records": len(records),
        "category_counts": {c: category_counts.get(c, 0) for c in CATEGORY_WEIGHTS},
        "status_counts": {s: status_counts.get(s, 0) for s in STATUS_WEIGHTS},
        "follow_up_count": follow_up,
        "follow_up_pct": 100.0 * follow_up / len(records),
    }


def validate(records: list[dict]) -> dict:
    """Check every structural threshold; raise DatasetValidationError listing all failures."""
    summary = summarize(records)
    failures: list[str] = []

    if summary["n_records"] < 40:
        failures.append(f"only {summary['n_records']} records (need >= 40)")
    for category, count in summary["category_counts"].items():
        if count < MIN_PER_CATEGORY:
            failures.append(f"category '{category}' has {count} records (need >= {MIN_PER_CATEGORY})")
    for status, count in summary["status_counts"].items():
        if count < MIN_PER_STATUS:
            failures.append(f"status '{status}' has {count} records (need >= {MIN_PER_STATUS})")
    low, high = FOLLOW_UP_PCT_BAND
    if not low <= summary["follow_up_pct"] <= high:
        failures.append(f"follow_up_required is {summary['follow_up_pct']:.1f}% (need {low}-{high}%)")

    for r in records:
        band_low, band_high = FEE_BANDS_INR[r["category"]]
        if not band_low <= r["consultation_fee_inr"] <= band_high:
            failures.append(f"{r['record_id']} fee {r['consultation_fee_inr']} outside {r['category']} band")
        if not 0 <= r["days_since_created"] <= MAX_DAYS_SINCE_CREATED:
            failures.append(f"{r['record_id']} days_since_created out of range")

    if len({r["record_id"] for r in records}) != len(records):
        failures.append("duplicate record_id values")

    if failures:
        raise DatasetValidationError("; ".join(failures))
    return summary


# Module-level dataset consumed by the tools in Part 2. Validation runs on import so an
# invalid dataset can never be used silently.
APPOINTMENTS: list[dict] = generate_appointments()
SUMMARY: dict = validate(APPOINTMENTS)
APPOINTMENTS_BY_ID: dict[str, dict] = {r["record_id"]: r for r in APPOINTMENTS}


def main() -> None:
    print(f"Dataset: seed={SEED}, n={N_RECORDS}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print(f"Status weights:   {STATUS_WEIGHTS}")
    print(f"Fee bands (INR):  {FEE_BANDS_INR} (rounded to {FEE_ROUNDING_INR})")
    print(f"follow_up_required probability: {FOLLOW_UP_PROBABILITY}\n")

    print("Count per category (each must be >= 3):")
    for category, count in SUMMARY["category_counts"].items():
        print(f"  {category:<17} {count}")
    print("\nCount per status (each must be >= 1):")
    for status, count in SUMMARY["status_counts"].items():
        print(f"  {status:<17} {count}")
    print(
        f"\nfollow_up_required=True: {SUMMARY['follow_up_count']}/{SUMMARY['n_records']} "
        f"= {SUMMARY['follow_up_pct']:.1f}% (band 10-30%)"
    )
    fees = [r["consultation_fee_inr"] for r in APPOINTMENTS]
    print(f"Fee range observed: INR {min(fees)}-{max(fees)}")
    print("\nValidation: PASSED (all structural thresholds met)\n")

    print(f"{'record_id':<10} {'category':<17} {'status':<12} {'fee_inr':>7} {'days':>5} follow_up")
    for r in APPOINTMENTS:
        print(
            f"{r['record_id']:<10} {r['category']:<17} {r['status']:<12} "
            f"{r['consultation_fee_inr']:>7} {r['days_since_created']:>5} {r['follow_up_required']}"
        )


if __name__ == "__main__":
    main()
