"""Central configuration. Import this module before chromadb / crewai / sentence_transformers.

Telemetry and Hub-access variables are forced here (setdefault, so an explicit user value wins)
so that a script cannot accidentally make outbound calls by forgetting to source .env.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Supported Python ---------------------------------------------------------------------------
# crewai declares requires_python "<3.14,>=3.10"; tested on 3.12. Fail fast with a clear message
# instead of an obscure import error later.
SUPPORTED_PYTHON = ((3, 11), (3, 14))  # [min, max)


def check_python_version(version: tuple[int, int] | None = None) -> None:
    current = version or sys.version_info[:2]
    low, high = SUPPORTED_PYTHON
    if not low <= tuple(current) < high:
        raise RuntimeError(
            f"Python {current[0]}.{current[1]} is not supported. Use Python 3.11-3.13 "
            f"(tested on 3.12; crewai does not support 3.14+). Recreate the venv, e.g. "
            f"`py -3.12 -m venv .venvpracto`."
        )


check_python_version()

# --- Zero-network defaults (set before any third-party import) ---------------------------------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# HF_HUB_OFFLINE is NOT forced here: scripts/download_model.py needs the Hub once.
# Set HF_HUB_OFFLINE=1 in .env after the model is cached.

# --- Mode ---------------------------------------------------------------------------------------
MOCK_LLM: bool = os.environ.get("MOCK_LLM", "true").strip().lower() == "true"

# --- Paths --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "kb"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"
CALIBRATION_FILE = ARTIFACTS_DIR / "calibration.json"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
CHUNKING_EVAL_FILE = ARTIFACTS_DIR / "chunking_eval.json"
LOG_DIR = ROOT_DIR / "logs"
REQUEST_LOG_FILE = LOG_DIR / "requests.jsonl"

# --- Embeddings ---------------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking (Task 3) --------------------------------------------------------------------------
FIXED_CHUNK_WORDS = 40      # window size in words
FIXED_OVERLAP_WORDS = 10    # words shared between consecutive windows
SENTENCES_PER_CHUNK = 1     # sentence-based strategy

COLLECTION_FIXED = "kb_fixed"
COLLECTION_SENTENCE = "kb_sentence"
COLLECTIONS = (COLLECTION_FIXED, COLLECTION_SENTENCE)

# --- Retrieval / generation (Task 4) ------------------------------------------------------------
TOP_K = 3
MAX_ANSWER_SENTENCES = 2
FALLBACK_ANSWER = (
    "I don't know. I couldn't find this in the clinic's policy documents. "
    "Please contact Practo support for help with this question."
)


def get_grounding_threshold(collection: str) -> float:
    """Return the empirically calibrated top-1 similarity threshold for `collection`.

    Raises if calibration has not been run — there is deliberately no default value.
    """
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"{CALIBRATION_FILE} not found. Run `python -m eval.calibrate_threshold` first."
        )
    data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if collection not in data:
        raise KeyError(f"No calibrated threshold for collection '{collection}' in {CALIBRATION_FILE}.")
    calibrated_with = data[collection].get("embedder")
    if calibrated_with != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Threshold for '{collection}' was calibrated with '{calibrated_with}', but EMBEDDING_MODEL "
            f"is '{EMBEDDING_MODEL}'. Re-run the indexer and `python -m eval.calibrate_threshold`."
        )
    return float(data[collection]["threshold"])


def get_recommended_collection() -> str:
    """Collection recommended by Task 5 (written by eval/chunking_eval.py). Raises if not evaluated yet."""
    if not CHUNKING_EVAL_FILE.exists():
        raise FileNotFoundError(f"{CHUNKING_EVAL_FILE} not found. Run `python -m eval.chunking_eval` first.")
    data = json.loads(CHUNKING_EVAL_FILE.read_text(encoding="utf-8"))
    name = data.get("recommended_collection")
    if name not in COLLECTIONS:
        raise ValueError(f"Invalid recommended_collection {name!r} in {CHUNKING_EVAL_FILE}")
    return name


# --- Governance: runtime budget (Task 15) ------------------------------------------------------
# Unit: total LLM input tokens consumed by one request's crew run (all agent calls combined).
REQUEST_TOKEN_BUDGET = int(os.environ.get("REQUEST_TOKEN_BUDGET", "6000"))
# Calibrated by measurement (scripts/demo_governance.py prints the live numbers): CrewAI system prompts,
# tool schemas and inter-task context cost ~2,500-4,200 tokens per request independent of the query,
# and the query text is re-sent in roughly 2-3 prompts.
CREW_PROMPT_OVERHEAD_TOKENS = 4300
QUERY_TOKEN_FANOUT = 3
MOCK_COST_PER_1K_TOKENS_USD = 0.003  # illustrative rate, used for the cost estimate only

# --- Session memory (Task 8) -------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 10  # most recent messages scanned for record-id resolution

# --- Input limits (API) ------------------------------------------------------------------------
MAX_QUERY_CHARS = 20_000  # hard schema bound; the token budget is the governance control
