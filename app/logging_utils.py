"""Task 12 — ELK-style structured logging: one JSON-Lines entry per request.

Only masked text is ever passed in: the pipeline masks PII before calling `log_request`, and the
writer re-applies the same masker as defence in depth, so a fixed-format phone number cannot
reach disk in the clear even if a caller forgets.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import app.config as config
from app.guardrails.input_guard import mask_pii


def new_trace_id() -> str:
    return uuid.uuid4().hex


class StageTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.stages: dict[str, float] = {}
        self._mark = self._start

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = round((now - self._mark) * 1000, 2)
        self._mark = now

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


class JsonlRequestLogger:
    def __init__(self, path: Path = config.REQUEST_LOG_FILE):
        self.path = path
        self._lock = threading.Lock()

    def log_request(self, **fields) -> dict:
        entry = {"@timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        for key in ("masked_query", "masked_text"):
            if isinstance(entry.get(key), str):
                entry[key], _ = mask_pii(entry[key])
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return entry
