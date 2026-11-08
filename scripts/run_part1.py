"""Run every Part 1 task in order and write its transcript to transcripts/.

Run:  python -m scripts.run_part1
"""

from __future__ import annotations

import contextlib
import sys

import app.config as config


class _Tee:
    """Write to the console and a transcript file at the same time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _run_step(filename: str, title: str, func) -> None:
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRANSCRIPTS_DIR / filename
    with path.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(_Tee(sys.__stdout__, fh)):
        print(f"### {title}")
        print(f"### MOCK_LLM={config.MOCK_LLM} | embedder={config.EMBEDDING_MODEL}\n")
        func()
    print(f"--> transcript: {path.relative_to(config.ROOT_DIR)}\n")


def main() -> None:
    import dataset
    from app.rag import indexer
    from eval import calibrate_threshold, chunking_eval
    from scripts import demo_grounded_generation

    _run_step("01_dataset.txt", "Task 1 - Dataset design and validation", dataset.main)
    _run_step("03_indexing.txt", "Task 3 - Chunking, embedding, indexing", indexer.main)
    _run_step("04a_threshold_calibration.txt", "Task 4 - Threshold calibration", calibrate_threshold.main)
    _run_step("04b_grounded_generation.txt", "Task 4 - Grounded generation demo", demo_grounded_generation.main)
    _run_step("05_chunking_eval.txt", "Task 5 - Chunking strategy evaluation", chunking_eval.main)


if __name__ == "__main__":
    main()
