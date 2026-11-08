"""Run every task end to end under MOCK_LLM and write all transcripts to transcripts/.

Run:  python -m scripts.run_all
Order matters: Part 1 builds the index, calibrates the threshold and picks the collection that
Parts 2-4 use as a fixed input.
"""

from __future__ import annotations

import app.config as config
from scripts import demos
from scripts.run_part1 import _run_step


def run_parts_2_to_4(embedder, client=None, log_file=None) -> None:
    from app.rag.indexer import get_client
    from eval import judge_eval

    collection = config.get_recommended_collection()
    env = demos.DemoEnv(
        embedder=embedder,
        client=client or get_client(),
        collection=collection,
        threshold=config.get_grounding_threshold(collection),
        log_file=log_file or config.REQUEST_LOG_FILE,
    )
    env.log_file.parent.mkdir(parents=True, exist_ok=True)
    env.log_file.write_text("", encoding="utf-8")  # one clean log per full run
    print(f"Parts 2-4 use collection '{collection}' with threshold {env.threshold}\n")

    steps = [
        ("06_lookup_tool.txt", "Task 6 - Lookup tool and escalation score", demos.demo_lookup_tool),
        ("07_crew_kickoff.txt", "Task 7 - CrewAI crew via kickoff()", demos.demo_crew),
        ("08a_memory_multiturn.txt", "Task 8 - Session memory: multi-turn", demos.demo_memory_multiturn),
        ("08b_memory_fresh.txt", "Task 8 - Session memory: fresh session", demos.demo_memory_fresh),
        ("09_schema_validation.txt", "Task 9 - Structured output schema", demos.demo_schema),
        ("10_guardrails.txt", "Task 10 - Guardrails", demos.demo_guardrails),
        ("11_api.txt", "Task 11 - FastAPI HTTP and WebSocket", demos.demo_api),
        ("13_evaluation.txt", "Task 13 - Evaluation (15 queries, MOCK_LLM judge)",
         lambda e: judge_eval.print_report(judge_eval.run(e.pipeline()))),
        ("14_review.txt", "Task 14 - AutoGen review stage", demos.demo_review),
        ("15_governance.txt", "Task 15 - Governance", demos.demo_governance),
        ("16_cache.txt", "Task 16 - Response caching", demos.demo_cache),
        ("12_structured_logging.txt", "Task 12 - Structured JSONL logging", demos.demo_logging),
    ]
    for filename, title, func in steps:
        _run_step(filename, title, lambda f=func: f(env))


def main() -> None:
    from app.rag.embeddings import get_default_embedder
    from scripts import run_part1

    run_part1.main()
    run_parts_2_to_4(get_default_embedder())


if __name__ == "__main__":
    main()
