"""Task 13 — evaluate the full pipeline on 15 queries with the MOCK_LLM judge.

Run:  python -m eval.judge_eval   (or via scripts/run_all.py)
"""

from __future__ import annotations

import json

import app.config as config
from app.governance.budget import BudgetExceededError
from app.guardrails.input_guard import InputRejectedError
from app.llm.mock_judge import MockJudgeLLM
from app.pipeline import SupportPipeline
from eval.testset import TESTSET

METRICS = ("accuracy", "grounding", "completeness", "safety")
RESULTS_FILE = config.ARTIFACTS_DIR / "judge_eval.json"


def run(pipeline: SupportPipeline) -> dict:
    judge = MockJudgeLLM()
    rows = []
    for case in TESTSET:
        facts: list[str] = []
        try:
            resp = pipeline.ask(case["query"], session_id=f"eval-{case['id']}", endpoint="eval")
            behaviour = "refuse" if resp.refusal_reason else "answer"
            answer, sources = resp.answer, resp.sources
            facts = pipeline.last_trace.lookup_facts
        except InputRejectedError:
            behaviour, answer, sources = "reject", "(rejected by input guardrail)", []
        except BudgetExceededError:
            behaviour, answer, sources = "reject", "(rejected by budget)", []
        prompt = judge.render(case, behaviour, answer, sources)
        scores = judge.judge(prompt, case, behaviour, answer, sources, facts)
        rows.append({**{k: case[k] for k in ("id", "topic", "query", "expected")}, "behaviour": behaviour,
                     "answer": answer, "sources": sources, **scores})
    averages = {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}
    results = {"n": len(rows), "rows": rows, "averages": averages, "judge_calls": judge.calls,
               "sample_prompt": judge.render(TESTSET[0], rows[0]["behaviour"], rows[0]["answer"], rows[0]["sources"])}
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_report(results: dict) -> None:
    print("Judge prompt (rendered for T01, scored by MockJudgeLLM under MOCK_LLM):")
    print(results["sample_prompt"] + "\n")
    print("| ID | Topic | Behaviour | Accuracy | Grounding | Completeness | Safety |")
    print("|---|---|---|---|---|---|---|")
    for r in results["rows"]:
        print(f"| {r['id']} | {r['topic']} | {r['behaviour']} (exp. {r['expected']}) | {r['accuracy']:.2f} | "
              f"{r['grounding']:.2f} | {r['completeness']:.2f} | {r['safety']:.2f} |")
    a = results["averages"]
    print(f"| **Average (n={results['n']})** | | | **{a['accuracy']:.3f}** | **{a['grounding']:.3f}** | "
          f"**{a['completeness']:.3f}** | **{a['safety']:.3f}** |")
    print("\nPer-query answers:")
    for r in results["rows"]:
        print(f"  {r['id']} {r['query']}\n      -> {r['answer'][:220]}")
