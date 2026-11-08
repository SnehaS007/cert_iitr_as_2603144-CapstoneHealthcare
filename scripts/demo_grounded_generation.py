"""Task 4 demo — grounded generation on in-scope queries plus one out-of-scope query.

Expectation: every in-scope query is answered from retrieved context; the out-of-scope query
triggers the "I don't know" fallback. Exits non-zero if any expectation fails.

Run:  python -m scripts.demo_grounded_generation
"""

from __future__ import annotations

import sys

import app.config as config
from app.llm.mock_llm import get_generation_llm
from app.rag.embeddings import Embedder
from app.rag.generator import GroundedGenerator
from app.rag.retriever import Retriever
from eval.queries import DEMO_OUT_OF_SCOPE, EVAL_QUERIES


def run(embedder: Embedder, client=None) -> list[str]:
    """Run the demo on both collections; return a list of failed expectations."""
    failures: list[str] = []
    cases = [(q["query"], True) for q in EVAL_QUERIES] + [(DEMO_OUT_OF_SCOPE, False)]
    for name in config.COLLECTIONS:
        generator = GroundedGenerator(
            Retriever(name, embedder, client),
            get_generation_llm(embedder),
            threshold=config.get_grounding_threshold(name),
        )
        print(f"===== Collection: {name} | threshold = {generator.threshold:.4f} | k = {generator.k} =====")
        for query, expect_grounded in cases:
            result = generator.answer(query)
            ok = result.grounded == expect_grounded
            label = "in-scope" if expect_grounded else "OUT-OF-SCOPE"
            print(f"\nQ ({label}): {query}")
            print(
                f"  top-1 similarity {result.top1_similarity:.4f} "
                f"{'>=' if result.grounded else '<'} {result.threshold:.4f} -> "
                f"{'grounded answer' if result.grounded else 'fallback'} [{'PASS' if ok else 'FAIL'}]"
            )
            print("  retrieved: " + ", ".join(f"{c.chunk_id} ({c.similarity:.3f})" for c in result.retrieved))
            print(f"  A: {result.answer}")
            if not ok:
                failures.append(f"{name}: '{query}' expected grounded={expect_grounded}")
        print(f"\nMockGroundedLLM calls for {name}: {generator.llm.calls} (fallbacks never call the LLM)\n")
    return failures


def main() -> None:
    from app.rag.embeddings import get_default_embedder

    failures = run(get_default_embedder())
    if failures:
        print("FAILED expectations:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All expectations met.")


if __name__ == "__main__":
    main()
