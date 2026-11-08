"""Task 4 — calibrate the "I don't know" threshold from measured top-1 cosine similarities.

threshold = midpoint between the lowest in-scope and the highest out-of-scope top-1 similarity.
If the two clusters overlap there is no safe threshold, and this script raises instead of guessing.

Run:  python -m eval.calibrate_threshold
"""

from __future__ import annotations

import json

import app.config as config
from app.rag.embeddings import Embedder
from app.rag.retriever import Retriever
from eval.queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE


class CalibrationError(Exception):
    pass


def calibrate_collection(retriever: Retriever) -> dict:
    in_scope = [(q, retriever.retrieve(q, k=1)[0].similarity) for q in CALIBRATION_IN_SCOPE]
    out_scope = [(q, retriever.retrieve(q, k=1)[0].similarity) for q in CALIBRATION_OUT_OF_SCOPE]
    min_in = min(s for _, s in in_scope)
    max_out = max(s for _, s in out_scope)
    if max_out >= min_in:
        raise CalibrationError(
            f"[{retriever.collection_name}] clusters overlap: highest out-of-scope {max_out:.4f} >= "
            f"lowest in-scope {min_in:.4f}. Revise the KB or queries; no threshold separates them."
        )
    return {
        "in_scope": [{"query": q, "top1_similarity": round(s, 4)} for q, s in in_scope],
        "out_of_scope": [{"query": q, "top1_similarity": round(s, 4)} for q, s in out_scope],
        "min_in_scope": round(min_in, 4),
        "max_out_of_scope": round(max_out, 4),
        "margin": round(min_in - max_out, 4),
        "threshold": round((min_in + max_out) / 2, 4),
    }


def run(embedder: Embedder, client=None) -> dict:
    results = {
        name: calibrate_collection(Retriever(name, embedder, client)) for name in config.COLLECTIONS
    }
    for name, r in results.items():
        r["embedder"] = embedder.name
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_report(results: dict) -> None:
    for name, r in results.items():
        print(f"[{name}] top-1 cosine similarity")
        print(f"  {'type':<13} {'similarity':>10}  query")
        for row in r["in_scope"]:
            print(f"  {'in-scope':<13} {row['top1_similarity']:>10.4f}  {row['query']}")
        for row in r["out_of_scope"]:
            print(f"  {'out-of-scope':<13} {row['top1_similarity']:>10.4f}  {row['query']}")
        print(
            f"  lowest in-scope = {r['min_in_scope']:.4f}, highest out-of-scope = "
            f"{r['max_out_of_scope']:.4f}, margin = {r['margin']:.4f}"
        )
        print(f"  chosen threshold (midpoint) = {r['threshold']:.4f}\n")
    print(f"Written to {config.CALIBRATION_FILE.relative_to(config.ROOT_DIR)}")


def main() -> None:
    from app.rag.embeddings import get_default_embedder

    print_report(run(get_default_embedder()))


if __name__ == "__main__":
    main()
