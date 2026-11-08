"""Task 5 — compare chunking strategies with document-level precision and recall.

Retrieved chunks are mapped to their parent doc_id and de-duplicated (first occurrence kept)
before scoring:
    precision = |retrieved_docs ∩ relevant| / |retrieved_docs|
    recall    = |retrieved_docs ∩ relevant| / |relevant|

Run:  python -m eval.chunking_eval
"""

from __future__ import annotations

import json
from fractions import Fraction

import app.config as config
from app.rag.embeddings import Embedder
from app.rag.retriever import Retriever
from eval.queries import EVAL_QUERIES

RESULTS_FILE = config.CHUNKING_EVAL_FILE
RECOMMENDATION_RULE = "highest mean per-query F1; tie -> highest mean recall; tie -> kb_sentence"


def score(retrieved_doc_ids: list[str], relevant: set[str]) -> dict:
    docs = list(dict.fromkeys(retrieved_doc_ids))  # dedup, order preserved
    if not docs or not relevant:
        raise ValueError("retrieved docs and relevant docs must both be non-empty")
    hits = [d for d in docs if d in relevant]
    precision = Fraction(len(hits), len(docs))
    recall = Fraction(len(hits), len(relevant))
    return {
        "retrieved_docs": docs,
        "hits": hits,
        "precision": float(precision),
        "recall": float(recall),
        "precision_expr": f"{len(hits)}/{len(docs)}",
        "recall_expr": f"{len(hits)}/{len(relevant)}",
    }


def evaluate_collection(retriever: Retriever, k: int = config.TOP_K) -> dict:
    rows = []
    for item in EVAL_QUERIES:
        chunks = retriever.retrieve(item["query"], k=k)
        row = score([c.doc_id for c in chunks], item["relevant"])
        row.update(query=item["query"], relevant=sorted(item["relevant"]), chunk_ids=[c.chunk_id for c in chunks])
        rows.append(row)
    n = len(rows)
    for r in rows:
        p, rc = r["precision"], r["recall"]
        r["f1"] = 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)
    return {
        "k": k,
        "rows": rows,
        "mean_precision": sum(r["precision"] for r in rows) / n,
        "mean_recall": sum(r["recall"] for r in rows) / n,
        "mean_f1": sum(r["f1"] for r in rows) / n,
    }


def recommend(collections: dict) -> str:
    """Deterministic rule (see RECOMMENDATION_RULE)."""
    order = {config.COLLECTION_SENTENCE: 1, config.COLLECTION_FIXED: 0}
    return max(
        collections,
        key=lambda n: (round(collections[n]["mean_f1"], 6), round(collections[n]["mean_recall"], 6), order[n]),
    )


def run(embedder: Embedder, client=None) -> dict:
    collections = {name: evaluate_collection(Retriever(name, embedder, client)) for name in config.COLLECTIONS}
    results = {
        "embedder": embedder.name,
        "collections": collections,
        "recommended_collection": recommend(collections),
        "rule": RECOMMENDATION_RULE,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_report(results: dict) -> None:
    collections = results["collections"]
    for name, r in collections.items():
        print(f"===== {name} (top-{r['k']} chunks -> parent docs, deduplicated) =====")
        for i, row in enumerate(r["rows"], 1):
            print(f"Q{i}: {row['query']}")
            print(f"    chunks:    {row['chunk_ids']}")
            print(f"    retrieved: {row['retrieved_docs']}")
            print(f"    relevant:  {row['relevant']}   hits: {row['hits']}")
            print(
                f"    P = {row['precision_expr']} = {row['precision']:.3f}   "
                f"R = {row['recall_expr']} = {row['recall']:.3f}"
            )
        print(
            f"Mean precision = {r['mean_precision']:.3f}   Mean recall = {r['mean_recall']:.3f}   "
            f"Mean F1 = {r['mean_f1']:.3f}\n"
        )

    print("Summary (markdown):")
    print("| Query | Fixed P | Fixed R | Sentence P | Sentence R |")
    print("|---|---|---|---|---|")
    fixed, sent = collections[config.COLLECTION_FIXED], collections[config.COLLECTION_SENTENCE]
    for i, (f, s) in enumerate(zip(fixed["rows"], sent["rows"]), 1):
        print(
            f"| Q{i} | {f['precision_expr']} = {f['precision']:.2f} | {f['recall_expr']} = {f['recall']:.2f} "
            f"| {s['precision_expr']} = {s['precision']:.2f} | {s['recall_expr']} = {s['recall']:.2f} |"
        )
    print(
        f"| **Mean** | {fixed['mean_precision']:.3f} | {fixed['mean_recall']:.3f} "
        f"| {sent['mean_precision']:.3f} | {sent['mean_recall']:.3f} |"
    )
    print(f"\nRecommended collection ({results['rule']}): {results['recommended_collection']}")
    print(f"Written to {RESULTS_FILE.relative_to(config.ROOT_DIR)}")


def main() -> None:
    from app.rag.embeddings import get_default_embedder

    print_report(run(get_default_embedder()))


if __name__ == "__main__":
    main()
