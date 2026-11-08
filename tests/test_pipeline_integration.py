"""End-to-end Part 1 pipeline on the real KB with the test embedder and a real (temp) ChromaDB."""

import pytest

import app.config as config
from app.rag.chunking import chunk_corpus
from app.rag.indexer import build_indexes, get_client
from app.rag.kb import load_kb
from app.rag.retriever import Retriever
from eval import calibrate_threshold, chunking_eval
from scripts import demo_grounded_generation
from tests.fakes import HashingEmbedder


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "CALIBRATION_FILE", tmp_path / "calibration.json")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", HashingEmbedder.name)
    monkeypatch.setattr(chunking_eval, "RESULTS_FILE", tmp_path / "chunking_eval.json")
    monkeypatch.setattr(config, "CHUNKING_EVAL_FILE", tmp_path / "chunking_eval.json")
    embedder = HashingEmbedder()
    client = get_client(tmp_path / "chroma")
    counts = build_indexes(embedder, client=client)
    return embedder, client, counts


def test_two_separate_collections_indexed(env):
    embedder, client, counts = env
    corpus = chunk_corpus(load_kb())
    assert counts == {name: len(chunks) for name, chunks in corpus.items()}
    assert {c.name for c in client.list_collections()} == set(config.COLLECTIONS)


def test_retrieval_sorted_and_mismatched_embedder_rejected(env):
    embedder, client, _ = env
    hits = Retriever(config.COLLECTION_SENTENCE, embedder, client).retrieve("cardiology consultation fee", k=3)
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)
    assert hits[0].doc_id == "consultation_fee_structure"

    class Other(HashingEmbedder):
        name = "other-model"

    with pytest.raises(RuntimeError, match="indexed with"):
        Retriever(config.COLLECTION_SENTENCE, Other(), client)


def test_calibrate_demo_and_eval_run_end_to_end(env, capsys):
    embedder, client, _ = env
    calib = calibrate_threshold.run(embedder, client)
    for r in calib.values():
        assert r["max_out_of_scope"] < r["threshold"] < r["min_in_scope"]

    failures = demo_grounded_generation.run(embedder, client)
    out = capsys.readouterr().out
    assert out.count("fallback [PASS]") == len(config.COLLECTIONS)  # OOS query falls back in both collections
    assert not [f for f in failures if "butter chicken" in f]

    results = chunking_eval.run(embedder, client)
    assert results["recommended_collection"] in config.COLLECTIONS
    for r in results["collections"].values():
        assert len(r["rows"]) >= 5
        assert all(0.0 <= row["precision"] <= 1.0 and 0.0 <= row["recall"] <= 1.0 for row in r["rows"])
