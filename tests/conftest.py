"""Shared fixtures: a real ChromaDB index of the real KB, built once with the test-only embedder."""

import pytest

from app.logging_utils import JsonlRequestLogger
from app.pipeline import SupportPipeline
from app.rag.indexer import build_indexes, get_client
from app.rag.retriever import Retriever
from eval import calibrate_threshold
from tests.fakes import HashingEmbedder


@pytest.fixture(scope="session")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("index")
    embedder, client = HashingEmbedder(), get_client(root / "chroma")
    build_indexes(embedder, client=client)
    threshold = calibrate_threshold.calibrate_collection(Retriever("kb_sentence", embedder, client))["threshold"]
    return embedder, client, threshold


@pytest.fixture()
def pipeline(indexed, tmp_path):
    embedder, client, threshold = indexed
    return SupportPipeline(embedder, client, collection="kb_sentence", threshold=threshold,
                           logger=JsonlRequestLogger(tmp_path / "requests.jsonl"))
