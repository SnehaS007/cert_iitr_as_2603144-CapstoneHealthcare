import pytest

from app.rag.retriever import RetrievedChunk
from eval import calibrate_threshold
from eval.chunking_eval import score


def test_score_dedups_parent_docs():
    r = score(["a", "a", "b", "c"], {"a", "d"})
    assert r["retrieved_docs"] == ["a", "b", "c"]
    assert r["precision_expr"] == "1/3" and r["recall_expr"] == "1/2"
    assert r["precision"] == pytest.approx(1 / 3) and r["recall"] == 0.5


class _StubRetriever:
    collection_name = "stub"

    def __init__(self, sims: dict[str, float]):
        self.sims = sims

    def retrieve(self, query, k=1):
        return [RetrievedChunk("c", "d", "t", "x", self.sims.get(query, 0.0))]


def test_calibration_midpoint(monkeypatch):
    monkeypatch.setattr(calibrate_threshold, "CALIBRATION_IN_SCOPE", ["i1", "i2"])
    monkeypatch.setattr(calibrate_threshold, "CALIBRATION_OUT_OF_SCOPE", ["o1"])
    r = calibrate_threshold.calibrate_collection(_StubRetriever({"i1": 0.6, "i2": 0.5, "o1": 0.2}))
    assert r["threshold"] == pytest.approx(0.35)


def test_calibration_overlap_raises(monkeypatch):
    monkeypatch.setattr(calibrate_threshold, "CALIBRATION_IN_SCOPE", ["i1"])
    monkeypatch.setattr(calibrate_threshold, "CALIBRATION_OUT_OF_SCOPE", ["o1"])
    with pytest.raises(calibrate_threshold.CalibrationError):
        calibrate_threshold.calibrate_collection(_StubRetriever({"i1": 0.4, "o1": 0.45}))
