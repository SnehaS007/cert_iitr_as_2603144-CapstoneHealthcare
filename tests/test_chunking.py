import app.config as config
from app.rag.chunking import chunk_corpus, fixed_size_chunks, sentence_chunks
from app.rag.kb import KBDocument, load_kb


def _doc(n_words: int) -> KBDocument:
    return KBDocument("d", "Title", " ".join(f"w{i}" for i in range(n_words)) + ".")


def test_fixed_windows_overlap_and_cover_all_words():
    doc = _doc(95)
    chunks = fixed_size_chunks(doc, size=40, overlap=10)
    words = [c.text.split() for c in chunks]
    assert [len(w) for w in words] == [40, 40, 35]
    assert words[0][-10:] == words[1][:10] and words[1][-10:] == words[2][:10]
    assert words[-1][-1] == doc.text.split()[-1]


def test_short_doc_gives_single_fixed_chunk():
    assert len(fixed_size_chunks(_doc(12), size=40, overlap=10)) == 1


def test_sentence_chunks_one_per_sentence():
    for doc in load_kb():
        chunks = sentence_chunks(doc, per_chunk=1)
        assert [c.text for c in chunks] == doc.sentences


def test_corpus_ids_unique_and_strategies_separate():
    corpus = chunk_corpus(load_kb())
    assert set(corpus) == {config.COLLECTION_FIXED, config.COLLECTION_SENTENCE}
    for name, chunks in corpus.items():
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.strategy == name for c in chunks)
