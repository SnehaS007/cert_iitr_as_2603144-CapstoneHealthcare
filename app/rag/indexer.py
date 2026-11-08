"""Task 3 — embed both chunk sets and index each into its OWN ChromaDB collection."""

from __future__ import annotations

from pathlib import Path

import app.config as config  # sets telemetry env vars before chromadb is imported
import chromadb
from chromadb.config import Settings

from app.rag.chunking import Chunk, chunk_corpus
from app.rag.embeddings import Embedder
from app.rag.kb import KBDocument, load_kb


def get_client(path: Path = config.CHROMA_DIR) -> chromadb.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _recreate_collection(client: chromadb.ClientAPI, name: str, embedder_name: str):
    """Drop and recreate so chunk IDs from an older chunking configuration cannot linger."""
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "embedder": embedder_name},
        embedding_function=None,  # we always supply embeddings; never let Chroma download a default model
    )


def upsert_chunks(collection, chunks: list[Chunk], embedder: Embedder) -> int:
    if not chunks:
        raise ValueError(f"No chunks to index into '{collection.name}'")
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embedder.encode([c.embed_text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return collection.count()


def build_indexes(
    embedder: Embedder, client: chromadb.ClientAPI | None = None, docs: list[KBDocument] | None = None
) -> dict[str, int]:
    """Rebuild both collections from the KB. Returns {collection_name: chunk_count}."""
    client = client or get_client()
    docs = docs if docs is not None else load_kb()
    counts: dict[str, int] = {}
    for name, chunks in chunk_corpus(docs).items():
        collection = _recreate_collection(client, name, embedder.name)
        counts[name] = upsert_chunks(collection, chunks, embedder)
        if counts[name] != len(chunks):
            raise RuntimeError(f"'{name}': indexed {counts[name]} chunks, expected {len(chunks)}")
    return counts


def main() -> None:
    from app.rag.embeddings import get_default_embedder

    docs = load_kb()
    corpus = chunk_corpus(docs)
    print(f"Knowledge base: {len(docs)} documents from {config.KB_DIR.name}/")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(
        f"Fixed-size strategy: {config.FIXED_CHUNK_WORDS}-word windows, "
        f"{config.FIXED_OVERLAP_WORDS}-word overlap"
    )
    print(f"Sentence strategy: {config.SENTENCES_PER_CHUNK} sentence(s) per chunk\n")
    for name, chunks in corpus.items():
        print(f"[{name}] {len(chunks)} chunks; first two:")
        for c in chunks[:2]:
            print(f"  {c.chunk_id}: {c.text}")
    counts = build_indexes(get_default_embedder(), docs=docs)
    print(f"\nIndexed (collection.upsert): {counts} at {config.CHROMA_DIR.relative_to(config.ROOT_DIR)}/")


if __name__ == "__main__":
    main()
