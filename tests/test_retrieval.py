"""Tests for hybrid retrieval."""

import numpy as np
import pytest

from rag_eval.ingest import chunk_documents
from rag_eval.retrieval import (
    HashEmbeddingBackend,
    HybridRetriever,
    get_embedding_backend,
)


@pytest.fixture()
def retriever(sample_documents, backend):
    chunks = chunk_documents(sample_documents, chunk_size=400, chunk_overlap=40)
    return HybridRetriever().build(chunks, backend=backend)


def test_search_returns_relevant_doc_first(retriever):
    results = retriever.search("How do I verify a webhook signature?", top_k=3)
    assert results
    assert results[0].chunk.doc_id == "webhooks"


def test_search_billing_query(retriever):
    results = retriever.search("How much does the Growth plan cost?", top_k=3)
    assert results[0].chunk.doc_id == "billing"


def test_search_scores_sorted_descending(retriever):
    results = retriever.search("shipment tracking statuses", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_search_top_k_clamped(retriever):
    results = retriever.search("shipping", top_k=100)
    assert 0 < len(results) <= 3  # only 3 docs indexed


def test_search_requires_build(backend):
    with pytest.raises(RuntimeError):
        HybridRetriever().search("hello")


def test_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_build_rejects_empty_chunks(backend):
    with pytest.raises(ValueError):
        HybridRetriever().build([], backend=backend)


def test_save_and_load_roundtrip(retriever, tmp_path):
    index_dir = retriever.save(tmp_path / "index")
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "bm25.pkl").exists()
    assert (index_dir / "meta.json").exists()

    loaded = HybridRetriever.load(index_dir)
    before = [r.chunk.chunk_id for r in retriever.search("webhook signature", top_k=2)]
    after = [r.chunk.chunk_id for r in loaded.search("webhook signature", top_k=2)]
    assert before == after


def test_hash_backend_is_deterministic():
    backend = HashEmbeddingBackend(dim=64)
    first = backend.encode(["fleetops shipping api", "unrelated text"])
    second = backend.encode(["fleetops shipping api", "unrelated text"])
    assert first.shape == (2, 64)
    assert np.allclose(first, second)
    norms = np.linalg.norm(first, axis=1)
    assert np.allclose(norms, 1.0)


def test_get_embedding_backend_returns_working_backend():
    backend = get_embedding_backend()
    vectors = backend.encode(["hello world"])
    assert vectors.shape == (1, backend.dim)
    assert vectors.dtype == np.float32
