"""Tests for document loading and chunking."""

import pytest

from rag_eval.ingest import chunk_documents, chunk_text, load_corpus


def test_chunk_text_short_text_unchanged():
    assert chunk_text("hello world", chunk_size=100, chunk_overlap=20) == ["hello world"]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_respects_chunk_size():
    text = " ".join(f"sentence number {i} about shipping logistics." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # No content lost: every word of the input appears in some chunk.
    rejoined_words = {w for c in chunks for w in c.split()}
    assert all(f"{i}" in " ".join(rejoined_words) for i in range(60))


def test_chunk_text_overlap():
    text = " ".join(f"word{i:03d}" for i in range(200))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    # Consecutive chunks share content (the overlap window).
    assert chunks[1][:20] in chunks[0]


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=50, chunk_overlap=60)


def test_chunk_documents_assigns_ids(sample_documents):
    chunks = chunk_documents(sample_documents, chunk_size=80, chunk_overlap=10)
    assert chunks
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.doc_id}::{chunk.index}"
        assert chunk.text.strip()


def test_load_corpus(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nbody b", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nbody a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["a", "b"]  # sorted, markdown only
    assert docs[0].source == "a.md"


def test_load_corpus_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_load_corpus_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
