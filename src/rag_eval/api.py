"""FastAPI service exposing the RAG pipeline.

Endpoints:

* ``GET /health`` — liveness probe plus index statistics.
* ``POST /ingest`` — rebuild the hybrid index from ``data/corpus``.
* ``POST /query`` — ask a question; returns an answer with citations.

The index is built lazily on first use so importing the app (e.g. in tests)
never triggers model downloads.
"""

from __future__ import annotations

from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import PROJECT_ROOT, __version__
from .ingest import chunk_documents, load_corpus
from .pipeline import RAGPipeline
from .retrieval import HybridRetriever

app = FastAPI(
    title="rag-eval-harness",
    version=__version__,
    description="Hybrid dense + BM25 RAG pipeline with citations.",
)

_pipeline: RAGPipeline | None = None
_lock = Lock()
_stats: dict[str, int] = {"docs": 0, "chunks": 0}


def _build_pipeline() -> tuple[RAGPipeline, int, int]:
    documents = load_corpus(PROJECT_ROOT / "data" / "corpus")
    chunks = chunk_documents(documents)
    retriever = HybridRetriever().build(chunks)
    return RAGPipeline(retriever), len(documents), len(chunks)


def get_pipeline() -> RAGPipeline:
    """Return the singleton pipeline, building the index on first use."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                pipeline, n_docs, n_chunks = _build_pipeline()
                _pipeline = pipeline
                _stats.update(docs=n_docs, chunks=n_chunks)
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved: list[RetrievedChunk]


class HealthResponse(BaseModel):
    status: str
    version: str
    indexed: bool
    docs: int
    chunks: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        indexed=_pipeline is not None,
        docs=_stats["docs"],
        chunks=_stats["chunks"],
    )


@app.post("/ingest")
def ingest() -> dict:
    """Rebuild the hybrid index from ``data/corpus``."""
    global _pipeline
    with _lock:
        pipeline, n_docs, n_chunks = _build_pipeline()
        _pipeline = pipeline
        _stats.update(docs=n_docs, chunks=n_chunks)
    return {"status": "rebuilt", "docs": n_docs, "chunks": n_chunks}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer *question* using the top-*k* retrieved chunks as context."""
    response = get_pipeline().query(request.question, top_k=request.top_k)
    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        retrieved=[
            RetrievedChunk(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                source=r.chunk.source,
                score=round(r.score, 4),
                text=r.chunk.text[:500],
            )
            for r in response.results
        ],
    )
