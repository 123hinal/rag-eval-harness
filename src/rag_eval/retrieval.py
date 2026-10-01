"""Hybrid dense + BM25 retrieval with weighted score fusion.

The retriever combines:

* **dense** retrieval — cosine similarity over sentence embeddings, served
  by a FAISS inner-product index over L2-normalized vectors;
* **lexical** retrieval — BM25 over tokenized chunks (``rank-bm25``).

Each candidate pool is min-max normalized to ``[0, 1]`` and fused with
configurable weights. This is a deliberately simple, cross-encoder-free
rerank: cheap, offline, and easy to reason about.
"""

from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from .ingest import Chunk

#: Default sentence-transformer model (384-dim, fast, good quality).
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 and the fallback embedder."""
    return re.findall(r"[a-z0-9]+", text.lower())


class EmbeddingBackend(Protocol):
    """Anything that maps texts to L2-normalized float32 vectors."""

    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of L2-normalized embeddings."""
        ...


class SentenceTransformerBackend:
    """Production backend: sentence-transformers (downloads weights on first use)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        import os

        # The hf-xet fast downloader can stall behind some corporate proxies;
        # prefer the plain HTTP downloader unless the user opted into xet.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbeddingBackend:
    """Deterministic hashing-trick embeddings — the offline fallback.

    Each token contributes signed counts into hashed buckets. Quality is
    below a trained model, but the vectors are deterministic, require no
    downloads, and keep every pipeline, test, and eval runnable offline.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in set(tokenize(text)):
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vectors[i, digest % self.dim] += 1.0
                vectors[i, (digest >> 16) % self.dim] -= 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)


def get_embedding_backend(model_name: str = DEFAULT_MODEL) -> EmbeddingBackend:
    """Return a transformer backend, falling back to hash embeddings offline.

    The fallback emits a :class:`RuntimeWarning` so degraded retrieval
    quality is never silent.
    """
    try:
        return SentenceTransformerBackend(model_name)
    except Exception as exc:  # noqa: BLE001 - downloads can fail in many ways
        warnings.warn(
            f"Could not load sentence-transformers model {model_name!r} ({exc}); "
            "falling back to deterministic hash embeddings. "
            "Retrieval quality will be reduced.",
            RuntimeWarning,
            stacklevel=2,
        )
        return HashEmbeddingBackend()


@dataclass
class RetrievalResult:
    """A retrieved chunk with its fused score and per-signal scores."""

    chunk: Chunk
    score: float  # fused score in [0, 1]
    dense_score: float  # raw cosine similarity
    bm25_score: float  # raw BM25 score


class HybridRetriever:
    """Weighted-fusion hybrid retriever over a fixed chunk collection."""

    def __init__(
        self,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        candidate_multiplier: int = 3,
    ) -> None:
        if dense_weight < 0 or bm25_weight < 0 or dense_weight + bm25_weight <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        total = dense_weight + bm25_weight
        self._dense_weight = dense_weight / total
        self._bm25_weight = bm25_weight / total
        self._candidate_multiplier = candidate_multiplier
        self._chunks: list[Chunk] = []
        self._backend: EmbeddingBackend | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._bm25 = None  # BM25Okapi, imported lazily to keep import light

    # ------------------------------------------------------------------ build

    def build(
        self, chunks: list[Chunk], backend: EmbeddingBackend | None = None
    ) -> "HybridRetriever":
        """Index *chunks* for hybrid search. Returns ``self`` for chaining."""
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self._chunks = list(chunks)
        self._backend = backend or get_embedding_backend()
        embeddings = self._backend.encode([c.text for c in self._chunks])
        self._index = faiss.IndexFlatIP(self._backend.dim)
        self._index.add(embeddings)
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])
        return self

    # ----------------------------------------------------------------- search

    @staticmethod
    def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the top-*k* chunks by fused dense + BM25 score.

        Each signal first retrieves ``top_k * candidate_multiplier``
        candidates; the union is reranked by the weighted fused score.
        """
        if self._index is None or self._backend is None or self._bm25 is None:
            raise RuntimeError("retriever has not been built; call build() first")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = max(1, min(top_k, len(self._chunks)))
        n_candidates = min(
            len(self._chunks), max(top_k * self._candidate_multiplier, top_k + 5)
        )

        query_vector = self._backend.encode([query])
        dense_scores, dense_idx = self._index.search(query_vector, n_candidates)
        dense_map = {int(i): float(s) for i, s in zip(dense_idx[0], dense_scores[0])}

        bm25_scores = self._bm25.get_scores(tokenize(query))
        bm25_top = np.argsort(bm25_scores)[::-1][:n_candidates]
        bm25_map = {int(i): float(bm25_scores[i]) for i in bm25_top}

        dense_norm = self._min_max_normalize(dense_map)
        bm25_norm = self._min_max_normalize(bm25_map)

        fused: list[tuple[int, float]] = []
        for i in set(dense_map) | set(bm25_map):
            score = (
                self._dense_weight * dense_norm.get(i, 0.0)
                + self._bm25_weight * bm25_norm.get(i, 0.0)
            )
            fused.append((i, score))
        fused.sort(key=lambda item: item[1], reverse=True)

        return [
            RetrievalResult(
                chunk=self._chunks[i],
                score=score,
                dense_score=dense_map.get(i, 0.0),
                bm25_score=bm25_map.get(i, 0.0),
            )
            for i, score in fused[:top_k]
        ]

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> Path:
        """Persist the FAISS index, chunks, BM25 state, and metadata to *directory*."""
        if self._index is None or self._backend is None:
            raise RuntimeError("nothing to save; call build() first")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / "index.faiss"))
        (target / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self._chunks], indent=2), encoding="utf-8"
        )
        with open(target / "bm25.pkl", "wb") as fh:
            pickle.dump(self._bm25, fh)
        (target / "meta.json").write_text(
            json.dumps(
                {
                    "backend": self._backend.name,
                    "dim": self._backend.dim,
                    "dense_weight": self._dense_weight,
                    "bm25_weight": self._bm25_weight,
                    "candidate_multiplier": self._candidate_multiplier,
                    "num_chunks": len(self._chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "HybridRetriever":
        """Reload a retriever previously persisted with :meth:`save`."""
        source = Path(directory)
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        backend_name: str = meta["backend"]
        if backend_name.startswith("hash-"):
            backend: EmbeddingBackend = HashEmbeddingBackend(dim=meta["dim"])
        else:
            try:
                backend = SentenceTransformerBackend(backend_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not reload embedding model {backend_name!r} "
                    f"({exc}). Rebuild the index instead of loading it."
                ) from exc
        retriever = cls(
            dense_weight=meta["dense_weight"],
            bm25_weight=meta["bm25_weight"],
            candidate_multiplier=meta["candidate_multiplier"],
        )
        retriever._chunks = [
            Chunk(**c)
            for c in json.loads((source / "chunks.json").read_text(encoding="utf-8"))
        ]
        retriever._backend = backend
        retriever._index = faiss.read_index(str(source / "index.faiss"))
        with open(source / "bm25.pkl", "rb") as fh:
            retriever._bm25 = pickle.load(fh)
        return retriever
