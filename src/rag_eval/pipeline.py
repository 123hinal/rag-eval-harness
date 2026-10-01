"""RAG pipeline: retrieve -> build context -> generate.

Generation is pluggable. :class:`OpenAIGenerator` is used when an
``OPENAI_API_KEY`` is available; otherwise the pipeline falls back to
:class:`ExtractiveGenerator`, which needs no key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from .retrieval import HybridRetriever, RetrievalResult


class Generator(Protocol):
    """Anything that turns a question + retrieved context into an answer."""

    def generate(self, question: str, context: list[RetrievalResult]) -> str:
        ...


class ExtractiveGenerator:
    """Offline fallback generator — no API key required.

    Returns the most relevant retrieved chunks verbatim, each prefixed with
    a citation marker (``[1]``, ``[2]``, ...). Because the answer is composed
    only of retrieved text, every claim in it is directly supported by the
    cited context, which makes faithfulness trivially high.
    """

    def __init__(self, max_chunks: int = 2) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.max_chunks = max_chunks

    def generate(self, question: str, context: list[RetrievalResult]) -> str:
        if not context:
            return "I could not find relevant documentation to answer this question."
        parts = [
            f"[{i}] {result.chunk.text.strip()} (source: {result.chunk.source})"
            for i, result in enumerate(context[: self.max_chunks], start=1)
        ]
        return "\n\n".join(parts)


class OpenAIGenerator:
    """Generator backed by the OpenAI chat completions API.

    The ``openai`` package is imported lazily so the rest of the project
    never depends on it, and construction fails fast with a clear message
    when no API key is configured.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for OpenAIGenerator "
                "(pip install openai)."
            ) from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use OpenAIGenerator."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        self.max_chunks = 5

    def generate(self, question: str, context: list[RetrievalResult]) -> str:
        context_block = "\n\n".join(
            f"[{i}] {result.chunk.text.strip()}"
            for i, result in enumerate(context, start=1)
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using ONLY the numbered "
                        "context passages below. Cite the passages you use "
                        "like [1], [2]. If the context does not contain the "
                        "answer, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()


@dataclass
class RAGResponse:
    """The answer plus the evidence it was built from."""

    answer: str
    citations: list[str]  # chunk_ids backing the answer
    results: list[RetrievalResult] = field(default_factory=list)


class RAGPipeline:
    """Retrieve-then-generate pipeline with a swappable generator."""

    def __init__(
        self, retriever: HybridRetriever, generator: Generator | None = None
    ) -> None:
        self.retriever = retriever
        self.generator = generator if generator is not None else self._default_generator()

    @staticmethod
    def _default_generator() -> Generator:
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIGenerator()
            except RuntimeError:
                pass  # e.g. `openai` not installed: fall through to extractive
        return ExtractiveGenerator()

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Retrieve top-*k* chunks for *question* and generate an answer."""
        results = self.retriever.search(question, top_k=top_k)
        answer = self.generator.generate(question, results)
        # Citations mirror the chunks the generator actually surfaces.
        n_cited = getattr(self.generator, "max_chunks", len(results))
        citations = [r.chunk.chunk_id for r in results[:n_cited]]
        return RAGResponse(answer=answer, citations=citations, results=results)
