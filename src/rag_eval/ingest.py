"""Document loading and text chunking.

Everything in this module is pure (no network access; the only I/O is the
explicit ``load_corpus`` read), so it is straightforward to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Separators tried in priority order by the recursive splitter.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


@dataclass(frozen=True)
class Document:
    """A single source document loaded from disk."""

    doc_id: str  # e.g. "tracking-api" (the markdown file stem)
    text: str
    source: str  # human-readable origin, e.g. the file name


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document — the unit of retrieval."""

    chunk_id: str  # e.g. "tracking-api::2"
    doc_id: str
    text: str
    source: str
    index: int  # position of the chunk within its document


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Load every ``*.md`` file in *corpus_dir* as a :class:`Document`.

    Files are read in sorted order so the result is deterministic.
    Raises :class:`FileNotFoundError` if the directory does not exist and
    :class:`ValueError` if it contains no markdown documents.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_path}")
    documents: list[Document] = []
    for md_file in sorted(corpus_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(Document(doc_id=md_file.stem, text=text, source=md_file.name))
    if not documents:
        raise ValueError(f"no markdown documents found in {corpus_path}")
    return documents


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """Recursively split *text* on the highest-priority separator it contains."""
    chosen: str | None = None
    for sep in separators:
        if sep == "":
            break
        if sep in text:
            chosen = sep
            break
    if chosen is None:
        return [text] if text else []
    pieces: list[str] = []
    for piece in text.split(chosen):
        piece = piece.strip()
        if not piece:
            continue
        # The piece no longer contains `chosen` (or any earlier separator),
        # so recursion proceeds to lower-priority separators only.
        pieces.extend(_split_recursive(piece, separators))
    return pieces


def _merge_splits(splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge splits into chunks of at most *chunk_size* characters.

    Consecutive chunks overlap: after emitting a chunk, splits are dropped
    from the front only until the remaining window fits within
    *chunk_overlap* characters *and* the next split still fits.
    """
    chunks: list[str] = []
    current: list[str] = []

    def current_len() -> int:
        return sum(len(s) for s in current) + max(0, len(current) - 1)

    for split in splits:
        if len(split) > chunk_size:
            # A single unsplittable token (e.g. a very long URL): hard-cut it.
            if current:
                chunks.append(" ".join(current))
                current = []
            for i in range(0, len(split), chunk_size):
                chunks.append(split[i : i + chunk_size])
            continue
        if current and current_len() + 1 + len(split) > chunk_size:
            chunks.append(" ".join(current))
            while current and (
                current_len() > chunk_overlap
                or current_len() + 1 + len(split) > chunk_size
            ):
                current.pop(0)
        current.append(split)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Uses recursive character splitting (paragraphs, then lines, then
    sentences, then words) so chunk boundaries prefer natural breaks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
    text = text.strip()
    if not text:
        return []
    splits = _split_recursive(text, _SEPARATORS)
    return _merge_splits(splits, chunk_size, chunk_overlap)


def chunk_documents(
    documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    """Chunk every document, assigning stable ``{doc_id}::{index}`` chunk ids."""
    chunks: list[Chunk] = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    text=text,
                    source=doc.source,
                    index=i,
                )
            )
    return chunks
