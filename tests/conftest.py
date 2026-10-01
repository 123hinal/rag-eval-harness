"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_eval.ingest import Document  # noqa: E402
from rag_eval.retrieval import EmbeddingBackend, get_embedding_backend  # noqa: E402


@pytest.fixture(scope="session")
def backend() -> EmbeddingBackend:
    """Embedding backend for tests.

    Uses the real sentence-transformer model when it can be downloaded,
    otherwise the deterministic offline hash fallback. Retrieval tests are
    written to pass under either backend.
    """
    return get_embedding_backend()


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            doc_id="webhooks",
            text=(
                "FleetOps sends webhooks for shipment events. Verify the "
                "X-FleetOps-Signature header using HMAC-SHA256 with your webhook "
                "secret. Failed deliveries are retried 5 times with exponential "
                "backoff over 24 hours."
            ),
            source="webhooks.md",
        ),
        Document(
            doc_id="billing",
            text=(
                "The Growth plan costs $199 per month and includes 5,000 labels. "
                "Invoices are issued on the first of each month."
            ),
            source="billing.md",
        ),
        Document(
            doc_id="tracking",
            text=(
                "Use GET /v2/track/{tracking_number} to follow a shipment. "
                "Statuses include in_transit, out_for_delivery, and delivered."
            ),
            source="tracking.md",
        ),
    ]
