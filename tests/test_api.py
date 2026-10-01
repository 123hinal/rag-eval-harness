"""Tests for the FastAPI service (via TestClient, no live server)."""

from fastapi.testclient import TestClient

from rag_eval.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ingest_rebuilds_index():
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rebuilt"
    assert body["docs"] >= 8
    assert body["chunks"] > 0


def test_query_returns_answer_with_citations():
    response = client.post(
        "/query", json={"question": "What uptime does FleetOps guarantee?", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert body["citations"]  # at least one citation
    assert len(body["retrieved"]) == 3
    assert all(r["text"].strip() for r in body["retrieved"])
    assert all(r["chunk_id"] for r in body["retrieved"])


def test_query_validation():
    assert client.post("/query", json={"question": "", "top_k": 3}).status_code == 422
    assert client.post("/query", json={"question": "hi", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "hi", "top_k": 99}).status_code == 422
