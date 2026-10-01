# rag-eval-harness

A production-style **hybrid RAG pipeline** (dense embeddings + BM25 with weighted
score fusion) over a sample knowledge base, with a **built-in offline evaluation
harness** that gates on retrieval and answer-quality metrics. No API keys required
for anything — the generator falls back to a clearly-labeled extractive mode when
`OPENAI_API_KEY` is absent.

Built as a portfolio project demonstrating applied LLM engineering: clean typing,
small focused modules, real tests, CI, and honest eval methodology.

## Features

- **Hybrid retrieval** — `sentence-transformers` (`all-MiniLM-L6-v2`) dense vectors
  in FAISS + BM25 (`rank-bm25`), min-max normalized and fused (default 0.6 / 0.4).
- **Recursive character chunking** with overlap and stable chunk ids.
- **Pluggable generation** — OpenAI (`gpt-4o-mini`) when a key is present, otherwise
  an extractive fallback that returns the top supporting chunks with citations.
- **Evaluation harness** — context precision@k, context recall, faithfulness
  (claim-support heuristic), and answer relevancy over a gold Q&A set. Writes JSON
  + Markdown reports and exits nonzero below configured thresholds.
- **FastAPI service** — `/health`, `/ingest`, `/query`.
- **Fully offline-capable** — if the embedding model can't be downloaded, a
  deterministic hash-embedding fallback keeps every test and eval green (with a
  loud warning).

## Architecture

```
                        ┌──────────────┐
                        │  user query  │
                        └──────┬───────┘
                               ▼
                 ┌─────────────────────────┐
                 │     HybridRetriever     │
                 │  ┌────────┐ ┌────────┐  │
                 │  │ dense  │ │  BM25  │  │
                 │  │ FAISS  │ │ lexical│  │
                 │  └───┬────┘ └───┬────┘  │
                 │      ▼    fusion  ▼     │  min-max normalize,
                 │   top-k reranked list   │  weighted sum, sort
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │       RAGPipeline       │
                 │  context ──► generator  │
                 │  (OpenAI │ extractive) │
                 └────────────┬────────────┘
                              ▼
                    answer + citations

Evaluation (offline):

   gold Q&A ──► pipeline (extractive) ──► metrics ──► reports/*.json|*.md
                                                     exit 1 if < thresholds
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite
pytest -q

# Ask a question (extractive mode, no API key needed)
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from rag_eval.ingest import load_corpus, chunk_documents
from rag_eval.retrieval import HybridRetriever
from rag_eval.pipeline import RAGPipeline

retriever = HybridRetriever().build(
    chunk_documents(load_corpus("data/corpus")))
response = RAGPipeline(retriever).query("What uptime does FleetOps guarantee?")
print(response.answer)
print("citations:", response.citations)
EOF

# Run the offline evaluation (writes reports/eval_report.{json,md})
python scripts/run_eval.py

# Serve the API
uvicorn --app-dir src rag_eval.api:app --port 8000
# GET  /health   -> liveness + index stats
# POST /ingest   -> rebuild the index from data/corpus
# POST /query    -> {"question": "...", "top_k": 5}
```

Set `OPENAI_API_KEY` to switch generation from extractive to `gpt-4o-mini`.
(The `openai` package is an optional, lazily-imported dependency.)

> **Note (corporate proxies):** if the embedding-model download stalls, the
> pipeline automatically disables the `hf-xet` fast downloader
> (`HF_HUB_DISABLE_XET=1`) and retries over plain HTTPS. If the model still
> can't be fetched, it falls back to deterministic hash embeddings with a
> `RuntimeWarning` — everything keeps working offline.

## Evaluation methodology

The harness (`src/rag_eval/eval.py`, CLI: `scripts/run_eval.py`) runs the
pipeline in extractive mode over 14 gold Q&A pairs (`data/gold.json`) about the
sample FleetOps knowledge base, retrieving `top_k = 3` chunks per question:

| Metric | Definition |
| --- | --- |
| `context_precision_at_k` | Fraction of the top-k retrieved documents in the gold relevant set |
| `context_recall` | Fraction of gold relevant documents retrieved |
| `faithfulness` | Fraction of answer sentences with ≥50% of content words supported by a single retrieved chunk (lexical claim-support heuristic — no model calls) |
| `answer_relevancy` | Fraction of the question's content words covered by the answer |

Gold `reference_answer` fields are not scored by these retrieval-grounded
metrics; they exist for qualitative review and future LLM-judge extensions.

### Sample results

Run on the bundled 10-document FleetOps corpus (14 questions, `top_k=3`):

| Metric | Average | Threshold | Status |
| --- | --- | --- | --- |
| `context_precision_at_k` | 0.476 | 0.30 | ✅ |
| `context_recall` | 0.964 | 0.80 | ✅ |
| `faithfulness` | 0.950 | 0.85 | ✅ |
| `answer_relevancy` | 0.765 | 0.50 | ✅ |

Thresholds (`eval_config.json`) were calibrated from baseline runs on the sample
corpus in both online (transformer embeddings) and fully offline (hash-embedding
fallback) modes, so `scripts/run_eval.py` passes in either environment. Tighten
them as your corpus and pipeline improve — that's the point of the gate.

## Project structure

```
rag-eval-harness/
├── src/rag_eval/
│   ├── __init__.py      # version + PROJECT_ROOT
│   ├── ingest.py        # markdown loading, recursive chunking with overlap
│   ├── retrieval.py     # hybrid dense+BM25 retriever, FAISS persist/load
│   ├── pipeline.py      # retrieve→generate; OpenAI or extractive generator
│   ├── eval.py          # metrics, report writers, threshold gating
│   └── api.py           # FastAPI: /health, /ingest, /query
├── data/
│   ├── corpus/          # 10 markdown docs: fictional "FleetOps" logistics SaaS
│   └── gold.json        # 14 gold Q&A pairs with relevant doc ids
├── scripts/run_eval.py  # CLI: build index → evaluate → write reports
├── tests/               # pytest: chunking, retrieval, metrics, API
├── eval_config.json     # top_k + metric thresholds
├── Dockerfile
└── .github/workflows/ci.yml
```

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Retrieval `top_k` for eval | `eval_config.json` | `3` |
| Metric thresholds | `eval_config.json` → `thresholds` | see above |
| Dense/BM25 fusion weights | `HybridRetriever(dense_weight=…, bm25_weight=…)` | `0.6 / 0.4` |
| Chunk size / overlap | `chunk_documents(…, chunk_size=…, chunk_overlap=…)` | `800 / 120` |
| Generation backend | `OPENAI_API_KEY` env var | extractive fallback |
| Embedding model | `HybridRetriever.build(…, backend=…)` / `DEFAULT_MODEL` | `all-MiniLM-L6-v2` |

## API reference

```bash
curl localhost:8000/health
# {"status":"ok","version":"0.1.0","indexed":false,"docs":0,"chunks":0}

curl -X POST localhost:8000/ingest
# {"status":"rebuilt","docs":10,"chunks":14}

curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I verify a webhook signature?","top_k":3}'
# {"answer":"[1] ... (source: webhooks.md) ...","citations":[...],"retrieved":[...]}
```

## License

MIT — do what you want with it.
