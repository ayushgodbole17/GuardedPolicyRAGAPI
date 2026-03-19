# Guarded Policy RAG API

FastAPI backend for policy document Q&A. Uses RAG with similarity-based guardrails — questions without supporting evidence in the index are refused rather than hallucinated.

## Features

- Ingest PDF and DOCX files
- OpenAI embeddings stored in a persistent FAISS index
- Similarity threshold guardrail — low-confidence queries are refused
- Streaming and non-streaming responses
- Langfuse tracing (optional)
- Prometheus metrics + Grafana dashboard

## Quickstart

### Docker (recommended)

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up --build
```

| Service    | URL                          |
|------------|------------------------------|
| API docs   | http://localhost:8000/docs   |
| Grafana    | http://localhost:3001        |
| Prometheus | http://localhost:9090        |

Grafana login: admin / admin

### Manual

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /ingest | Upload PDF or DOCX files |
| POST | /ask | Ask a question |
| POST | /ask/stream | Streaming response (SSE) |
| GET | /documents | List ingested documents |
| DELETE | /documents/{doc_id} | Delete a document |
| GET | /health | Liveness probe |
| GET | /ready | Readiness probe |
| GET | /metrics | Prometheus metrics |

## Langfuse (optional)

Tracing is disabled if keys are not set. To enable:

1. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to `.env`
3. Create a prompt named `system_prompt-v1` (text type) in Langfuse Prompt Management

## Evals

Requires a running API with documents already ingested.

```bash
docker compose run --rm evals          # keyword check + RAGAS metrics
```

To run keyword checks only (faster, no extra API calls):

```bash
docker compose run --rm evals bash -c "pip install -q -r requirements-evals.txt && python -m evals.run_evals --base-url http://api:8000"
```

## Load testing

```bash
pip install locust
locust -f load_test/locustfile.py --host http://localhost:8000
```

Open http://localhost:8089. Simulates policy users (weight 3) and out-of-domain noise users (weight 1) to generate realistic refusal rate data.

## Configuration

All settings are loaded from `.env`. See `.env.example` for available options and defaults.

## Storage

`storage/` holds the FAISS index and chunk metadata. It is gitignored. In Docker it is bind-mounted so data persists across container restarts.
