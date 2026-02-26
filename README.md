Guarded Policy RAG API
FastAPI backend for policy document Q&A using Retrieval-Augmented Generation (RAG) with similarity-based refusal guardrails.

Features:
- Multi-file ingestion (.docx, .pdf)
- Semantic chunking with overlap
- OpenAI embeddings (normalized)
- Persistent FAISS vector index
- Similarity threshold guardrail
- Grounded LLM responses
- Source attribution + confidence score
- Structured logging
- Config via .env

Setup (Docker)
1. Copy .env.example to .env and add your OpenAI key
2. docker compose up --build

Setup (manual)
1. python -m venv .venv && .venv\Scripts\activate
2. pip install -r requirements.txt
3. Copy .env.example to .env and add your OpenAI key
4. uvicorn app.main:app --reload

Docs:
http://127.0.0.1:8000/docs

Endpoints
POST /ingest

Upload multiple .docx or .pdf files.

Stores:
- chunk embeddings
- metadata
- persistent FAISS index

POST /ask
Input:
{
  "question": "How many sick leave days are allowed?"
}

Output:
- answer
- refused
- confidence
- hits
- similarity trace
- latency

Guardrail Logic:
Refuse if:
- max_similarity < threshold

Confidence = mean similarity of retrieved chunks.
Prevents low-evidence hallucination.

Storage

storage/ is gitignored. The FAISS index and metadata are created locally on first ingest.
When using Docker, storage/ is bind-mounted so data persists across container restarts.
