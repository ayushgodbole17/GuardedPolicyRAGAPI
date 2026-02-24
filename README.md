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

Setup
1. Create venv
python -m venv .venv
.venv\Scripts\activate

2. Install
pip install -r requirements.txt

3. Configure .env
OPENAI_API_KEY=your_key
SIMILARITY_THRESHOLD=0.45
DEFAULT_TOP_K=5
CHUNK_SIZE=2200
CHUNK_OVERLAP=300

4. Run
uvicorn app.main:app --reload

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

Persistent files:

storage/faiss.index
storage/metadata.json
