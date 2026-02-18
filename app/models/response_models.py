# app/models/response_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChunkHit(BaseModel):
    doc_id: str
    document_name: str
    chunk_id: str
    page: Optional[int] = None
    similarity: float
    snippet: str

class IngestResponse(BaseModel):
    doc_id: str
    document_name: str
    num_chunks: int
    avg_chunk_chars: float
    embedding_model: str

class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    confidence: float
    top_k: int
    hits: List[ChunkHit]
    latency_ms: int
    trace: Dict[str, Any]
