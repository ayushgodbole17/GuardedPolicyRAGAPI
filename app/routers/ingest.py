# app/routers/ingest.py

from fastapi import APIRouter
from uuid import uuid4
from time import time
from app.models.request_models import IngestTextRequest
from app.models.response_models import IngestResponse
from app.services.chunking import chunk_text
from app.services.container import container

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):

    start_time = time()

    doc_id = str(uuid4())

    # 1️⃣ Chunk
    chunks = chunk_text(req.text)

    if not chunks:
        return IngestResponse(
            doc_id=doc_id,
            document_name=req.document_name,
            num_chunks=0,
            avg_chunk_chars=0,
            embedding_model=container.embedder.model_name
        )

    texts = [c.text for c in chunks]

    # 2️⃣ Embed
    vectors = await container.embedder.embed_texts_async(texts)

    # 3️⃣ Attach metadata
    metadatas = []
    for chunk in chunks:
        metadatas.append({
            "doc_id": doc_id,
            "document_name": req.document_name,
            "chunk_id": chunk.chunk_id,
            "page": None,
            "text": chunk.text,
            "char_length": len(chunk.text),
        })

    # 4️⃣ Store
    container.vectorstore.add(vectors, metadatas)

    avg_len = sum(len(c.text) for c in chunks) / len(chunks)

    return IngestResponse(
        doc_id=doc_id,
        document_name=req.document_name,
        num_chunks=len(chunks),
        avg_chunk_chars=avg_len,
        embedding_model=container.embedder.model_name
    )
