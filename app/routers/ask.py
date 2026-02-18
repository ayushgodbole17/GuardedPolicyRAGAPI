# app/routers/ask.py

from fastapi import APIRouter
from time import time

from app.models.request_models import AskRequest
from app.models.response_models import AskResponse, ChunkHit

from app.services.retrieval import retrieve
from app.services.guardrails import decide
from app.services.llm import generate_answer
from app.services.container import (
    SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):

    start_time = time()
    top_k = req.top_k if req.top_k is not None else DEFAULT_TOP_K

    # Retrieval
    hits, similarities = await retrieve(req.question, top_k)

    # Guardrail
    decision = decide(similarities, SIMILARITY_THRESHOLD)

    latency_ms = int((time() - start_time) * 1000)

    if decision.refused:
        return AskResponse(
            answer="I cannot find this information in the provided documents.",
            refused=True,
            refusal_reason=decision.reason,
            confidence=decision.confidence,
            top_k=top_k,
            hits=[],
            latency_ms=latency_ms,
            trace={
                "similarities": similarities,
                "threshold": SIMILARITY_THRESHOLD,
            },
        )

    # Pass to LLM
    context_texts = [hit["text"] for hit in hits]

    answer = await generate_answer(req.question, context_texts)

    hit_models = [
        ChunkHit(
            doc_id=hit["doc_id"],
            document_name=hit["document_name"],
            chunk_id=hit["chunk_id"],
            page=hit["page"],
            similarity=hit["similarity"],
            snippet=hit["text"][:400], 
        )
        for hit in hits
    ]

    return AskResponse(
        answer=answer,
        refused=False,
        refusal_reason=None,
        confidence=decision.confidence,
        top_k=top_k,
        hits=hit_models,
        latency_ms=latency_ms,
        trace={
            "similarities": similarities,
            "threshold": SIMILARITY_THRESHOLD,
        },
    )
