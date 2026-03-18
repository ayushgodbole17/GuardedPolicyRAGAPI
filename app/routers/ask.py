import json
from contextlib import contextmanager
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from time import time
from typing import AsyncGenerator

from app.models.request_models import AskRequest
from app.models.response_models import AskResponse, ChunkHit

from app.services.retrieval import retrieve
from app.services.guardrails import decide
from app.services.llm import generate_answer, stream_answer
from app.services.observability import get_langfuse
from app.services.metrics import (
    rag_requests_total,
    rag_latency_seconds,
    rag_max_similarity,
    rag_confidence,
)
from app.utils.config import settings
from app.utils.logger import logger

router = APIRouter()


@contextmanager
def _noop():
    yield


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start_time = time()
    top_k = req.top_k if req.top_k is not None else settings.DEFAULT_TOP_K
    lf = get_langfuse()

    with lf.start_as_current_observation(name="rag-query", input={"question": req.question}) if lf else _noop():

        with lf.start_as_current_observation(name="retrieval", as_type="retriever") if lf else _noop():
            hits, similarities = await retrieve(req.question, top_k)
            if lf:
                lf.update_current_span(output={
                    "hits_count": len(hits),
                    "max_similarity": max(similarities) if similarities else 0,
                })

        decision = decide(similarities, settings.SIMILARITY_THRESHOLD)
        max_sim = max(similarities) if similarities else 0.0
        rag_max_similarity.observe(max_sim)

        if decision.refused:
            latency_ms = int((time() - start_time) * 1000)
            rag_requests_total.labels(outcome="refused").inc()
            rag_latency_seconds.observe(latency_ms / 1000)

            logger.info(
                f"REFUSED | question='{req.question}' | "
                f"max_sim={max_sim:.4f} | reason={decision.reason} | latency_ms={latency_ms}"
            )
            if lf:
                lf.update_current_span(output={"refused": True, "reason": decision.reason})

            return AskResponse(
                answer="I cannot find this information in the provided documents.",
                refused=True,
                refusal_reason=decision.reason,
                confidence=decision.confidence,
                top_k=top_k,
                hits=[],
                latency_ms=latency_ms,
                trace={"similarities": similarities, "threshold": settings.SIMILARITY_THRESHOLD},
            )

        context_texts = [hit["text"] for hit in hits]
        answer = await generate_answer(req.question, context_texts)

        latency_ms = int((time() - start_time) * 1000)
        rag_requests_total.labels(outcome="answered").inc()
        rag_latency_seconds.observe(latency_ms / 1000)
        rag_confidence.observe(decision.confidence)

        logger.info(
            f"ANSWERED | question='{req.question}' | "
            f"max_sim={max_sim:.4f} | confidence={decision.confidence:.4f} | latency_ms={latency_ms}"
        )
        if lf:
            lf.update_current_span(output={"answer": answer, "refused": False})

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
            trace={"similarities": similarities, "threshold": settings.SIMILARITY_THRESHOLD},
        )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    start_time = time()
    top_k = req.top_k if req.top_k is not None else settings.DEFAULT_TOP_K

    hits, similarities = await retrieve(req.question, top_k)
    decision = decide(similarities, settings.SIMILARITY_THRESHOLD)
    max_sim = max(similarities) if similarities else 0.0
    rag_max_similarity.observe(max_sim)

    if decision.refused:
        latency_ms = int((time() - start_time) * 1000)
        rag_requests_total.labels(outcome="refused").inc()
        rag_latency_seconds.observe(latency_ms / 1000)

        logger.info(
            f"REFUSED | question='{req.question}' | "
            f"max_sim={max_sim:.4f} | reason={decision.reason} | latency_ms={latency_ms}"
        )

        payload = json.dumps({
            "refused": True,
            "reason": decision.reason,
            "confidence": decision.confidence,
        })

        async def refused_body() -> AsyncGenerator[str, None]:
            yield f"data: {payload}\n\n"

        return StreamingResponse(refused_body(), media_type="text/event-stream")

    context_texts = [hit["text"] for hit in hits]

    async def token_stream() -> AsyncGenerator[str, None]:
        async for token in stream_answer(req.question, context_texts):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

        latency_ms = int((time() - start_time) * 1000)
        rag_requests_total.labels(outcome="answered").inc()
        rag_latency_seconds.observe(latency_ms / 1000)
        rag_confidence.observe(decision.confidence)

        logger.info(
            f"STREAMED | question='{req.question}' | "
            f"max_sim={max_sim:.4f} | confidence={decision.confidence:.4f} | latency_ms={latency_ms}"
        )

    return StreamingResponse(token_stream(), media_type="text/event-stream")
