import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.routers.ingest import router as ingest_router
from app.routers.ask import router as ask_router
from app.routers.documents import router as documents_router
from app.services.container import container
from app.utils.config import settings
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Startup | model={settings.EMBEDDING_MODEL} | "
        f"dim={settings.VECTOR_DIM} | "
        f"llm={settings.LLM_MODEL} | "
        f"chunks_indexed={container.vectorstore.index.ntotal}"
    )
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Guarded Policy RAG API",
    version="0.2.0",
    description=(
        "Retrieval-augmented generation for policy documents with similarity-based guardrails. "
        "Answers are grounded in ingested content; low-confidence queries are refused."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response



@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe")
async def ready():
    return {
        "status": "ok",
        "chunks_indexed": container.vectorstore.index.ntotal,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
    }



app.include_router(ingest_router, tags=["ingestion"])
app.include_router(ask_router, tags=["query"])
app.include_router(documents_router, tags=["documents"])
