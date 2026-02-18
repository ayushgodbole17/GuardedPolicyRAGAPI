# app/main.py
from fastapi import FastAPI
from app.routers.ingest import router as ingest_router
from app.routers.ask import router as ask_router

app = FastAPI(title="Guarded Policy RAG API", version="0.1.0")
app.include_router(ingest_router, prefix="")
app.include_router(ask_router, prefix="")
