# app/routers/ingest.py

import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from uuid import uuid4
from pypdf import PdfReader
from docx import Document

from app.services.chunking import chunk_text
from app.services.container import container
from app.utils.config import settings
from app.utils.logger import logger

router = APIRouter()

_MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


def _extract_pdf_pages(content: bytes) -> List[tuple[str, int]]:
    """
    Extract text per-page from a PDF so page numbers can be stored in metadata.

    Returns:
        List of (page_text, page_number) tuples (1-indexed, empty pages skipped).
    """
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((text.strip(), i))
    return pages


def _extract_docx_text(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


@router.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Ingest one or more PDF or DOCX policy documents.

    Each document is chunked, embedded, and stored in the FAISS vector index.
    Page numbers are tracked for PDF files and surfaced in query hit metadata.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []

    for file in files:
        filename = file.filename or ""
        lower = filename.lower()
        doc_id = str(uuid4())

        # ── File size guard ────────────────────────────────────────────────
        content = await file.read()
        if len(content) > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{filename}' exceeds the {settings.MAX_UPLOAD_MB} MB limit "
                    f"({len(content) / 1_048_576:.1f} MB)."
                ),
            )

        # ── Text extraction ────────────────────────────────────────────────
        all_chunks = []
        all_metadatas = []
        chunk_counter = 0

        if lower.endswith(".pdf"):
            pages = _extract_pdf_pages(content)
            if not pages:
                results.append({"document_name": filename, "doc_id": doc_id,
                                 "num_chunks": 0, "status": "empty_document"})
                continue

            for page_text, page_num in pages:
                page_chunks = chunk_text(page_text, chunk_id_offset=chunk_counter)
                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "doc_id": doc_id,
                        "document_name": filename,
                        "chunk_id": chunk.chunk_id,
                        "page": page_num,
                        "text": chunk.text,
                        "char_length": len(chunk.text),
                    })
                chunk_counter += len(page_chunks)

        elif lower.endswith(".docx"):
            full_text = _extract_docx_text(content)
            if not full_text.strip():
                results.append({"document_name": filename, "doc_id": doc_id,
                                 "num_chunks": 0, "status": "empty_document"})
                continue

            for chunk in chunk_text(full_text):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "doc_id": doc_id,
                    "document_name": filename,
                    "chunk_id": chunk.chunk_id,
                    "page": None,
                    "text": chunk.text,
                    "char_length": len(chunk.text),
                })

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: '{filename}'. Only PDF and DOCX are accepted.",
            )

        # ── Embed & store ─────────────────────────────────────────────────
        texts = [c.text for c in all_chunks]
        vectors = await container.embedder.embed_texts_async(texts)
        container.vectorstore.add(vectors, all_metadatas)

        logger.info(
            f"Ingested '{filename}' | doc_id={doc_id} | chunks={len(all_chunks)}"
        )

        results.append({
            "document_name": filename,
            "doc_id": doc_id,
            "num_chunks": len(all_chunks),
            "status": "ingested",
        })

    return {
        "total_files_processed": len(files),
        "results": results,
    }
