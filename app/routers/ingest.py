# app/routers/ingest.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from uuid import uuid4
from pypdf import PdfReader
from docx import Document

from app.services.chunking import chunk_text
from app.services.container import container

router = APIRouter()


def extract_text_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_docx(file_obj) -> str:
    document = Document(file_obj)
    return "\n".join([p.text for p in document.paragraphs])


@router.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Ingest multiple PDF or DOCX policy documents.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ingestion_results = []

    for file in files:

        filename = file.filename.lower()
        doc_id = str(uuid4())
        
        # Extract Text

        if filename.endswith(".pdf"):
            full_text = extract_text_from_pdf(file.file)

        elif filename.endswith(".docx"):
            full_text = extract_text_from_docx(file.file)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )

        if not full_text.strip():
            ingestion_results.append({
                "document_name": file.filename,
                "doc_id": doc_id,
                "num_chunks": 0,
                "status": "empty_document"
            })
            continue
        
        # Chunk
        chunks = chunk_text(full_text)
        texts = [c.text for c in chunks]

        # Embed
        vectors = await container.embedder.embed_texts_async(texts)

        # Metadata
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "doc_id": doc_id,
                "document_name": file.filename,
                "chunk_id": chunk.chunk_id,
                "page": None,
                "text": chunk.text,
                "char_length": len(chunk.text),
            })

        # Store
        container.vectorstore.add(vectors, metadatas)

        ingestion_results.append({
            "document_name": file.filename,
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "status": "ingested"
        })

    return {
        "total_files_processed": len(files),
        "results": ingestion_results
    }
