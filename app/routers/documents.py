# app/routers/documents.py

from fastapi import APIRouter, HTTPException
from app.services.container import container

router = APIRouter(prefix="/documents")


@router.get("", summary="List all ingested documents")
async def list_documents():
    """
    Return a summary of every document currently in the vector store,
    including its doc_id, original filename, and chunk count.
    """
    return {"documents": container.vectorstore.list_documents()}


@router.delete("/{doc_id}", summary="Delete a document and all its chunks")
async def delete_document(doc_id: str):
    """
    Remove all vector store chunks that belong to *doc_id*.

    The FAISS index is rebuilt in-place and flushed to disk.
    Returns 404 if no chunks with that doc_id are found.
    """
    removed = container.vectorstore.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with doc_id='{doc_id}'.",
        )
    return {"doc_id": doc_id, "chunks_removed": removed}
