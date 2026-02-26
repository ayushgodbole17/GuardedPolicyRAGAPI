from fastapi import APIRouter, HTTPException
from app.services.container import container

router = APIRouter(prefix="/documents")


@router.get("", summary="List all ingested documents")
async def list_documents():
    return {"documents": container.vectorstore.list_documents()}


@router.delete("/{doc_id}", summary="Delete a document and all its chunks")
async def delete_document(doc_id: str):
    removed = container.vectorstore.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with doc_id='{doc_id}'.",
        )
    return {"doc_id": doc_id, "chunks_removed": removed}
