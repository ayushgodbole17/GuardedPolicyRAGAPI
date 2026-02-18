# app/services/retrieval.py

from typing import List, Tuple, Dict, Any
from app.services.container import container


async def retrieve(question: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Executes semantic retrieval:
        1. Embed question
        2. Search vector index
        3. Return metadata hits + similarity scores

    Returns:
        hits: full metadata including full chunk text
        similarities: list of similarity scores
    """

    # Embed question (shape: 1 x dim)
    query_vector = await container.embedder.embed_texts_async([question])

    # Search FAISS
    results = container.vectorstore.search(query_vector, top_k)

    hits: List[Dict[str, Any]] = []
    similarities: List[float] = []

    for score, metadata in results:
        similarities.append(float(score))

        hits.append({
            "doc_id": metadata["doc_id"],
            "document_name": metadata["document_name"],
            "chunk_id": metadata["chunk_id"],
            "page": metadata.get("page"),
            "similarity": float(score),
            "text": metadata["text"],  # FULL chunk passed to LLM
        })

    return hits, similarities
