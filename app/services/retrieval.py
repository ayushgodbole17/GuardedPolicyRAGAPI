from typing import List, Tuple, Dict, Any
from app.services.container import container


async def retrieve(question: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    query_vector = await container.embedder.embed_texts_async([question])
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
            "text": metadata["text"],
        })

    return hits, similarities
