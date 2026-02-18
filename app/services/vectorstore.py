# app/services/vectorstore.py
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

class VectorStore:
    def __init__(self, dim: int):
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.dim = dim
        # Inner product index; with normalized vectors, IP == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        assert vectors.dtype == np.float32
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.meta.extend(metadatas)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        assert query_vec.dtype == np.float32
        assert query_vec.shape == (1, self.dim)
        scores, idxs = self.index.search(query_vec, top_k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i == -1:
                continue
            results.append((float(s), self.meta[i]))
        return results
