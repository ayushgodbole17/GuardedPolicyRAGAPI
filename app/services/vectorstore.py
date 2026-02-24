# app/services/vectorstore.py

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple


INDEX_PATH = "storage/faiss.index"
META_PATH = "storage/metadata.json"


class VectorStore:
    """
    Persistent FAISS-backed vector store.
    """

    def __init__(self, dim: int):
        self.dim = dim

        os.makedirs("storage", exist_ok=True)

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexFlatIP(dim)

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        self.index.add(vectors)
        self.meta.extend(metadatas)

        self._persist()

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:

        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[idx]))

        return results

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)