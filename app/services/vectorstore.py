import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple


INDEX_PATH = "storage/faiss.index"
META_PATH = "storage/metadata.json"


class VectorStore:
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

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        self.index.add(vectors)
        self.meta.extend(metadatas)
        self._persist()

    def delete_by_doc_id(self, doc_id: str) -> int:
        keep_mask = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
        removed = len(self.meta) - len(keep_mask)

        if removed == 0:
            return 0

        new_index = faiss.IndexFlatIP(self.dim)

        if keep_mask:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            kept_vectors = all_vectors[keep_mask].astype(np.float32)
            new_index.add(kept_vectors)

        self.index = new_index
        self.meta = [self.meta[i] for i in keep_mask]
        self._persist()

        return removed

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

    def list_documents(self) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for m in self.meta:
            did = m["doc_id"]
            if did not in seen:
                seen[did] = {
                    "doc_id": did,
                    "document_name": m["document_name"],
                    "chunk_count": 0,
                }
            seen[did]["chunk_count"] += 1
        return list(seen.values())

    def _persist(self) -> None:
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
