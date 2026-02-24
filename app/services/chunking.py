# app/services/chunking.py

from dataclasses import dataclass
from typing import List, Optional
from app.utils.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: Optional[int] = None


def chunk_text(text: str) -> List[Chunk]:

    target_chars = settings.CHUNK_SIZE
    overlap_chars = settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + target_chars, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(Chunk(chunk_id=f"c{idx}", text=chunk))
            idx += 1

        if end == n:
            break

        start = max(0, end - overlap_chars)

    return chunks