from dataclasses import dataclass
from typing import List, Optional
from app.utils.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: Optional[int] = None


def chunk_text(text: str, chunk_id_offset: int = 0) -> List[Chunk]:
    target_chars = settings.CHUNK_SIZE
    overlap_chars = settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = chunk_id_offset
    n = len(text)

    while start < n:
        end = min(start + target_chars, n)

        if end < n:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(Chunk(chunk_id=f"c{idx}", text=chunk))
            idx += 1

        if end >= n:
            break

        new_start = max(0, end - overlap_chars)
        if new_start > 0:
            boundary = text.find(" ", new_start)
            if boundary != -1 and boundary < end:
                new_start = boundary + 1
        start = new_start

    return chunks
