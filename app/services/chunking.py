# app/services/chunking.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: Optional[int] = None

def chunk_text(
    text: str,
    target_chars: int = 2200,   # ~400–600 tokens ballpark depending on language
    overlap_chars: int = 300,   # ~10–20% overlap
) -> List[Chunk]:
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
