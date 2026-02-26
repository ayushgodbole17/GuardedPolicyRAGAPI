import os
from dotenv import load_dotenv

load_dotenv()

_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Settings:
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    @property
    def VECTOR_DIM(self) -> int:
        return _MODEL_DIMS.get(self.EMBEDDING_MODEL, 1536)


settings = Settings()
