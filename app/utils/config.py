# app/utils/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    VECTOR_DIM = 1536
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2200))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))

settings = Settings()