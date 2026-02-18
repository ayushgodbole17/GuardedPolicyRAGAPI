# app/services/container.py

import os
from dotenv import load_dotenv

from app.services.embedding import Embedder
from app.services.vectorstore import VectorStore


# Load environment variables
load_dotenv()

# ----------------------------
# Configuration
# ----------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = 1536  # dimension for text-embedding-3-small
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.78))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))


class ServiceContainer:
    """
    Holds long-lived service instances.
    Ensures FAISS index persists while server runs.
    """

    def __init__(self):
        self.embedder = Embedder(model_name=EMBEDDING_MODEL)
        self.vectorstore = VectorStore(dim=VECTOR_DIM)


# Singleton instance
container = ServiceContainer()
