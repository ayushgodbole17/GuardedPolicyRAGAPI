# app/services/container.py

from app.services.embedding import Embedder
from app.services.vectorstore import VectorStore
from app.utils.config import settings


class ServiceContainer:
    """
    Holds long-lived service instances.
    Ensures FAISS index persists while server runs.
    """

    def __init__(self):
        self.embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        self.vectorstore = VectorStore(dim=settings.VECTOR_DIM)


# Singleton instance
container = ServiceContainer()