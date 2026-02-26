from app.services.embedding import Embedder
from app.services.vectorstore import VectorStore
from app.utils.config import settings


class ServiceContainer:
    def __init__(self):
        self.embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        self.vectorstore = VectorStore(dim=settings.VECTOR_DIM)


container = ServiceContainer()