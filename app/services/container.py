# app/services/container.py

import os
from uuid import uuid4
from dotenv import load_dotenv
from app.services.embedding import Embedder
from app.services.vectorstore import VectorStore

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536  # OpenAI embedding dimension

class ServiceContainer:
    def __init__(self):
        self.embedder = Embedder(model_name=EMBEDDING_MODEL)
        self.vectorstore = VectorStore(dim=VECTOR_DIM)

container = ServiceContainer()
