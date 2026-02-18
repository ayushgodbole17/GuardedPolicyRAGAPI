# app/services/embedding.py

import numpy as np
from openai import AsyncOpenAI

class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = AsyncOpenAI()

    async def embed_texts_async(self, texts):
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        vectors = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32
        )

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors = vectors / norms

        return vectors
