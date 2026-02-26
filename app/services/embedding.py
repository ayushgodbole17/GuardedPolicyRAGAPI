import os
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Ensure it is set in your .env file."
    )


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def embed_texts_async(self, texts: list[str]) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors = vectors / norms

        return vectors
