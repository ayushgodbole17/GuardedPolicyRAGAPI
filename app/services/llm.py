from __future__ import annotations

from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.utils.config import settings
from app.services.observability import get_langfuse, get_system_prompt

client = AsyncOpenAI()

_system_prompt_text, _prompt_obj = get_system_prompt()


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    lf = get_langfuse()
    messages = [
        {"role": "system", "content": _system_prompt_text},
        {"role": "user", "content": f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion:\n{question}"},
    ]

    if lf:
        with lf.start_as_current_observation(name="llm-generation", as_type="generation"):
            lf.update_current_generation(
                model=settings.LLM_MODEL,
                input=messages,
            )
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
            lf.update_current_generation(
                output=answer,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            )
            return answer

    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


async def stream_answer(question: str, context_chunks: list[str]) -> AsyncGenerator[str, None]:
    messages = [
        {"role": "system", "content": _system_prompt_text},
        {"role": "user", "content": f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion:\n{question}"},
    ]

    stream = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0,
        stream=True,
    )

    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
