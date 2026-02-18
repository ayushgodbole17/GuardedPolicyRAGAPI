# app/services/llm.py

from openai import AsyncOpenAI

client = AsyncOpenAI()

SYSTEM_PROMPT = """
You are a policy assistant.

You must answer ONLY using the provided context.
If the answer is not explicitly contained in the context,
respond with: "I cannot find this information in the provided documents."

Do not use outside knowledge.
"""


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates grounded answer using retrieved context.
    """

    context_block = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""
Context:
{context_block}

Question:
{question}
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
