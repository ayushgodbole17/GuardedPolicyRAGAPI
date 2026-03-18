from __future__ import annotations

from app.utils.logger import logger
from app.utils.config import settings

_FALLBACK_SYSTEM_PROMPT = (
    "You are a policy assistant.\n\n"
    "You must answer ONLY using the provided context.\n"
    "If the answer is not explicitly contained in the context,\n"
    'respond with: "I cannot find this information in the provided documents."\n\n'
    "Do not use outside knowledge."
)

_langfuse_client = None
_client_initialised = False


def get_langfuse():
    global _langfuse_client, _client_initialised

    if _client_initialised:
        return _langfuse_client

    _client_initialised = True

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.info("Langfuse credentials not set — tracing disabled.")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        logger.info(f"Langfuse tracing enabled → {settings.LANGFUSE_BASE_URL}")
    except Exception as exc:
        logger.warning(f"Langfuse init failed (tracing disabled): {exc}")

    return _langfuse_client


def get_system_prompt() -> tuple[str, object | None]:
    lf = get_langfuse()
    if not lf:
        return _FALLBACK_SYSTEM_PROMPT, None

    try:
        prompt = lf.get_prompt("system_prompt-v1")
        logger.info(f"Loaded prompt 'system_prompt-v1' version={prompt.version} from Langfuse")
        return prompt.prompt, prompt
    except Exception as exc:
        logger.warning(f"Could not fetch prompt from Langfuse (using fallback): {exc}")
        return _FALLBACK_SYSTEM_PROMPT, None
