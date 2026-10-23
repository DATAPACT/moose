from __future__ import annotations

from moose.config import Settings
from moose.llm.base import LLMClient


def create_client(settings: Settings, model: str | None = None, api_key: str | None = None) -> LLMClient:
    provider = settings.MOOSE_LLM_PROVIDER.lower()
    model = model or settings.MOOSE_MODEL
    if not model:
        raise ValueError("No LLM model given. Pass llm.model in the request or set MOOSE_MODEL.")

    if provider == "gateway":
        from moose.llm.openai_compat import OpenAICompatibleClient

        # api_key: a caller's own gateway key; otherwise Moose's key.
        return OpenAICompatibleClient(
            base_url=settings.MOOSE_LLM_BASE_URL,
            api_key=api_key or settings.MOOSE_LLM_API_KEY,
            model=model,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    if provider == "ollama":
        from moose.llm.ollama import OllamaClient

        return OllamaClient(
            host=settings.MOOSE_OLLAMA_HOST,
            token=settings.MOOSE_OLLAMA_TOKEN,
            model=model,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    if provider == "openrouter":
        from moose.llm.openrouter import OpenRouterClient

        return OpenRouterClient(
            api_key=settings.MOOSE_OPENROUTER_API_KEY,
            base_url=settings.MOOSE_OPENROUTER_BASE_URL,
            model=model,
            timeout=settings.MOOSE_TIMEOUT_SECS,
            provider_sort=settings.MOOSE_OPENROUTER_PROVIDER_SORT or None,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.MOOSE_LLM_PROVIDER}")
