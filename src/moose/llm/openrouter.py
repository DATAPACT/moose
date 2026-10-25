from __future__ import annotations

from moose.llm.openai_compat import ApiKey, OpenAICompatibleClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_routing(provider_sort: str | None) -> dict[str, object] | None:
    """OpenRouter request fields that choose among the providers serving a model. "throughput"
    picks the fastest one available (your own provider keys still come first); speeds differ
    by more than ten times, e.g. 70 against 1,300 tokens per second for gpt-oss-120b."""
    return {"provider": {"sort": provider_sort}} if provider_sort else None


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(
        self,
        api_key: ApiKey,
        model: str,
        timeout: float = 120.0,
        base_url: str = OPENROUTER_BASE_URL,
        provider_sort: str | None = "throughput",
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        super().__init__(
            base_url=base_url,
            model=model,
            timeout=timeout,
            api_key=api_key,
            provider="openrouter",
            extra_headers={"X-Title": "Moose"},
            extra_body=openrouter_routing(provider_sort),
        )
