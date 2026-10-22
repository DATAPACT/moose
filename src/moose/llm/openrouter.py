from __future__ import annotations

from moose.llm.openai_compat import ApiKey, OpenAICompatibleClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(
        self,
        api_key: ApiKey,
        model: str,
        timeout: float,
        base_url: str = OPENROUTER_BASE_URL,
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
        )
