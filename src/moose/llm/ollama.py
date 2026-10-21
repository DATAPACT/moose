from __future__ import annotations

import asyncio

import ollama

from moose.llm.base import LLMClient
from moose.llm.usage import record_llm_call


class OllamaClient(LLMClient):
    def __init__(self, host: str, token: str | None, model: str, timeout: float) -> None:
        super().__init__(provider="ollama", model=model)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = ollama.AsyncClient(host=host, headers=headers, timeout=timeout)

    async def generate(self, prompt: str) -> str:
        data = await self._client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0},
        )
        if "response" not in data:
            raise RuntimeError("Ollama response missing 'response' field")
        record_llm_call(
            model=f"ollama/{self.model}",
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        return data["response"]

    async def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
