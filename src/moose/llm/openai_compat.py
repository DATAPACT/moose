from __future__ import annotations

from typing import Callable

import httpx

from moose.llm.base import EmptyResponse, LLMClient
from moose.llm.usage import record_llm_call

# A key may be given directly or as a callable, so it can be read from a keychain or
# secrets manager at request time and rotated without rebuilding the client.
ApiKey = str | Callable[[], str | None] | None


def _error_message(resp: httpx.Response) -> str:
    try:
        error = resp.json().get("error")
    except ValueError:
        return resp.text[:500]
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    return str(error or resp.text)[:500]


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _record_usage(resp: httpx.Response, data: dict, requested_model: str) -> None:
    usage = data.get("usage") or {}
    # A LiteLLM gateway names the provider model that actually answered (after any fallback).
    model = resp.headers.get("x-litellm-model-name") or data.get("model") or requested_model
    cost = _float_or_none(resp.headers.get("x-litellm-response-cost"))
    if not cost:
        details = usage.get("cost_details") or {}
        cost = _float_or_none(details.get("upstream_inference_cost")) or _float_or_none(usage.get("cost"))
    record_llm_call(
        model=model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cost_usd=cost,
        fallbacks=int(_float_or_none(resp.headers.get("x-litellm-attempted-fallbacks")) or 0),
    )


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible /chat/completions endpoint (e.g. a LiteLLM gateway)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        api_key: ApiKey = None,
        provider: str = "gateway",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(provider=provider, model=model)
        self._api_key = api_key
        self._extra_headers = dict(extra_headers or {})
        # Added to every chat request, e.g. OpenRouter's provider routing preferences.
        self._extra_body = dict(extra_body or {})
        self._client = httpx.AsyncClient(timeout=timeout, base_url=base_url.rstrip("/"))

    def _resolve_api_key(self) -> str | None:
        if callable(self._api_key):
            return self._api_key()
        return self._api_key

    def _headers(self) -> dict[str, str]:
        headers = dict(self._extra_headers)
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def generate(self, prompt: str) -> str:
        payload = {
            **self._extra_body,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        resp = await self._client.post("/chat/completions", json=payload, headers=self._headers())
        if resp.is_error:
            raise RuntimeError(f"{self.provider} returned HTTP {resp.status_code}: {_error_message(resp)}")
        data = resp.json()
        _record_usage(resp, data, self.model)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.provider} response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise EmptyResponse(f"{self.provider} response missing content (finish reason: {choices[0].get('finish_reason')})")
        return content

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/models", headers=self._headers())
        resp.raise_for_status()
        return [item["id"] for item in resp.json().get("data", []) if item.get("id")]

    async def key_info(self) -> dict | None:
        """LiteLLM gateway: the calling key's own record, or None if the gateway rejects the key."""
        root = str(self._client.base_url).rstrip("/").removesuffix("/v1")
        resp = await self._client.get(f"{root}/key/info", headers=self._headers())
        if resp.status_code in (400, 401, 403, 404):
            return None
        resp.raise_for_status()
        return resp.json().get("info") or {}

    async def close(self) -> None:
        await self._client.aclose()
