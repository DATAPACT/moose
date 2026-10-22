from __future__ import annotations

import httpx
import pytest

from moose.config import Settings
from moose.llm import OpenAICompatibleClient, OpenRouterClient, create_client


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_create_client_defaults_to_gateway() -> None:
    client = create_client(_settings(MOOSE_LLM_BASE_URL="http://gw:4000/v1", MOOSE_MODEL="local-qwen"))
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "gateway"
    assert client.model == "local-qwen"


def test_create_client_request_model_overrides_default() -> None:
    client = create_client(_settings(MOOSE_MODEL="local-qwen"), "or-claude")
    assert client.model == "or-claude"


def test_create_client_requires_a_model() -> None:
    with pytest.raises(ValueError, match="No LLM model"):
        create_client(_settings())


def test_openrouter_requires_key() -> None:
    with pytest.raises(ValueError):
        OpenRouterClient(api_key=None, model="m", timeout=1)


@pytest.mark.asyncio
async def test_api_key_callable_is_resolved_per_request() -> None:
    keys = iter(["key-1", "key-2"])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(base_url="http://gw/v1", model="m", timeout=1, api_key=lambda: next(keys))
    client._client = httpx.AsyncClient(base_url="http://gw/v1", transport=httpx.MockTransport(handler))
    assert await client.generate("hi") == "ok"
    assert await client.generate("hi") == "ok"
    await client.close()
    assert seen == ["Bearer key-1", "Bearer key-2"]


def test_submit_is_rejected_when_server_has_no_model() -> None:
    from fastapi.testclient import TestClient

    from moose_api.main import app

    env = {"MOOSE_API_KEY": "test-key", "MOOSE_MONGO_URL": "", "MOOSE_MODEL": ""}
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        from moose.config import get_settings

        get_settings.cache_clear()
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/schemas/coarse/ner",
                    json={"text": "Alice works at Moose."},
                    headers={"X-API-Key": "test-key"},
                )
        finally:
            get_settings.cache_clear()
    assert resp.status_code == 503
    assert "No shared models" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_gateway_error_message_is_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "OpenrouterException - Missing Authentication header"}})

    client = OpenAICompatibleClient(base_url="http://gw/v1", model="m", timeout=1, api_key="vk")
    client._client = httpx.AsyncClient(base_url="http://gw/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="HTTP 401: OpenrouterException - Missing Authentication header"):
        await client.generate("hi")
    await client.close()



@pytest.mark.asyncio
async def test_usage_records_actual_model_tokens_and_cost() -> None:
    from moose.llm import track_llm_usage

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-litellm-model-name": "openrouter/openai/gpt-oss-120b", "x-litellm-attempted-fallbacks": "1"},
            json={
                "model": "moose",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost_details": {"upstream_inference_cost": 0.0002}},
            },
        )

    client = OpenAICompatibleClient(base_url="http://gw/v1", model="moose", timeout=1, api_key="vk")
    client._client = httpx.AsyncClient(base_url="http://gw/v1", transport=httpx.MockTransport(handler))
    with track_llm_usage("moose") as usage:
        await client.generate("a")
        await client.generate("b")
    await client.generate("outside tracking")
    await client.close()
    assert usage.as_dict() == {
        "requested_model": "moose",
        "models_used": ["openrouter/openai/gpt-oss-120b"],
        "calls": 2,
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "cost_usd": 0.0004,
        "fallbacks": 2,
    }
