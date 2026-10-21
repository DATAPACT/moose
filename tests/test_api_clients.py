from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moose.config import get_settings
from moose.llm import OpenAICompatibleClient
from moose_api.main import app

ADMIN_KEY = "admin-key"
SHARED = ["openai/gpt-oss-120b", "minimax/minimax-m3.1-flash:free"]
PROVIDER_KEY = "sk-or-v1-user-owned-provider-key-0123456789"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    async def fake_list_models(self) -> list[str]:
        return SHARED

    monkeypatch.setattr(OpenAICompatibleClient, "list_models", fake_list_models)
    env = {
        "MOOSE_API_KEY": ADMIN_KEY,
        "MOOSE_LLM_API_KEY": "sk-moose-gateway",
        "MOOSE_MONGO_URL": "",
        "MOOSE_MODEL": "openai/gpt-oss-120b",
        "MOOSE_WORKER_COUNT": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def _admin() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}


def _register(client: TestClient, name: str) -> dict[str, str]:
    invite = client.post("/admin/invites", json={"note": name}, headers=_admin()).json()
    resp = client.post("/vault/register", json={"invite_code": invite["invite_code"], "name": name})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _add_key(client: TestClient, token: str) -> str:
    resp = client.post(
        "/vault/keys",
        json={"provider": "openrouter", "api_key": PROVIDER_KEY, "label": "mine", "default_model": "x/y"},
        headers={"X-API-Key": token},
    )
    assert resp.status_code == 200, resp.text
    assert PROVIDER_KEY not in resp.text
    return resp.json()["key_id"]


def _submit(client: TestClient, token: str, **llm) -> str:
    body = {"text": "Alice works at Moose."}
    if llm:
        body["llm"] = llm
    resp = client.post("/schemas/coarse/ner", json=body, headers={"X-API-Key": token})
    assert resp.status_code == 200, resp.text
    return resp.json()["job_id"]


def test_unknown_token_is_rejected(client: TestClient) -> None:
    for token in ("moose_pt_not-a-real-token", "sk-whatever"):
        assert client.get("/schemas", headers={"X-API-Key": token}).status_code == 401


def test_register_lists_own_keys_masked(client: TestClient) -> None:
    user = _register(client, "Partner A")
    key_id = _add_key(client, user["token"])
    models = client.get("/models", headers={"X-API-Key": user["token"]}).json()
    assert models["models"] == SHARED
    assert [k["key_id"] for k in models["own_keys"]] == [key_id]
    me = client.get("/vault/me", headers={"X-API-Key": user["token"]}).text
    assert PROVIDER_KEY not in me


def test_job_on_own_key_keeps_the_key_in_memory(client: TestClient) -> None:
    user = _register(client, "Partner A")
    key_id = _add_key(client, user["token"])
    job_id = _submit(client, user["token"], key_id=key_id)

    stored = app.state.job_store._jobs[job_id]
    assert stored.payload["llm"] == {"model": "x/y", "key_id": key_id, "provider": "openrouter"}
    assert PROVIDER_KEY not in repr(stored.__dict__)
    queued_job_id, route = app.state.worker_pool._local_queue.get_nowait()
    assert queued_job_id == job_id
    assert (route.base_url, route.api_key) == ("https://openrouter.ai/api/v1", PROVIDER_KEY)


def test_users_are_isolated_from_each_other_and_from_admin(client: TestClient) -> None:
    alice, bob = _register(client, "Alice"), _register(client, "Bob")
    alice_key = _add_key(client, alice["token"])
    job_id = _submit(client, alice["token"], key_id=alice_key)

    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": alice["token"]}).status_code == 200
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": bob["token"]}).status_code == 404
    assert client.get(f"/jobs/{job_id}", headers=_admin()).status_code == 404

    stolen = client.post(
        "/schemas/coarse/ner",
        json={"text": "x", "llm": {"key_id": alice_key}},
        headers={"X-API-Key": bob["token"]},
    )
    assert stolen.status_code == 404
    assert client.post("/schemas/coarse/ner", json={"text": "x", "llm": {"key_id": alice_key}}, headers=_admin()).status_code == 400
    assert client.get("/vault/keys", headers=_admin()).status_code == 403


def test_shared_models_still_work_for_users(client: TestClient) -> None:
    user = _register(client, "Partner A")
    _submit(client, user["token"])
    refused = client.post(
        "/schemas/coarse/ner",
        json={"text": "x", "llm": {"model": "very-expensive"}},
        headers={"X-API-Key": user["token"]},
    )
    assert refused.status_code == 400


def test_tokens_and_recovery_through_the_api(client: TestClient) -> None:
    user = _register(client, "Partner A")
    headers = {"X-API-Key": user["token"]}
    second = client.post("/vault/tokens", json={"label": "laptop"}, headers=headers).json()
    assert client.get("/vault/me", headers={"X-API-Key": second["token"]}).status_code == 200

    assert client.delete(f"/vault/tokens/{user['token_id']}", headers={"X-API-Key": second["token"]}).json() == {
        "revoked": True
    }
    assert client.get("/vault/me", headers=headers).status_code == 401

    recovered = client.post("/vault/recover", json={"recovery_code": user["recovery_code"]}).json()
    assert client.get("/vault/me", headers={"X-API-Key": recovered["token"]}).status_code == 200
    assert client.post("/vault/recover", json={"recovery_code": "moose_rc_wrong"}).status_code == 400


def test_unsupported_provider_is_rejected(client: TestClient) -> None:
    user = _register(client, "Partner A")
    resp = client.post(
        "/vault/keys",
        json={"provider": "custom", "api_key": PROVIDER_KEY, "label": "x"},
        headers={"X-API-Key": user["token"]},
    )
    assert resp.status_code == 400


def test_schema_changes_need_admin_key(client: TestClient) -> None:
    user = _register(client, "Partner A")
    for method, path in [
        ("post", "/schemas/reload"),
        ("delete", "/schemas/some_schema"),
        ("patch", "/schemas/some_schema"),
        ("post", "/schemas/ingest"),
        ("post", "/schemas/ingest/preview"),
        ("post", "/admin/invites"),
    ]:
        resp = client.request(method, path, json={}, headers={"X-API-Key": user["token"]})
        assert resp.status_code == 403, (method, path, resp.status_code)
    assert client.post("/schemas/reload", headers=_admin()).status_code == 200


@pytest.mark.asyncio
async def test_worker_calls_the_users_provider_with_their_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from moose.config import Settings
    from moose_api.queue import InMemoryJobStore, InMemoryQueue, JobRecord, WorkerPool
    from moose_api.vault_store import ProviderRoute

    seen: dict[str, str] = {}

    async def fake_generate(self, prompt: str) -> str:
        seen.update(base_url=str(self._client.base_url).rstrip("/"), key=self._resolve_api_key(), model=self.model)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)
    settings = Settings(_env_file=None, MOOSE_MODEL="shared", MOOSE_WORKER_COUNT=0, MOOSE_MAX_RETRIES=0)
    store = InMemoryJobStore()
    pool = WorkerPool(InMemoryQueue(10), store, None, settings)
    await store.put_job(
        JobRecord(
            job_id="j1",
            endpoint_type="ner",
            payload={"schema": "coarse", "text": "Alice", "llm": {"model": "x/y", "key_id": "k1"}},
            status="queued",
            created_at="now",
            updated_at="now",
            retries=0,
            client="user:u1",
        )
    )
    route = ProviderRoute(provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key=PROVIDER_KEY, secret_id="k1")
    await pool._process_job("j1", route)

    assert seen == {"base_url": "https://openrouter.ai/api/v1", "key": PROVIDER_KEY, "model": "x/y"}
    job = await store.get_job("j1")
    assert job.status == "failed" and PROVIDER_KEY not in (job.error or "")
