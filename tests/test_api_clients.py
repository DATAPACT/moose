from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moose.config import get_settings
from moose.llm import OpenAICompatibleClient
from moose_api.main import app
from oidc_helpers import ISSUER, admin_login, bearer, login_token

SHARED_KEY = "shared-key-for-external-users"
SHARED = ["openai/gpt-oss-120b", "minimax/minimax-m3.1-flash:free"]
PROVIDER_KEY = "sk-or-v1-user-owned-provider-key-0123456789"
OPERATOR_KEY = "sk-or-v1-operator-shared-key-9876543210"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, test_signing_key):
    env = {
        "MOOSE_API_KEY": SHARED_KEY,
        "MOOSE_OIDC_ISSUER": ISSUER,
        "MOOSE_SECRET_KEY": "server-secret-for-tests",
        "MOOSE_MONGO_URL": "",
        "MOOSE_WORKER_COUNT": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        headers = admin_login()
        assert test_client.put(
            "/admin/shared-models/key", json={"provider": "openrouter", "api_key": OPERATOR_KEY}, headers=headers
        ).status_code == 200
        assert test_client.put(
            "/admin/shared-models", json={"models": SHARED, "default_model": SHARED[0]}, headers=headers
        ).status_code == 200
        yield test_client
    get_settings.cache_clear()


def _admin() -> dict[str, str]:
    return admin_login()


def _shared() -> dict[str, str]:
    return {"X-API-Key": SHARED_KEY}


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
    queued_job_id, route, data_key = app.state.worker_pool._local_queue.get_nowait()
    assert queued_job_id == job_id and data_key is not None
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
        for caller in ({"X-API-Key": user["token"]}, _shared(), bearer(login_token(["moose-user"]))):
            resp = client.request(method, path, json={}, headers=caller)
            assert resp.status_code in (401, 403), (method, path, caller, resp.status_code)
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


def test_shared_models_are_admin_only_and_key_stays_encrypted(client: TestClient) -> None:
    user = _register(client, "Partner A")
    for method, path in [("get", "/admin/shared-models"), ("put", "/admin/shared-models"), ("put", "/admin/shared-models/key")]:
        for caller in ({"X-API-Key": user["token"]}, _shared(), bearer(login_token(["moose-user"]))):
            assert client.request(method, path, json={}, headers=caller).status_code in (401, 403), (path, caller)

    described = client.get("/admin/shared-models", headers=_admin()).json()
    assert described["models"] == SHARED and described["default_model"] == SHARED[0]
    assert OPERATOR_KEY not in repr(described)
    assert OPERATOR_KEY not in repr(app.state.shared_models._config.docs)

    assert client.get("/models", headers={"X-API-Key": user["token"]}).json()["models"] == SHARED
    bad_default = client.put("/admin/shared-models", json={"models": SHARED, "default_model": "other"}, headers=_admin())
    assert bad_default.status_code == 400


def test_shared_key_needs_a_server_secret(monkeypatch: pytest.MonkeyPatch, test_signing_key) -> None:
    for key, value in {
        "MOOSE_API_KEY": SHARED_KEY,
        "MOOSE_OIDC_ISSUER": ISSUER,
        "MOOSE_MONGO_URL": "",
        "MOOSE_WORKER_COUNT": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MOOSE_SECRET_KEY", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as test_client:
            resp = test_client.put(
                "/admin/shared-models/key", json={"provider": "openrouter", "api_key": OPERATOR_KEY}, headers=_admin()
            )
            assert resp.status_code == 400 and "MOOSE_SECRET_KEY" in resp.json()["detail"]
            # Without shared models, jobs need the user's own key.
            assert test_client.post("/schemas/coarse/ner", json={"text": "x"}, headers=_shared()).status_code == 503
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_worker_runs_shared_models_on_the_operator_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from moose.config import Settings
    from moose_api.queue import InMemoryJobStore, InMemoryQueue, JobRecord, WorkerPool
    from moose_api.shared_models import SharedModelsStore

    seen: dict[str, str] = {}

    async def fake_generate(self, prompt: str) -> str:
        seen.update(base_url=str(self._client.base_url).rstrip("/"), key=self._resolve_api_key(), model=self.model)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)
    shared = SharedModelsStore.in_memory("server-secret-for-tests")
    await shared.set_key("openrouter", OPERATOR_KEY)
    settings = Settings(_env_file=None, MOOSE_WORKER_COUNT=0, MOOSE_MAX_RETRIES=0)
    store = InMemoryJobStore()
    pool = WorkerPool(InMemoryQueue(10), store, shared, settings)
    await store.put_job(
        JobRecord(
            job_id="j2",
            endpoint_type="ner",
            payload={"schema": "coarse", "text": "Alice", "llm": {"model": SHARED[0]}},
            status="queued",
            created_at="now",
            updated_at="now",
            retries=0,
            client="user:u1",
        )
    )
    await pool._process_job("j2")
    assert seen == {"base_url": "https://openrouter.ai/api/v1", "key": OPERATOR_KEY, "model": SHARED[0]}


def test_shared_key_uses_shared_models_but_cannot_administer(client: TestClient) -> None:
    job_id = _submit(client, SHARED_KEY)
    assert client.get(f"/jobs/{job_id}", headers=_shared()).status_code == 200
    assert client.get("/models", headers=_shared()).json()["models"] == SHARED
    assert client.put(
        "/admin/shared-models", json={"models": ["anthropic/claude-opus-4"]}, headers=_shared()
    ).status_code == 401
    assert client.get("/admin/shared-models", headers=_admin()).json()["models"] == SHARED


def test_admin_login_can_follow_its_own_jobs(client: TestClient) -> None:
    resp = client.post("/schemas/coarse/ner", json={"text": "Alice works at Moose."}, headers=_admin())
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}", headers=_admin()).status_code == 200
    assert client.get(f"/jobs/{job_id}", headers=_shared()).status_code == 404


@pytest.mark.asyncio
async def test_shared_key_from_another_secret_must_be_reentered() -> None:
    from moose_api.shared_models import SharedModelsStore

    old = SharedModelsStore.in_memory("old-secret")
    await old.set_key("openrouter", OPERATOR_KEY)
    rotated = SharedModelsStore(old._config, "new-secret")
    described = await rotated.describe()
    assert described["key_unreadable"] and described["masked_key"] is None


def test_schema_preview_without_shared_key_explains_itself(monkeypatch: pytest.MonkeyPatch, test_signing_key) -> None:
    for key, value in {
        "MOOSE_API_KEY": SHARED_KEY,
        "MOOSE_SECRET_KEY": "server-secret-for-tests",
        "MOOSE_OIDC_ISSUER": ISSUER,
        "MOOSE_MONGO_URL": "",
        "MOOSE_WORKER_COUNT": "0",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        with TestClient(app) as test_client:
            assert test_client.put(
                "/admin/shared-models", json={"models": SHARED}, headers=admin_login()
            ).status_code == 200
            resp = test_client.post(
                "/schemas/ingest/preview",
                json={"name": "x", "source_type": "file", "source_path": "missing.json", "use_llm_fallback": True},
                headers=admin_login(),
            )
            assert resp.status_code == 503 and "provider key" in resp.json()["detail"]
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_restart_fails_only_users_jobs() -> None:
    from moose_api.queue import MongoJobStore

    class _Result:
        modified_count = 0

    class _Jobs:
        query = None

        async def update_many(self, query, update):
            _Jobs.query = query
            return _Result()

    await MongoJobStore({"jobs": _Jobs()}).fail_interrupted_partner_jobs()
    assert _Jobs.query == {
        "status": {"$in": ["queued", "processing"]},
        "client": {"$regex": "^user:"},
    }


TEXT = "Maria Rossi, maria.rossi@example.com, was treated for diabetes."


async def _run_local(pool, job_id: str, route, data_key) -> None:
    await pool._process_job(job_id, route, data_key)


def test_users_job_input_and_result_are_encrypted_at_rest(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import moose_api.queue as queue_module

    async def fake_run_text_ner(tasks, schema, llm_client, settings=None):
        return {"results": [{"task_id": tasks[0]["task_id"], "entities": [{"text": tasks[0]["text"][:11]}]}]}

    monkeypatch.setattr(queue_module, "run_text_ner", fake_run_text_ner)
    user = _register(client, "Partner A")
    resp = client.post("/schemas/coarse/ner", json={"text": TEXT}, headers={"X-API-Key": user["token"]})
    job_id = resp.json()["job_id"]

    stored = app.state.job_store._jobs[job_id]
    assert TEXT not in repr(stored.__dict__) and "Maria" not in repr(stored.__dict__)
    queued_job_id, route, data_key = app.state.worker_pool._local_queue.get_nowait()
    asyncio.run(_run_local(app.state.worker_pool, queued_job_id, route, data_key))

    stored = app.state.job_store._jobs[job_id]
    assert stored.status == "completed"
    assert "Maria" not in repr(stored.__dict__)  # result sealed at rest

    owner = client.get(f"/jobs/{job_id}", headers={"X-API-Key": user["token"]}).json()
    assert owner["result"] == {"entities": [{"text": "Maria Rossi"}]}
    assert client.get(f"/jobs/{job_id}", headers=_admin()).status_code == 404


def test_users_job_errors_are_encrypted_and_readable_by_owner(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import moose_api.queue as queue_module

    async def failing_run_text_ner(tasks, schema, llm_client, settings=None):
        raise RuntimeError(f"could not parse: {tasks[0]['text']}")

    monkeypatch.setattr(queue_module, "run_text_ner", failing_run_text_ner)
    user = _register(client, "Partner A")
    job_id = client.post("/schemas/coarse/ner", json={"text": TEXT}, headers={"X-API-Key": user["token"]}).json()["job_id"]
    queued = app.state.worker_pool._local_queue.get_nowait()
    asyncio.run(_run_local(app.state.worker_pool, *queued))

    stored = app.state.job_store._jobs[job_id]
    assert stored.status == "failed" and "Maria" not in repr(stored.__dict__)
    owner = client.get(f"/jobs/{job_id}", headers={"X-API-Key": user["token"]}).json()
    assert owner["status"] == "failed" and TEXT in owner["error"]


def test_users_job_without_its_key_fails_cleanly(client: TestClient) -> None:
    import asyncio

    user = _register(client, "Partner A")
    job_id = client.post("/schemas/coarse/ner", json={"text": TEXT}, headers={"X-API-Key": user["token"]}).json()["job_id"]
    app.state.worker_pool._local_queue.get_nowait()
    # As after a restart: the job is still stored, but its vault key is gone.
    asyncio.run(app.state.worker_pool._process_job(job_id))
    owner = client.get(f"/jobs/{job_id}", headers={"X-API-Key": user["token"]}).json()
    assert owner["status"] == "failed" and "submit it again" in owner["error"]


def test_shared_key_jobs_stay_readable(client: TestClient) -> None:
    job_id = _submit(client, SHARED_KEY)
    assert app.state.job_store._jobs[job_id].payload["text"] == "Alice works at Moose."
