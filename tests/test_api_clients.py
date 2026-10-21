from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from moose.config import get_settings
from moose.llm import OpenAICompatibleClient
from moose_api.main import app

ADMIN_KEY = "admin-key"
MOOSE_GATEWAY_KEY = "sk-moose"
TEAM_A_KEY = "sk-team-a"
TEAM_A_SECOND_KEY = "sk-team-a-2"
TEAM_B_KEY = "sk-team-b"
# An "LLM API" key: may call models, but the gateway refuses it /key/info.
TEAM_A_LLM_ONLY_KEY = "sk-team-a-llm-only"

SHARED = ["openai/gpt-oss-120b", "minimax/minimax-m3.1-flash:free"]

# What a LiteLLM gateway reports for each virtual key.
GATEWAY_KEYS = {
    MOOSE_GATEWAY_KEY: ({"team_id": None}, SHARED),
    TEAM_A_KEY: ({"team_id": "team-a"}, ["partner-a/private", *SHARED]),
    TEAM_A_SECOND_KEY: ({"team_id": "team-a"}, ["partner-a/private", *SHARED]),
    TEAM_B_KEY: ({"team_id": "team-b"}, SHARED),
    TEAM_A_LLM_ONLY_KEY: (None, ["partner-a/private", *SHARED]),
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    async def fake_key_info(self) -> dict | None:
        entry = GATEWAY_KEYS.get(self._resolve_api_key())
        return entry[0] if entry else None

    async def fake_list_models(self) -> list[str]:
        key = self._resolve_api_key()
        if key not in GATEWAY_KEYS:
            request = httpx.Request("GET", "http://gw/v1/models")
            raise httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))
        return GATEWAY_KEYS[key][1]

    monkeypatch.setattr(OpenAICompatibleClient, "key_info", fake_key_info)
    monkeypatch.setattr(OpenAICompatibleClient, "list_models", fake_list_models)
    env = {
        "MOOSE_API_KEY": ADMIN_KEY,
        "MOOSE_LLM_API_KEY": MOOSE_GATEWAY_KEY,
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


def _submit(client: TestClient, key: str, **extra) -> str:
    resp = client.post(
        "/schemas/coarse/ner",
        json={"text": "Alice works at Moose.", **extra},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["job_id"]


def test_unknown_key_is_rejected(client: TestClient) -> None:
    assert client.get("/schemas", headers={"X-API-Key": "sk-unknown"}).status_code == 401


def test_each_key_sees_its_own_models(client: TestClient) -> None:
    def models(key: str) -> list[str]:
        return client.get("/models", headers={"X-API-Key": key}).json()["models"]

    assert models(ADMIN_KEY) == SHARED
    assert models(TEAM_A_KEY) == ["partner-a/private", *SHARED]
    assert models(TEAM_B_KEY) == SHARED


def test_partners_cannot_pick_other_teams_models(client: TestClient) -> None:
    _submit(client, TEAM_A_KEY, llm={"model": "partner-a/private"})
    refused = client.post(
        "/schemas/coarse/ner",
        json={"text": "x", "llm": {"model": "partner-a/private"}},
        headers={"X-API-Key": TEAM_B_KEY},
    )
    assert refused.status_code == 400
    assert "not available" in refused.json()["detail"]


def test_jobs_are_visible_to_the_same_team_only(client: TestClient) -> None:
    job_id = _submit(client, TEAM_A_KEY)
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": TEAM_A_SECOND_KEY}).status_code == 200
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": TEAM_B_KEY}).status_code == 404
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": ADMIN_KEY}).status_code == 200

    admin_job = _submit(client, ADMIN_KEY)
    assert client.get(f"/jobs/{admin_job}", headers={"X-API-Key": TEAM_A_KEY}).status_code == 404


def test_partner_keys_stay_in_memory(client: TestClient) -> None:
    job_id = _submit(client, TEAM_A_KEY)
    stored = app.state.job_store._jobs[job_id]
    assert TEAM_A_KEY not in repr(stored.__dict__)
    assert stored.client == "team:team-a"
    # The job waits in this process's in-memory queue, not the shared queue.
    assert app.state.worker_pool.local_queue_size() == 1


def test_schema_changes_need_admin_key(client: TestClient) -> None:
    for method, path in [
        ("post", "/schemas/reload"),
        ("delete", "/schemas/some_schema"),
        ("patch", "/schemas/some_schema"),
        ("post", "/schemas/ingest"),
        ("post", "/schemas/ingest/preview"),
    ]:
        resp = client.request(method, path, json={}, headers={"X-API-Key": TEAM_A_KEY})
        assert resp.status_code == 403, (method, path, resp.status_code)
    assert client.post("/schemas/reload", headers={"X-API-Key": ADMIN_KEY}).status_code == 200


def test_llm_only_keys_are_accepted_but_stand_for_themselves(client: TestClient) -> None:
    models = client.get("/models", headers={"X-API-Key": TEAM_A_LLM_ONLY_KEY})
    assert models.status_code == 200
    assert "partner-a/private" in models.json()["models"]

    job_id = _submit(client, TEAM_A_LLM_ONLY_KEY)
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": TEAM_A_LLM_ONLY_KEY}).status_code == 200
    # Without /key/info the team is unknown, so other keys of the team do not see the job.
    assert client.get(f"/jobs/{job_id}", headers={"X-API-Key": TEAM_A_KEY}).status_code == 404
