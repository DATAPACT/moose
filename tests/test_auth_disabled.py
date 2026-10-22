from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moose.config import Settings, get_settings
from moose_api.main import app, check_auth_settings

LOCAL = {
    "MOOSE_API_KEY": "shared-key",
    "MOOSE_SECRET_KEY": "server-secret-for-tests",
    "MOOSE_MONGO_URL": "",
    "MOOSE_WORKER_COUNT": "0",
    "MOOSE_AUTH_ENABLED": "false",
    "MOOSE_BIND_ADDRESS": "127.0.0.1",
    "MOOSE_DASHBOARD_URL": "http://localhost:8502",
}


@pytest.fixture
def local_client(monkeypatch: pytest.MonkeyPatch):
    for key, value in LOCAL.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_login_off_makes_everyone_the_local_developer(local_client: TestClient) -> None:
    account = local_client.get("/account").json()
    assert account["name"] == "Local developer"
    assert account["is_admin"] and account["can_use_vault"]

    assert local_client.get("/admin/shared-models").status_code == 200
    assert local_client.put("/admin/shared-models", json={"models": ["openai/gpt-oss-120b"]}).status_code == 200
    created = local_client.post("/account/vault")
    assert created.status_code == 200 and created.json()["token"].startswith("moose_pt_")


@pytest.mark.parametrize(
    "overrides",
    [
        {"MOOSE_DASHBOARD_URL": "https://moose-ui.zooverse.dev"},
        {"MOOSE_DASHBOARD_URL": ""},
        {"MOOSE_BIND_ADDRESS": "0.0.0.0"},
    ],
    ids=["public-dashboard", "no-dashboard-url", "public-bind"],
)
def test_login_cannot_be_turned_off_on_a_server(overrides: dict[str, str]) -> None:
    settings = Settings(_env_file=None, **{**LOCAL, **overrides})
    with pytest.raises(RuntimeError, match="only allowed locally"):
        check_auth_settings(settings)


def test_login_is_on_by_default() -> None:
    settings = Settings(_env_file=None, MOOSE_API_KEY="k", MOOSE_DASHBOARD_URL="https://moose-ui.zooverse.dev")
    assert settings.MOOSE_AUTH_ENABLED
    check_auth_settings(settings)
