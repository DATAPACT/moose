from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from moose.config import get_settings
from moose_api.main import app
from oidc_helpers import ISSUER, bearer, login_token

OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(**overrides) -> str:
    roles = overrides.pop("roles", None)
    return login_token(roles, **overrides)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, test_signing_key):
    env = {
        "MOOSE_API_KEY": "admin-key",
        "MOOSE_MONGO_URL": "",
        "MOOSE_WORKER_COUNT": "0",
        "MOOSE_OIDC_ISSUER": ISSUER,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


_bearer = bearer


def test_first_login_creates_one_vault_per_account(client: TestClient) -> None:
    token = _token()
    assert client.get("/account", headers=_bearer(token)).json() == {
        "name": "Partner One",
        "email": "one@example.org",
        "has_vault": False,
    }
    created = client.post("/account/vault", headers=_bearer(token))
    assert created.status_code == 200
    vault = created.json()
    assert vault["token"].startswith("moose_pt_") and vault["recovery_code"].startswith("moose_rc_")

    account = client.get("/account", headers=_bearer(token)).json()
    assert account["has_vault"] and account["user_id"] == vault["user_id"]
    assert client.post("/account/vault", headers=_bearer(token)).status_code == 409

    # The vault token works like any other API token.
    me = client.get("/vault/me", headers={"X-API-Key": vault["token"]})
    assert me.status_code == 200 and me.json()["user_id"] == vault["user_id"]


def test_accounts_get_separate_vaults(client: TestClient) -> None:
    first = client.post("/account/vault", headers=_bearer(_token(sub="a"))).json()
    second = client.post("/account/vault", headers=_bearer(_token(sub="b"))).json()
    assert first["user_id"] != second["user_id"]


@pytest.mark.parametrize(
    "token, status",
    [
        (_token(exp=int(time.time()) - 10), 401),
        (_token(iss="http://evil.test/realms/moose"), 401),
        (_token(azp="some-other-app"), 401),
        (_token(_key=OTHER_KEY), 401),
        (_token(roles=["offline_access"]), 403),
    ],
    ids=["expired", "wrong-issuer", "other-client", "forged-signature", "missing-role"],
)
def test_bad_login_tokens_are_rejected(client: TestClient, token: str, status: int) -> None:
    assert client.get("/account", headers=_bearer(token)).status_code == status
    assert client.post("/account/vault", headers=_bearer(token)).status_code == status


def test_login_token_is_not_an_api_key(client: TestClient) -> None:
    assert client.get("/vault/me", headers={"X-API-Key": _token()}).status_code == 401
    assert client.get("/account").status_code == 401


def test_account_login_off_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {"MOOSE_API_KEY": "k", "MOOSE_MONGO_URL": "", "MOOSE_WORKER_COUNT": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MOOSE_OIDC_ISSUER", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as test_client:
            assert test_client.get("/account", headers=_bearer(_token())).status_code == 503
    finally:
        get_settings.cache_clear()
