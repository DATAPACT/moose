from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from moose.config import get_settings
from moose_api.main import app

ISSUER = "http://keycloak.test/realms/moose"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(**overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": "account-1",
        "azp": "moose-dashboard",
        "iat": now,
        "exp": now + 300,
        "name": "Partner One",
        "email": "one@example.org",
        "realm_access": {"roles": ["moose-user"]},
    }
    key = overrides.pop("_key", SIGNING_KEY)
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    class _SigningKey:
        key = SIGNING_KEY.public_key()

    monkeypatch.setattr(jwt.PyJWKClient, "get_signing_key_from_jwt", lambda self, token: _SigningKey())

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


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
        (_token(realm_access={"roles": ["offline_access"]}), 403),
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
            assert test_client.get("/account", headers=_bearer(_token())).status_code == 404
    finally:
        get_settings.cache_clear()
