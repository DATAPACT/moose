from __future__ import annotations

import time

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

# Stand-in for the account provider (Keycloak): tokens signed with a test key.
ISSUER = "http://keycloak.test/realms/moose"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def login_token(roles: list[str] | None = None, **overrides) -> str:
    """An access token as Keycloak would issue it to the dashboard."""
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": "account-1",
        "azp": "moose-dashboard",
        "typ": "Bearer",
        "iat": now,
        "exp": now + 300,
        "name": "Partner One",
        "email": "one@example.org",
        "realm_access": {"roles": roles if roles is not None else ["moose-user"]},
    }
    key = overrides.pop("_key", SIGNING_KEY)
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def admin_login() -> dict[str, str]:
    return bearer(login_token(["moose-admin"], sub="admin-1", name="Admin"))
