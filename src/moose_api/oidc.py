"""Verify OpenID Connect access tokens from the account provider (Keycloak).

Moose never sees users' passwords: the dashboard logs users in with Keycloak and
passes their access token, which Moose verifies itself (signature, issuer, expiry,
client and role) before linking the account to a vault.
"""

from __future__ import annotations

import asyncio
from typing import Any

import jwt


class OIDCError(Exception):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


class OIDCVerifier:
    def __init__(self, issuer: str, jwks_url: str | None, client_ids: list[str]) -> None:
        self.issuer = issuer.rstrip("/")
        self.client_ids = client_ids
        self._jwks = jwt.PyJWKClient(
            jwks_url or f"{self.issuer}/protocol/openid-connect/certs",
            lifespan=300,
            # Proxies such as Cloudflare reject Python's default "Python-urllib" user agent.
            headers={"User-Agent": "moose (+https://github.com/DATAPACT/moose)"},
        )

    def _verify(self, token: str, role: str | None) -> dict[str, Any]:
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256", "ES256", "PS256"],
                issuer=self.issuer,
                # Keycloak access tokens name the requesting client in `azp`; `aud` is
                # usually "account", so the client is checked below instead.
                options={"verify_aud": False, "require": ["exp", "iat", "iss", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise OIDCError(f"Invalid login token: {exc}") from exc
        if self.client_ids and claims.get("azp") not in self.client_ids:
            raise OIDCError("Login token was issued to another application.")
        roles = (claims.get("realm_access") or {}).get("roles") or []
        if role and role not in roles:
            raise OIDCError(f"Your account lacks the '{role}' role.", status_code=403)
        return claims

    async def verify(self, token: str, role: str | None) -> dict[str, Any]:
        """Claims of a valid token whose account has `role` (a realm role)."""
        # Fetching the signing keys is blocking I/O.
        return await asyncio.to_thread(self._verify, token, role)
