from __future__ import annotations

import jwt
import pytest

from oidc_helpers import SIGNING_KEY


@pytest.fixture
def test_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make token verification use the test signing key instead of fetching Keycloak's."""

    class _SigningKey:
        key = SIGNING_KEY.public_key()

    monkeypatch.setattr(jwt.PyJWKClient, "get_signing_key_from_jwt", lambda self, token: _SigningKey())
