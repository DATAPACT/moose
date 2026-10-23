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


@pytest.fixture(autouse=True)
def no_system1_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run as on a server without Laya (so "auto" schemas annotate in one step and no
    model is loaded); a test that wants the System 1 path patches it back on."""
    import moose.linkers

    monkeypatch.setattr(moose.linkers, "laya_available", lambda: False)
