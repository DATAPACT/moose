from __future__ import annotations

import json

import pytest

from moose import vault
from moose_api.vault_store import VaultStore

PROVIDER_KEY = "sk-or-v1-user-owned-provider-key-0123456789"


def test_wrap_unwrap_round_trip_and_wrong_credential() -> None:
    data_key = vault.new_data_key()
    token = vault.new_token()
    wrapped = vault.wrap_data_key(data_key, token)
    assert vault.unwrap_data_key(wrapped, token) == data_key
    with pytest.raises(vault.VaultError):
        vault.unwrap_data_key(wrapped, vault.new_token())


def test_tampering_and_moved_ciphertext_are_detected() -> None:
    data_key = vault.new_data_key()
    sealed = vault.encrypt(data_key, PROVIDER_KEY, context="user1:key1")
    assert vault.decrypt(data_key, sealed, context="user1:key1") == PROVIDER_KEY
    # Copied into another user's record: the bound context no longer matches.
    with pytest.raises(vault.VaultError):
        vault.decrypt(data_key, sealed, context="user2:key1")
    flipped = dict(sealed, ct=sealed["ct"][:-4] + ("AAAA" if not sealed["ct"].endswith("AAAA") else "BBBB"))
    with pytest.raises(vault.VaultError):
        vault.decrypt(data_key, flipped, context="user1:key1")


async def _registered(store: VaultStore) -> tuple[dict[str, str], vault.VaultError | None]:
    invite = await store.create_invite("partner A")
    return await store.register(invite["invite_code"], "Partner A"), None


@pytest.mark.asyncio
async def test_invites_work_once() -> None:
    store = VaultStore.in_memory()
    invite = await store.create_invite(None)
    await store.register(invite["invite_code"], "first")
    with pytest.raises(vault.VaultError):
        await store.register(invite["invite_code"], "second")


@pytest.mark.asyncio
async def test_tokens_recovery_and_revocation() -> None:
    store = VaultStore.in_memory()
    account, _ = await _registered(store)
    unlocked = await store.unlock(account["token"])
    assert unlocked is not None
    await store.add_secret(unlocked, "openrouter", PROVIDER_KEY, "mine", "openai/gpt-oss-120b")

    second = await store.create_token(unlocked, "laptop")
    recovered = await store.recover(account["recovery_code"], "recovered")
    for token in (second["token"], recovered["token"]):
        other = await store.unlock(token)
        key_id = (await store.list_secrets(other.user_id))[0]["key_id"]
        route, _ = await store.route(other, key_id)
        assert route.api_key == PROVIDER_KEY

    assert await store.revoke_token(unlocked.user_id, account["token_id"])
    assert await store.unlock(account["token"]) is None


@pytest.mark.asyncio
async def test_operator_cannot_decrypt_stored_data() -> None:
    store = VaultStore.in_memory()
    account, _ = await _registered(store)
    unlocked = await store.unlock(account["token"])
    await store.add_secret(unlocked, "openrouter", PROVIDER_KEY, "mine", None)

    stored = {
        name: list(getattr(store, name).docs.values()) for name in ("users", "tokens", "secrets", "invites")
    }
    dump = json.dumps(stored)
    # No plaintext secret or credential is stored anywhere.
    for value in (PROVIDER_KEY, account["token"], account["recovery_code"]):
        assert value not in dump

    # Everything the operator holds (hashes, wrapped keys, ciphertext) unlocks nothing.
    token_doc = stored["tokens"][0]
    user_doc = stored["users"][0]
    for guess in (token_doc["hash"], user_doc["recovery_hash"], "admin-key", ""):
        for wrapped in (token_doc["wrapped"], user_doc["recovery_wrapped"]):
            with pytest.raises(vault.VaultError):
                vault.unwrap_data_key(wrapped, guess)


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected() -> None:
    store = VaultStore.in_memory()
    account, _ = await _registered(store)
    unlocked = await store.unlock(account["token"])
    with pytest.raises(ValueError):
        await store.add_secret(unlocked, "http://10.0.0.5", PROVIDER_KEY, "x", None)
