"""Vault accounts: invitations, users, API tokens, recovery codes and encrypted
provider keys. Only hashes, wrapped data keys and ciphertext are stored."""

from __future__ import annotations

import secrets as pysecrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from moose import vault

INVITE_PREFIX = "moose_inv_"

# Providers users may bring keys for: name -> OpenAI-compatible base URL. A fixed list,
# so users cannot point the server at arbitrary (for example internal) addresses.
PROVIDERS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return pysecrets.token_hex(12)


@dataclass
class Unlocked:
    """A user's vault, unlocked in memory for one request."""

    user_id: str
    token_id: str
    data_key: bytes


@dataclass
class ProviderRoute:
    """Where and with which key to send a user's LLM calls. Lives in memory only."""

    provider: str
    base_url: str
    api_key: str
    secret_id: str


class MemoryCollection:
    """The subset of a Motor collection the vault uses, for tests and runs without MongoDB."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(doc.get(k) == v for k, v in query.items())

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next((dict(d) for d in self.docs.values() if self._matches(d, query)), None)

    async def insert_one(self, doc: dict[str, Any]) -> None:
        self.docs[doc["_id"]] = dict(doc)

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> None:
        for doc in self.docs.values():
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                return

    async def delete_one(self, query: dict[str, Any]) -> int:
        for key, doc in list(self.docs.items()):
            if self._matches(doc, query):
                del self.docs[key]
                return 1
        return 0

    async def find_all(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(d) for d in self.docs.values() if self._matches(d, query)]


class MongoCollection:
    def __init__(self, collection) -> None:
        self._c = collection

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return await self._c.find_one(query)

    async def insert_one(self, doc: dict[str, Any]) -> None:
        await self._c.insert_one(doc)

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> None:
        await self._c.update_one(query, update)

    async def delete_one(self, query: dict[str, Any]) -> int:
        return int((await self._c.delete_one(query)).deleted_count)

    async def find_all(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._c.find(query).to_list(length=None)


class VaultStore:
    def __init__(self, users, tokens, secrets, invites) -> None:
        self.users = users
        self.tokens = tokens
        self.secrets = secrets
        self.invites = invites

    @classmethod
    def in_memory(cls) -> "VaultStore":
        return cls(MemoryCollection(), MemoryCollection(), MemoryCollection(), MemoryCollection())

    @classmethod
    async def mongo(cls, db) -> "VaultStore":
        await db["vault_tokens"].create_index("hash", unique=True)
        await db["vault_users"].create_index("recovery_hash", unique=True)
        await db["vault_secrets"].create_index("user_id")
        return cls(
            MongoCollection(db["vault_users"]),
            MongoCollection(db["vault_tokens"]),
            MongoCollection(db["vault_secrets"]),
            MongoCollection(db["vault_invites"]),
        )

    # Invitations (created by the admin; redeemed once by the new user)

    async def create_invite(self, note: str | None, valid_days: int = 7) -> dict[str, str]:
        code = INVITE_PREFIX + pysecrets.token_urlsafe(24)
        expires = (datetime.now(timezone.utc) + timedelta(days=valid_days)).isoformat()
        await self.invites.insert_one(
            {"_id": vault.credential_hash(code), "note": note, "created_at": _now(), "expires_at": expires, "used_at": None}
        )
        return {"invite_code": code, "expires_at": expires}

    async def register(self, invite_code: str, name: str) -> dict[str, str]:
        invite = await self.invites.find_one({"_id": vault.credential_hash(invite_code)})
        if not invite or invite.get("used_at") or invite["expires_at"] < _now():
            raise vault.VaultError("Invitation is invalid, used or expired.")
        await self.invites.update_one({"_id": invite["_id"]}, {"$set": {"used_at": _now()}})

        user_id = _new_id()
        data_key = vault.new_data_key()
        recovery_code = vault.new_recovery_code()
        await self.users.insert_one(
            {
                "_id": user_id,
                "name": name,
                "created_at": _now(),
                "recovery_hash": vault.credential_hash(recovery_code),
                "recovery_wrapped": vault.wrap_data_key(data_key, recovery_code),
            }
        )
        token = await self._add_token(user_id, data_key, "first token")
        return {"user_id": user_id, "token": token["token"], "token_id": token["token_id"], "recovery_code": recovery_code}

    # Tokens

    async def _add_token(self, user_id: str, data_key: bytes, label: str) -> dict[str, str]:
        token = vault.new_token()
        token_id = _new_id()
        await self.tokens.insert_one(
            {
                "_id": token_id,
                "user_id": user_id,
                "hash": vault.credential_hash(token),
                "wrapped": vault.wrap_data_key(data_key, token),
                "label": label,
                "created_at": _now(),
            }
        )
        return {"token": token, "token_id": token_id}

    async def unlock(self, token: str) -> Unlocked | None:
        if not token.startswith(vault.TOKEN_PREFIX):
            return None
        doc = await self.tokens.find_one({"hash": vault.credential_hash(token)})
        if not doc:
            return None
        data_key = vault.unwrap_data_key(doc["wrapped"], token)
        return Unlocked(user_id=doc["user_id"], token_id=doc["_id"], data_key=data_key)

    async def create_token(self, unlocked: Unlocked, label: str) -> dict[str, str]:
        return await self._add_token(unlocked.user_id, unlocked.data_key, label)

    async def list_tokens(self, user_id: str) -> list[dict[str, Any]]:
        docs = await self.tokens.find_all({"user_id": user_id})
        return [{"token_id": d["_id"], "label": d["label"], "created_at": d["created_at"]} for d in docs]

    async def revoke_token(self, user_id: str, token_id: str) -> bool:
        return await self.tokens.delete_one({"_id": token_id, "user_id": user_id}) == 1

    async def recover(self, recovery_code: str, label: str) -> dict[str, str]:
        user = await self.users.find_one({"recovery_hash": vault.credential_hash(recovery_code)})
        if not user:
            raise vault.VaultError("Recovery code is not valid.")
        data_key = vault.unwrap_data_key(user["recovery_wrapped"], recovery_code)
        return await self._add_token(user["_id"], data_key, label)

    # Provider keys

    async def add_secret(
        self, unlocked: Unlocked, provider: str, api_key: str, label: str, default_model: str | None
    ) -> dict[str, Any]:
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
        secret_id = _new_id()
        doc = {
            "_id": secret_id,
            "user_id": unlocked.user_id,
            "provider": provider,
            "label": label,
            "masked": vault.mask(api_key),
            "default_model": default_model,
            "sealed": vault.encrypt(unlocked.data_key, api_key, context=f"{unlocked.user_id}:{secret_id}"),
            "created_at": _now(),
        }
        await self.secrets.insert_one(doc)
        return self._public(doc)

    @staticmethod
    def _public(doc: dict[str, Any]) -> dict[str, Any]:
        return {k: doc.get(k) for k in ("provider", "label", "masked", "default_model", "created_at")} | {
            "key_id": doc["_id"]
        }

    async def list_secrets(self, user_id: str) -> list[dict[str, Any]]:
        return [self._public(d) for d in await self.secrets.find_all({"user_id": user_id})]

    async def delete_secret(self, user_id: str, secret_id: str) -> bool:
        return await self.secrets.delete_one({"_id": secret_id, "user_id": user_id}) == 1

    async def route(self, unlocked: Unlocked, secret_id: str) -> tuple[ProviderRoute, dict[str, Any]]:
        doc = await self.secrets.find_one({"_id": secret_id, "user_id": unlocked.user_id})
        if not doc:
            raise KeyError(secret_id)
        api_key = vault.decrypt(unlocked.data_key, doc["sealed"], context=f"{unlocked.user_id}:{secret_id}")
        route = ProviderRoute(
            provider=doc["provider"], base_url=PROVIDERS[doc["provider"]], api_key=api_key, secret_id=secret_id
        )
        return route, self._public(doc)
