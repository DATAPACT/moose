"""Shared models: the operator's own provider key and the models everyone may use.

Unlike users' vaults, this key must be usable whenever any user runs a job, so it
is encrypted with a server key (MOOSE_SECRET_KEY) that Moose holds. That is
appropriate because it is the operator's own key and money.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from moose import vault
from moose_api.vault_store import PROVIDERS, MemoryCollection, MongoCollection, ProviderRoute

_CONFIG_ID = "shared"
_KEY_CONTEXT = "shared-models:provider-key"


class SharedModelsError(Exception):
    pass


def _server_key(secret: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"moose-shared-models-v1").derive(
        secret.encode()
    )


class SharedModelsStore:
    def __init__(self, collection, secret_key: str | None) -> None:
        self._config = collection
        self._key = _server_key(secret_key) if secret_key else None

    @classmethod
    def in_memory(cls, secret_key: str | None) -> "SharedModelsStore":
        return cls(MemoryCollection(), secret_key)

    @classmethod
    def mongo(cls, db, secret_key: str | None) -> "SharedModelsStore":
        return cls(MongoCollection(db["shared_models"]), secret_key)

    async def _doc(self) -> dict[str, Any]:
        return await self._config.find_one({"_id": _CONFIG_ID}) or {"_id": _CONFIG_ID}

    async def _save(self, doc: dict[str, Any]) -> None:
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        if await self._config.find_one({"_id": _CONFIG_ID}):
            await self._config.update_one({"_id": _CONFIG_ID}, {"$set": doc})
        else:
            await self._config.insert_one(doc)

    async def describe(self) -> dict[str, Any]:
        doc = await self._doc()
        key_readable = False
        if self._key is not None and doc.get("sealed"):
            try:
                vault.decrypt(self._key, doc["sealed"], _KEY_CONTEXT)
                key_readable = True
            except vault.VaultError:
                # Stored under a different MOOSE_SECRET_KEY; it must be entered again.
                pass
        return {
            "provider": doc.get("provider") if key_readable else None,
            "masked_key": doc.get("masked") if key_readable else None,
            "key_unreadable": bool(doc.get("sealed")) and not key_readable,
            "models": list(doc.get("models") or []),
            "default_model": doc.get("default_model"),
            "secret_key_configured": self._key is not None,
        }

    async def set_key(self, provider: str, api_key: str) -> dict[str, Any]:
        if self._key is None:
            raise SharedModelsError("Set MOOSE_SECRET_KEY on the server before storing the shared provider key.")
        if provider not in PROVIDERS:
            raise SharedModelsError(f"Unsupported provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
        doc = await self._doc()
        doc.update(provider=provider, masked=vault.mask(api_key), sealed=vault.encrypt(self._key, api_key, _KEY_CONTEXT))
        await self._save(doc)
        return await self.describe()

    async def set_models(self, models: list[str], default_model: str | None) -> dict[str, Any]:
        models = list(dict.fromkeys(m.strip() for m in models if m.strip()))
        if default_model and default_model not in models:
            raise SharedModelsError("The default model must be one of the shared models.")
        doc = await self._doc()
        doc.update(models=models, default_model=default_model or (models[0] if models else None))
        await self._save(doc)
        return await self.describe()

    async def models(self) -> tuple[list[str], str | None]:
        doc = await self._doc()
        return list(doc.get("models") or []), doc.get("default_model")

    async def route(self) -> ProviderRoute:
        """The operator's key, decrypted in memory for one job."""
        doc = await self._doc()
        if self._key is None or not doc.get("sealed"):
            raise SharedModelsError("Shared models are not set up: the administrator must add a provider key.")
        api_key = vault.decrypt(self._key, doc["sealed"], _KEY_CONTEXT)
        return ProviderRoute(provider=doc["provider"], base_url=PROVIDERS[doc["provider"]], api_key=api_key, secret_id="shared")
