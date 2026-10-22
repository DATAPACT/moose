"""Encrypt users' job inputs, results and error details with their vault's data key.

Only the job's owner can read them: the data key exists in memory only while one of
their API tokens unlocks the vault. Job metadata (status, timestamps, model, usage)
stays readable for operations. See docs/design/partner-keys.md, phase 3.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from moose import vault

SEALED = "_sealed"


def _context(job_id: str, kind: str) -> str:
    # Binds each ciphertext to its job and field, so it cannot be moved to another job.
    return f"job:{job_id}:{kind}"


def is_sealed(value: Any) -> bool:
    return isinstance(value, dict) and SEALED in value


def seal_payload(data_key: bytes, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Everything but the `llm` block (model, provider, key reference: no content)."""
    content = {k: v for k, v in payload.items() if k != "llm"}
    sealed: dict[str, Any] = {SEALED: vault.encrypt(data_key, json.dumps(content), _context(job_id, "payload"))}
    if "llm" in payload:
        sealed["llm"] = payload["llm"]
    return sealed


def open_payload(data_key: bytes, job_id: str, stored: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(vault.decrypt(data_key, stored[SEALED], _context(job_id, "payload")))
    if "llm" in stored:
        payload["llm"] = stored["llm"]
    return payload


def seal_value(data_key: bytes, job_id: str, kind: str, value: Any) -> dict[str, Any]:
    # Encoded as the API would return it, so any value a job may return can be sealed.
    return {SEALED: vault.encrypt(data_key, json.dumps(jsonable_encoder(value)), _context(job_id, kind))}


def open_value(data_key: bytes, job_id: str, kind: str, stored: dict[str, Any]) -> Any:
    return json.loads(vault.decrypt(data_key, stored[SEALED], _context(job_id, kind)))
