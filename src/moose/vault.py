"""Per-user vault: envelope encryption for user-owned secrets.

Each user has one random data key. Secrets are encrypted with it (AES-256-GCM).
The data key is stored only wrapped, once per unlock credential (an API token or
the recovery code), with a wrapping key derived from that credential (HKDF-SHA256).
The server stores token hashes and wrapped keys, never a credential, so it cannot
decrypt anything without the user. See docs/design/partner-keys.md.
"""

from __future__ import annotations

import base64
import hashlib
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

TOKEN_PREFIX = "moose_pt_"
RECOVERY_PREFIX = "moose_rc_"
_WRAP_INFO = b"moose-vault-wrap-v1"


class VaultError(Exception):
    """A credential does not unlock the vault, or stored data was tampered with."""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode()


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode())


def new_data_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def new_recovery_code() -> str:
    return RECOVERY_PREFIX + secrets.token_urlsafe(32)


def credential_hash(credential: str) -> str:
    """Lookup hash of a token or recovery code. Credentials have 256 bits of entropy,
    so a fast hash is enough; it is never used as an encryption key."""
    return hashlib.sha256(credential.encode()).hexdigest()


def _wrapping_key(credential: str, salt: bytes) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=_WRAP_INFO)
    return hkdf.derive(credential.encode())


def wrap_data_key(data_key: bytes, credential: str) -> dict[str, str]:
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(_wrapping_key(credential, salt)).encrypt(nonce, data_key, _WRAP_INFO)
    return {"salt": _b64(salt), "nonce": _b64(nonce), "ct": _b64(ciphertext)}


def unwrap_data_key(wrapped: dict[str, str], credential: str) -> bytes:
    try:
        key = _wrapping_key(credential, _unb64(wrapped["salt"]))
        return AESGCM(key).decrypt(_unb64(wrapped["nonce"]), _unb64(wrapped["ct"]), _WRAP_INFO)
    except (InvalidTag, KeyError, ValueError) as exc:
        raise VaultError("Credential does not unlock this vault.") from exc


def encrypt(data_key: bytes, plaintext: str, context: str) -> dict[str, str]:
    """Encrypt a secret. `context` (for example the secret's id) is bound to the
    ciphertext, so a stored value cannot be swapped into another record."""
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(data_key).encrypt(nonce, plaintext.encode(), context.encode())
    return {"nonce": _b64(nonce), "ct": _b64(ciphertext)}


def decrypt(data_key: bytes, sealed: dict[str, str], context: str) -> str:
    try:
        return AESGCM(data_key).decrypt(_unb64(sealed["nonce"]), _unb64(sealed["ct"]), context.encode()).decode()
    except (InvalidTag, KeyError, ValueError) as exc:
        raise VaultError("Stored secret cannot be decrypted.") from exc


def mask(secret: str) -> str:
    return f"{secret[:6]}…{secret[-4:]}" if len(secret) > 12 else "…"
