"""Account login (Keycloak via Streamlit's st.login) and the user's vault."""

from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def login_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:  # noqa: BLE001 - no secrets file at all
        return False


def access_token() -> str | None:
    """The logged-in account's access token, passed to the Moose API, which verifies it."""
    if not login_configured() or not st.user.is_logged_in:
        return None
    try:
        return st.user.tokens["access"]
    except Exception:  # noqa: BLE001
        return None


def account_request(method: str, base_url: str, path: str) -> httpx.Response:
    token = access_token() or ""
    with httpx.Client(timeout=30.0) as client:
        return client.request(method, f"{base_url.rstrip('/')}{path}", headers={"Authorization": f"Bearer {token}"})


def vault_request(method: str, base_url: str, api_key: str, path: str, payload: dict | None = None) -> httpx.Response:
    with httpx.Client(timeout=30.0) as client:
        return client.request(method, f"{base_url.rstrip('/')}{path}", headers={"X-API-Key": api_key}, json=payload)


@st.cache_data(ttl=3600)
def openrouter_catalogue() -> list[dict[str, Any]]:
    """OpenRouter's public model list with prices (no key needed)."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(OPENROUTER_MODELS_URL)
        resp.raise_for_status()
    models = []
    for item in resp.json().get("data", []):
        pricing = item.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt") or 0) * 1e6
            completion = float(pricing.get("completion") or 0) * 1e6
        except (TypeError, ValueError):
            prompt = completion = 0.0
        models.append({"id": item["id"], "prompt_per_m": prompt, "completion_per_m": completion})
    return sorted(models, key=lambda m: m["id"])


def model_label(model: dict[str, Any]) -> str:
    if model["prompt_per_m"] == 0 and model["completion_per_m"] == 0:
        return f"{model['id']} — free"
    return f"{model['id']} — ${model['prompt_per_m']:.2f} in / ${model['completion_per_m']:.2f} out per M tokens"
