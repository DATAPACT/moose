from __future__ import annotations

from typing import Any

import streamlit as st

from .api import api_get


@st.cache_data(ttl=60)
def fetch_models(base_url: str, api_key: str) -> dict[str, Any]:
    return api_get(base_url, api_key, "/models")


@st.cache_data(ttl=15)
def fetch_health(base_url: str, api_key: str) -> dict[str, Any]:
    return api_get(base_url, api_key, "/health")


@st.cache_data(ttl=60)
def fetch_schemas(base_url: str, api_key: str) -> list[dict[str, Any]]:
    data = api_get(base_url, api_key, "/schemas", params={"include_type_count": "true"})
    return data.get("schemas", []) or []


@st.cache_data(ttl=60)
def fetch_policy_packs(base_url: str, api_key: str) -> list[str]:
    data = api_get(base_url, api_key, "/policy-packs")
    return data.get("policy_packs", []) or []


@st.cache_data(ttl=60)
def fetch_privacy_profiles(base_url: str, api_key: str) -> dict[str, Any]:
    return api_get(base_url, api_key, "/privacy/profiles")


@st.cache_data(ttl=60)
def fetch_assets(base_url: str, api_key: str) -> dict[str, Any]:
    return api_get(base_url, api_key, "/assets")


def clear_metadata_caches() -> None:
    fetch_health.clear()
    fetch_schemas.clear()
    fetch_policy_packs.clear()
    fetch_privacy_profiles.clear()
    fetch_assets.clear()


def schemas_supporting(
    schemas: list[dict[str, Any]], *, text: bool = False, table: bool = False, cpa: bool = False
) -> list[str]:
    out: list[str] = []
    for s in schemas:
        if not isinstance(s, dict):
            continue
        if text and not s.get("supports_text"):
            continue
        if table and not s.get("supports_table"):
            continue
        if cpa and not s.get("supports_cpa"):
            continue
        name = s.get("name")
        if isinstance(name, str):
            out.append(name)
    return sorted(set(out))


def schemas_supporting_full(
    schemas: list[dict[str, Any]], *, text: bool = False, table: bool = False, cpa: bool = False
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in schemas:
        if not isinstance(s, dict):
            continue
        if text and not s.get("supports_text"):
            continue
        if table and not s.get("supports_table"):
            continue
        if cpa and not s.get("supports_cpa"):
            continue
        out.append(s)
    return out


def prefer_schema(names: list[str], preferred: list[str]) -> str:
    for p in preferred:
        if p in names:
            return p
    return names[0] if names else ""
