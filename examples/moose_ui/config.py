from __future__ import annotations

import os
from typing import Any

import streamlit as st

DEFAULT_BASE_URL = os.getenv("MOOSE_API_BASE_URL", "http://localhost:8000")


def sidebar() -> dict[str, Any]:
    st.sidebar.header("Connection")
    base_url = st.sidebar.text_input("Moose API base URL", value=st.session_state.get("base_url", DEFAULT_BASE_URL)).strip()
    # Strip pasted whitespace: a trailing space or newline makes the key fail with 401.
    api_key = st.sidebar.text_input(
        "API key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Your Moose API token, or the Moose admin key.",
    ).strip()

    st.sidebar.divider()
    st.sidebar.header("LLM")
    # Shared models set up by the administrator, plus the user's own provider keys.
    model, key_id = _model_picker(base_url, api_key)

    st.sidebar.divider()
    st.sidebar.header("UI")
    auto_poll_default = st.sidebar.checkbox("Auto-poll by default", value=st.session_state.get("auto_poll_default", True))
    advanced_mode = st.sidebar.checkbox("Advanced mode", value=st.session_state.get("advanced_mode", False))

    show_legal_refs = st.sidebar.checkbox("Show legal refs", value=st.session_state.get("show_legal_refs", True))
    show_legal_detail = st.sidebar.checkbox("Show legal ref details", value=st.session_state.get("show_legal_detail", True))
    show_debug = st.sidebar.checkbox("Show debug output", value=st.session_state.get("show_debug", False))
    show_raw = st.sidebar.checkbox("Show raw JSON", value=st.session_state.get("show_raw", False))

    with st.sidebar.expander("Advanced options", expanded=False):
        developer_mode = st.checkbox("Developer mode", value=st.session_state.get("developer_mode", False))

    # persist state
    st.session_state.update(
        {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "key_id": key_id,
            "developer_mode": developer_mode,
            "auto_poll_default": auto_poll_default,
            "advanced_mode": advanced_mode,
            "show_legal_refs": show_legal_refs,
            "show_legal_detail": show_legal_detail,
            "show_debug": show_debug,
            "show_raw": show_raw,
        }
    )

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "key_id": key_id,
        "developer_mode": developer_mode,
        "auto_poll_default": auto_poll_default,
        "advanced_mode": advanced_mode,
        "show_legal_refs": show_legal_refs,
        "show_legal_detail": show_legal_detail,
        "show_debug": show_debug,
        "show_raw": show_raw,
    }


def _model_picker(base_url: str, api_key: str) -> tuple[str, str]:
    """(model, key_id). key_id is set when the user runs on one of their own provider keys."""
    from .account import model_label, openrouter_catalogue
    from .metadata import fetch_models

    if not api_key:
        st.sidebar.caption("Enter your API key to see the available models.")
        return "", ""
    try:
        data = fetch_models(base_url, api_key)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.caption(f"Could not list models: {exc}")
        return "", ""
    shared = list(data.get("models") or [])
    own_keys = list(data.get("own_keys") or [])
    sources = [{"kind": "shared", "model": m} for m in shared] + [{"kind": "key", "key": k} for k in own_keys]
    if not sources:
        return "", ""

    def label(source: dict[str, Any]) -> str:
        if source["kind"] == "shared":
            return f"{source['model']} (shared)"
        key = source["key"]
        return f"🔑 {key['label']} ({key['provider']})"

    current = st.session_state.get("key_id") or st.session_state.get("model") or data.get("default_model")
    index = next(
        (i for i, s in enumerate(sources) if current in (s.get("model"), s.get("key", {}).get("key_id"))), 0
    )
    source = st.sidebar.selectbox("Model", sources, index=index, format_func=label)
    if source["kind"] == "shared":
        return source["model"], ""

    key = source["key"]
    default = key.get("default_model") or ""
    if key["provider"] == "openrouter":
        try:
            catalogue = openrouter_catalogue()
            ids = [m["id"] for m in catalogue]
            chosen = st.sidebar.selectbox(
                "Model on your key (type to search)",
                catalogue,
                index=ids.index(default) if default in ids else 0,
                format_func=model_label,
            )
            return chosen["id"], key["key_id"]
        except Exception as exc:  # noqa: BLE001
            st.sidebar.caption(f"Could not load OpenRouter's model list: {exc}")
    return st.sidebar.text_input("Model on your key", value=default).strip(), key["key_id"]


def validate_common(cfg: dict[str, Any]) -> str | None:
    if not cfg.get("api_key"):
        return "An API key is required."
    return None


def llm_options(cfg: dict[str, Any]) -> dict[str, str]:
    options = {"model": cfg["model"]} if cfg.get("model") else {}
    if cfg.get("key_id"):
        options["key_id"] = cfg["key_id"]
    return options
