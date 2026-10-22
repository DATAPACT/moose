from __future__ import annotations

import streamlit as st

from moose_ui.account import model_label, openrouter_catalogue, vault_request
from moose_ui.config import sidebar
from moose_ui.metadata import fetch_models

st.title("Admin")

cfg = sidebar()
base_url, api_key = cfg["base_url"], cfg["api_key"]

if not api_key:
    st.info("Enter the Moose admin key in the sidebar.")
    st.stop()

resp = vault_request("GET", base_url, api_key, "/admin/shared-models")
if resp.status_code in (401, 403):
    st.warning("This page needs the Moose admin key.")
    st.stop()
resp.raise_for_status()
shared = resp.json()

st.subheader("Shared models")
st.write(
    "Models every user may use, on your own provider key and at your cost. "
    "Users' own keys are separate and stay in their vaults."
)

# --- Provider key -------------------------------------------------------------
if not shared["secret_key_configured"]:
    st.error("Set MOOSE_SECRET_KEY on the server first; it encrypts the shared provider key.")
    st.stop()

if shared.get("masked_key"):
    st.success(f"Provider key stored: {shared['provider']} · `{shared['masked_key']}`")
with st.expander("Replace provider key" if shared.get("masked_key") else "Add provider key", expanded=not shared.get("masked_key")):
    with st.form("shared-key"):
        providers = vault_request("GET", base_url, api_key, "/vault/providers").json()["providers"]
        provider = st.selectbox("Provider", providers, index=providers.index("openrouter") if "openrouter" in providers else 0)
        secret = st.text_input("API key", type="password")
        st.caption("Tip: set a credit limit on this key in your provider's dashboard to cap Moose's spending.")
        if st.form_submit_button("Save key", type="primary") and secret.strip():
            saved = vault_request("PUT", base_url, api_key, "/admin/shared-models/key", {"provider": provider, "api_key": secret.strip()})
            if saved.status_code == 200:
                st.rerun()
            st.error(saved.json().get("detail", saved.text))

# --- Model list ---------------------------------------------------------------
st.subheader("Which models")
current = shared.get("models") or []
selected: list[str] = current
if shared.get("provider") in (None, "openrouter"):
    try:
        catalogue = openrouter_catalogue()
        by_id = {m["id"]: m for m in catalogue}
        options = sorted(set(by_id) | set(current))
        selected = st.multiselect(
            "Shared models (type to search)",
            options,
            default=[m for m in current if m in options],
            format_func=lambda m: model_label(by_id[m]) if m in by_id else m,
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load OpenRouter's model list: {exc}")
        selected = [m.strip() for m in st.text_area("Shared models, one per line", "\n".join(current)).splitlines()]
else:
    selected = [m.strip() for m in st.text_area("Shared models, one per line", "\n".join(current)).splitlines()]

default = None
if selected:
    index = selected.index(shared["default_model"]) if shared.get("default_model") in selected else 0
    default = st.selectbox("Default model", selected, index=index)

if st.button("Save models", type="primary"):
    saved = vault_request("PUT", base_url, api_key, "/admin/shared-models", {"models": selected, "default_model": default})
    if saved.status_code == 200:
        fetch_models.clear()
        st.success("Saved. Users see these models in the sidebar.")
    else:
        st.error(saved.json().get("detail", saved.text))
