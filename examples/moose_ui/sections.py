"""Sections of the Account page. Which ones show depends on the account's roles; the API
checks the role again on every request, so hiding a section is only for convenience."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .account import account_request, admin_request, model_label, openrouter_catalogue, vault_request
from .metadata import fetch_models


def _show_once(title: str, values: dict[str, str]) -> None:
    st.warning(f"{title} Copy these now: they are shown only once and Moose cannot show or recover them.")
    for label, value in values.items():
        st.text(label)
        st.code(value, language=None)


def _model_picker(label: str, current: str | None) -> str:
    """OpenRouter catalogue with prices, searchable; free text if it cannot be loaded."""
    try:
        catalogue = openrouter_catalogue()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load OpenRouter's model list: {exc}")
        return st.text_input(label, value=current or "").strip()
    choice = st.selectbox(
        label,
        [None, *catalogue],
        index=next((i + 1 for i, m in enumerate(catalogue) if m["id"] == current), 0),
        format_func=lambda m: "—" if m is None else model_label(m),
    )
    return choice["id"] if choice else ""


# --- Vault (accounts with the user role) -------------------------------------
def vault_section(base_url: str, account: dict[str, Any]) -> None:
    st.header("My vault")
    if not account["has_vault"]:
        st.write(
            "Your vault holds your own provider keys, encrypted. You get an API token (to unlock it and "
            "call Moose) and a recovery code (to get a new token if you lose all of them)."
        )
        if st.button("Create my vault", type="primary"):
            created = account_request("POST", base_url, "/account/vault")
            created.raise_for_status()
            data = created.json()
            st.session_state["api_key"] = data["token"]
            st.session_state["vault_new_secrets"] = {"API token": data["token"], "Recovery code": data["recovery_code"]}
            st.rerun()
        return

    if new_secrets := st.session_state.pop("vault_new_secrets", None):
        _show_once("Your vault is ready.", new_secrets)

    api_key = st.session_state.get("api_key", "")
    me = vault_request("GET", base_url, api_key, "/vault/me") if api_key else None
    if me is None or me.status_code != 200 or me.json()["user_id"] != account["user_id"]:
        _unlock_form(base_url, account)
        return

    vault = me.json()
    cols = st.columns([4, 1])
    cols[0].success("Vault unlocked for this session.")
    if cols[1].button("Lock"):
        st.session_state["api_key"] = ""
        st.rerun()
    _provider_keys(base_url, api_key, vault)
    _api_tokens(base_url, api_key, vault)


def _unlock_form(base_url: str, account: dict[str, Any]) -> None:
    credential = st.text_input("Unlock with an API token or the recovery code", type="password").strip()
    if not (st.button("Unlock", type="primary") and credential):
        return
    if credential.startswith("moose_rc_"):
        recovered = vault_request("POST", base_url, "", "/vault/recover", {"recovery_code": credential, "label": "dashboard"})
        if recovered.status_code != 200:
            st.error("That recovery code does not unlock a vault.")
            return
        token = recovered.json()["token"]
        st.session_state["vault_new_secrets"] = {"New API token": token}
    else:
        token = credential
    check = vault_request("GET", base_url, token, "/vault/me")
    if check.status_code != 200 or check.json()["user_id"] != account["user_id"]:
        st.error("That token does not unlock your vault.")
        return
    st.session_state["api_key"] = token
    st.rerun()


def _provider_keys(base_url: str, api_key: str, vault: dict[str, Any]) -> None:
    st.subheader("My provider keys")
    st.caption("Stored encrypted; only a masked form is ever shown. Jobs on a key are billed to your provider account.")
    for key in vault["keys"]:
        cols = st.columns([4, 1])
        cols[0].write(
            f"**{key['label']}** · {key['provider']} · `{key['masked']}`"
            + (f" · default model `{key['default_model']}`" if key.get("default_model") else "")
        )
        if cols[1].button("Delete", key=f"del-key-{key['key_id']}"):
            vault_request("DELETE", base_url, api_key, f"/vault/keys/{key['key_id']}")
            fetch_models.clear()
            st.rerun()

    with st.expander("Add a provider key", expanded=not vault["keys"]):
        providers = vault_request("GET", base_url, api_key, "/vault/providers").json()["providers"]
        provider = st.selectbox("Provider", providers, key="own-provider")
        label = st.text_input("Label", value=f"my {provider} key")
        secret = st.text_input("API key", type="password", key="own-secret")
        if provider == "openrouter":
            default_model = _model_picker("Default model (type to search)", None)
        else:
            default_model = st.text_input("Default model", placeholder="e.g. gpt-4.1-mini").strip()
        if st.button("Save key", type="primary", key="own-save"):
            added = vault_request(
                "POST",
                base_url,
                api_key,
                "/vault/keys",
                {"provider": provider, "api_key": secret.strip(), "label": label, "default_model": default_model or None},
            )
            if added.status_code == 200:
                fetch_models.clear()
                st.rerun()
            st.error(added.json().get("detail", added.text))


def _api_tokens(base_url: str, api_key: str, vault: dict[str, Any]) -> None:
    st.subheader("My API tokens")
    st.caption("Each token unlocks your vault and calls Moose. Use one per script or machine; revoke any you no longer need.")
    for token in vault["tokens"]:
        cols = st.columns([4, 1])
        current = " (this session)" if token["token_id"] == vault["current_token_id"] else ""
        cols[0].write(f"**{token['label']}**{current} · created {token['created_at'][:10]}")
        if not current and cols[1].button("Revoke", key=f"revoke-{token['token_id']}"):
            vault_request("DELETE", base_url, api_key, f"/vault/tokens/{token['token_id']}")
            st.rerun()

    with st.form("new-token"):
        token_label = st.text_input("Label for a new token", placeholder="e.g. analysis notebook")
        if st.form_submit_button("Create token") and token_label:
            created = vault_request("POST", base_url, api_key, "/vault/tokens", {"label": token_label})
            created.raise_for_status()
            st.session_state["vault_new_secrets"] = {f"API token '{token_label}'": created.json()["token"]}
            st.rerun()


# --- Shared models (accounts with the admin role) ----------------------------
def shared_models_section(base_url: str) -> None:
    st.header("Shared models")
    st.caption(
        "Models every user may use, on your own provider key and at your cost. Users' own keys are separate "
        "and stay in their vaults."
    )
    resp = admin_request("GET", base_url, "/admin/shared-models")
    if resp.status_code != 200:
        st.error(resp.json().get("detail", resp.text))
        return
    shared = resp.json()
    if not shared["secret_key_configured"]:
        st.error("Set MOOSE_SECRET_KEY on the server first; it encrypts the shared provider key.")
        return

    if shared.get("key_unreadable"):
        st.warning("The stored provider key was encrypted with a different MOOSE_SECRET_KEY. Enter it again below.")
    if shared.get("masked_key"):
        st.success(f"Provider key stored: {shared['provider']} · `{shared['masked_key']}`")
    with st.expander(
        "Replace provider key" if shared.get("masked_key") else "Add provider key", expanded=not shared.get("masked_key")
    ):
        with st.form("shared-key"):
            providers = admin_request("GET", base_url, "/vault/providers").json()["providers"]
            provider = st.selectbox(
                "Provider", providers, index=providers.index("openrouter") if "openrouter" in providers else 0
            )
            secret = st.text_input("API key", type="password")
            st.caption("Tip: set a credit limit on this key in your provider's dashboard to cap Moose's spending.")
            if st.form_submit_button("Save key", type="primary") and secret.strip():
                saved = admin_request(
                    "PUT", base_url, "/admin/shared-models/key", {"provider": provider, "api_key": secret.strip()}
                )
                if saved.status_code == 200:
                    st.rerun()
                st.error(saved.json().get("detail", saved.text))

    current = shared.get("models") or []
    try:
        catalogue = openrouter_catalogue() if shared.get("provider") in (None, "openrouter") else []
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load OpenRouter's model list: {exc}")
        catalogue = []
    if catalogue:
        by_id = {m["id"]: m for m in catalogue}
        selected = st.multiselect(
            "Shared models (type to search)",
            sorted(set(by_id) | set(current)),
            default=current,
            format_func=lambda m: model_label(by_id[m]) if m in by_id else m,
        )
    else:
        selected = [m.strip() for m in st.text_area("Shared models, one per line", "\n".join(current)).splitlines() if m.strip()]

    default = None
    if selected:
        index = selected.index(shared["default_model"]) if shared.get("default_model") in selected else 0
        default = st.selectbox("Default model", selected, index=index)
    if st.button("Save models", type="primary"):
        saved = admin_request("PUT", base_url, "/admin/shared-models", {"models": selected, "default_model": default})
        if saved.status_code == 200:
            fetch_models.clear()
            st.success("Saved. Users see these models in the sidebar.")
        else:
            st.error(saved.json().get("detail", saved.text))
