from __future__ import annotations

import streamlit as st

from moose_ui.account import (
    account_request,
    login_configured,
    model_label,
    openrouter_catalogue,
    vault_request,
)
from moose_ui.config import sidebar
from moose_ui.metadata import fetch_models

st.title("Account")

cfg = sidebar()
base_url = cfg["base_url"]

if not login_configured():
    st.info("Account login is not configured on this server. Use an API token in the sidebar.")
    st.stop()

if not st.user.is_logged_in:
    st.write("Log in to manage your own LLM provider keys. Moose stores them encrypted; only your API tokens can unlock them.")
    st.button("Log in", type="primary", on_click=st.login)
    st.stop()

resp = account_request("GET", base_url, "/account")
if resp.status_code == 401:
    st.warning("Your login has expired.")
    st.button("Log in again", on_click=st.logout)
    st.stop()
if resp.status_code == 403:
    st.error(resp.json().get("detail", "Your account may not use Moose."))
    st.button("Log out", on_click=st.logout)
    st.stop()
resp.raise_for_status()
account = resp.json()

st.caption(f"Logged in as {account['name']}")
st.button("Log out", on_click=st.logout)


def show_once(title: str, values: dict[str, str]) -> None:
    st.warning(f"{title} Copy these now: they are shown only once and Moose cannot show or recover them.")
    for label, value in values.items():
        st.text(label)
        st.code(value, language=None)


# --- Vault: create or unlock ---------------------------------------------------
if not account["has_vault"]:
    st.subheader("Create your vault")
    st.write(
        "Your vault holds your provider keys, encrypted. You get an API token (to unlock it and call Moose) "
        "and a recovery code (to get a new token if you lose all of them)."
    )
    if st.button("Create my vault", type="primary"):
        created = account_request("POST", base_url, "/account/vault")
        created.raise_for_status()
        data = created.json()
        st.session_state["api_key"] = data["token"]
        st.session_state["vault_new_secrets"] = {"API token": data["token"], "Recovery code": data["recovery_code"]}
        st.rerun()
    st.stop()

if new_secrets := st.session_state.pop("vault_new_secrets", None):
    show_once("Your vault is ready.", new_secrets)

api_key = st.session_state.get("api_key", "")
me = vault_request("GET", base_url, api_key, "/vault/me") if api_key else None
unlocked = me is not None and me.status_code == 200 and me.json()["user_id"] == account["user_id"]

if not unlocked:
    st.subheader("Unlock your vault")
    credential = st.text_input("API token or recovery code", type="password").strip()
    if st.button("Unlock", type="primary") and credential:
        if credential.startswith("moose_rc_"):
            recovered = vault_request("POST", base_url, "", "/vault/recover", {"recovery_code": credential, "label": "dashboard"})
            if recovered.status_code != 200:
                st.error("That recovery code does not unlock a vault.")
                st.stop()
            token = recovered.json()["token"]
            st.session_state["vault_new_secrets"] = {"New API token": token}
        else:
            token = credential
        check = vault_request("GET", base_url, token, "/vault/me")
        if check.status_code != 200 or check.json()["user_id"] != account["user_id"]:
            st.error("That token does not unlock your vault.")
            st.stop()
        st.session_state["api_key"] = token
        st.rerun()
    st.stop()

vault = me.json()
st.success("Vault unlocked for this session.")
if st.button("Lock"):
    st.session_state["api_key"] = ""
    st.rerun()

# --- Provider keys ------------------------------------------------------------
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
    provider = st.selectbox("Provider", providers)
    label = st.text_input("Label", value=f"my {provider} key")
    secret = st.text_input("API key", type="password")
    default_model = ""
    if provider == "openrouter":
        try:
            catalogue = openrouter_catalogue()
            choice = st.selectbox(
                "Default model (type to search)",
                [None, *catalogue],
                format_func=lambda m: "—" if m is None else model_label(m),
            )
            default_model = choice["id"] if choice else ""
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Could not load OpenRouter's model list: {exc}")
            default_model = st.text_input("Default model")
    else:
        default_model = st.text_input("Default model", placeholder="e.g. gpt-4.1-mini")
    if st.button("Save key", type="primary"):
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

# --- API tokens ---------------------------------------------------------------
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
