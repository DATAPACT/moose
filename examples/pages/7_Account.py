from __future__ import annotations

import streamlit as st

from moose_ui.account import account_request, auth_disabled_banner, auth_enabled, logged_in, login_configured
from moose_ui.config import sidebar
from moose_ui.sections import shared_models_section, vault_section

st.title("Account")

cfg = sidebar()
base_url = cfg["base_url"]
auth_disabled_banner()

if not login_configured():
    st.info("Account login is not configured on this server. Use an API token in the sidebar.")
    st.stop()

if not logged_in():
    st.write("Log in to manage your own LLM provider keys, or the shared models if you are an administrator.")
    st.button("Log in", type="primary", on_click=st.login)
    st.stop()

resp = account_request("GET", base_url, "/account")
if resp.status_code == 401:
    st.warning("Your login has expired.")
    st.button("Log in again", on_click=st.logout)
    st.stop()
resp.raise_for_status()
account = resp.json()

cols = st.columns([4, 1])
cols[0].caption(f"Logged in as {account['name']}" + (" · administrator" if account["is_admin"] else ""))
if auth_enabled():
    cols[1].button("Log out", on_click=st.logout)

# Sections follow the account's roles: users manage their vault, administrators the
# shared models. An account can have both.
if account["can_use_vault"]:
    vault_section(base_url, account)
if account["is_admin"]:
    shared_models_section(base_url)
if not (account["can_use_vault"] or account["is_admin"]):
    st.error("Your account has no access to Moose yet. Ask an administrator to give it the moose-user role.")
