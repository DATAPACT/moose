#!/bin/sh
# Start the Streamlit demo. When account login is configured, write Streamlit's
# [auth] settings from the environment (never baked into the image).
set -eu

auth_enabled="$(printf '%s' "${MOOSE_AUTH_ENABLED:-true}" | tr '[:upper:]' '[:lower:]')"
case "$auth_enabled" in false | 0 | no | off) auth_enabled=false ;; *) auth_enabled=true ;; esac

if [ "$auth_enabled" = true ] && [ -n "${MOOSE_DASHBOARD_OIDC_CLIENT_ID:-}" ]; then
    mkdir -p .streamlit
    umask 077
    python - > .streamlit/secrets.toml <<'PY'
import json
import os

def toml(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string

env = os.environ
print("[auth]")
print("redirect_uri =", toml(env["MOOSE_DASHBOARD_URL"].rstrip("/") + "/oauth2callback"))
print("cookie_secret =", toml(env["MOOSE_DASHBOARD_COOKIE_SECRET"]))
print("client_id =", toml(env["MOOSE_DASHBOARD_OIDC_CLIENT_ID"]))
print("client_secret =", toml(env["MOOSE_DASHBOARD_OIDC_CLIENT_SECRET"]))
print("server_metadata_url =", toml(env["MOOSE_DASHBOARD_OIDC_METADATA_URL"]))
# The access token is passed to the Moose API, which verifies it itself.
print('expose_tokens = ["access"]')
PY
fi

exec streamlit run examples/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
