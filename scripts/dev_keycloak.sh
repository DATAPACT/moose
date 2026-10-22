#!/bin/sh
# Local stand-in for the production Keycloak, for developing the dashboard login.
# Starts Keycloak on http://localhost:8180 with the `moose` realm, sets a random
# dashboard client secret, and creates a test user with a random password.
# Secrets are written to .env (client and cookie secrets) and data/dev-keycloak.txt
# (test user login), never printed.
#
#   ./scripts/dev_keycloak.sh          start or reuse
#   ./scripts/dev_keycloak.sh stop     remove the container
set -eu

cd "$(dirname "$0")/.."

NAME=moose-dev-keycloak
IMAGE=quay.io/keycloak/keycloak:26.8.1@sha256:79c623a22abacf3fe6c00a5fb30f5b7843d717b234033b80995891c06116d244
PORT=8180
ENV_FILE="${MOOSE_ENV_FILE:-.env}"

if [ "${1:-}" = "stop" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "Stopped $NAME."
    exit 0
fi

# Never replace a real account provider: refuse if the env file already points
# Moose at another Keycloak (for example the production one on a server).
configured_issuer="$(sed -n 's/^MOOSE_OIDC_ISSUER=//p' "$ENV_FILE" 2>/dev/null | tail -n 1)"
if [ -n "$configured_issuer" ] && [ "$configured_issuer" != "http://localhost:$PORT/realms/moose" ]; then
    echo "Refusing: $ENV_FILE already uses the account provider $configured_issuer." >&2
    echo "This script is for local development only and would overwrite that setting." >&2
    exit 1
fi

random() { python3 -c "import secrets; print(secrets.token_urlsafe(24))"; }

set_env() {
    # set_env NAME VALUE: replace or append NAME=VALUE in the env file.
    python3 - "$ENV_FILE" "$1" "$2" <<'EOF'
import re, sys
path, name, value = sys.argv[1:4]
text = open(path).read()
line = f"{name}={value}"
if re.search(rf"(?m)^{name}=", text):
    text = re.sub(rf"(?m)^{name}=.*$", lambda _: line, text)
else:
    text = text.rstrip("\n") + "\n" + line + "\n"
open(path, "w").write(text)
EOF
}

if ! docker inspect "$NAME" >/dev/null 2>&1; then
    admin_password="$(random)"
    docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8080" \
        -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
        -e KC_BOOTSTRAP_ADMIN_PASSWORD="$admin_password" \
        -e KC_HOSTNAME="http://localhost:$PORT" \
        -e KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true \
        -v "$PWD/deploy/keycloak/moose-realm.json:/opt/keycloak/data/import/moose-realm.json:ro" \
        "$IMAGE" start-dev --import-realm >/dev/null
    echo "Waiting for Keycloak..."
    until curl -fs "http://127.0.0.1:$PORT/realms/moose/.well-known/openid-configuration" >/dev/null 2>&1; do
        sleep 2
    done

    kc() { docker exec "$NAME" /opt/keycloak/bin/kcadm.sh "$@"; }
    kc config credentials --server http://localhost:8080 --realm master --user admin --password "$admin_password" >/dev/null

    client_secret="$(random)"
    client_uuid="$(kc get clients -r moose -q clientId=moose-dashboard --fields id --format csv --noquotes)"
    kc update "clients/$client_uuid" -r moose -s "secret=$client_secret"

    user_password="$(random)"
    kc create users -r moose -s username=test-partner -s email=test-partner@example.org \
        -s firstName=Test -s lastName=Partner -s enabled=true -s emailVerified=true >/dev/null
    kc set-password -r moose --username test-partner --new-password "$user_password"
    kc add-roles -r moose --uusername test-partner --rolename moose-user

    mkdir -p data
    umask 077
    printf 'Keycloak admin: admin / %s\nTest user: test-partner / %s\n' "$admin_password" "$user_password" > data/dev-keycloak.txt

    set_env MOOSE_OIDC_ISSUER "http://localhost:$PORT/realms/moose"
    set_env MOOSE_OIDC_JWKS_URL "http://host.docker.internal:$PORT/realms/moose/protocol/openid-connect/certs"
    set_env MOOSE_OIDC_CLIENT_IDS moose-dashboard
    set_env MOOSE_DASHBOARD_OIDC_METADATA_URL "http://host.docker.internal:$PORT/realms/moose/.well-known/openid-configuration"
    set_env MOOSE_DASHBOARD_OIDC_CLIENT_ID moose-dashboard
    set_env MOOSE_DASHBOARD_OIDC_CLIENT_SECRET "$client_secret"
    set_env MOOSE_DASHBOARD_URL "http://localhost:8502"
    set_env MOOSE_DASHBOARD_COOKIE_SECRET "$(random)"
fi

echo "Keycloak: http://localhost:$PORT (logins in data/dev-keycloak.txt). Rerun ./scripts/deploy.sh to apply."
