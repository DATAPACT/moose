#!/bin/sh
# Start or update Moose, locally or in production. Extra arguments go to `docker compose up`.
set -eu

cd "$(dirname "$0")/.."

ENV_FILE="${MOOSE_ENV_FILE:-.env}"

compose() {
    docker compose --env-file "$ENV_FILE" "$@"
}

# Validate before creating bind mounts or replacing running containers.
compose config --quiet

proxy_network="$(sed -n 's/^MOOSE_PROXY_NETWORK=//p' "$ENV_FILE" | tail -n 1)"
proxy_network="${proxy_network:-moose-proxy}"
if ! docker network inspect "$proxy_network" >/dev/null 2>&1; then
    docker network create "$proxy_network" >/dev/null
    echo "Created Docker network $proxy_network."
fi

mkdir -p data/moose-user
if [ -z "$(sed -n 's/^MOOSE_MONGO_VOLUME=//p' "$ENV_FILE" | tail -n 1)" ]; then
    mkdir -p data/mongo
fi
if [ ! -e data/user_vocabularies.json ]; then
    printf '[]\n' > data/user_vocabularies.json
fi
if [ ! -f data/user_vocabularies.json ]; then
    echo 'data/user_vocabularies.json must be a JSON file, not a directory.' >&2
    exit 1
fi

compose up -d --build --wait --wait-timeout 300 "$@"
