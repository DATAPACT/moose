#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

compose() {
    docker compose --env-file "${MOOSE_ENV_FILE:-.env}" \
        -f docker-compose.yml -f docker-compose.prod.yml "$@"
}

# Validate before creating bind mounts or replacing running containers.
compose config --quiet
mkdir -p data/moose-user
if [ ! -e data/user_vocabularies.json ]; then
    printf '[]\n' > data/user_vocabularies.json
fi
if [ ! -f data/user_vocabularies.json ]; then
    echo 'data/user_vocabularies.json must be a JSON file, not a directory.' >&2
    exit 1
fi

compose up -d --build --wait --wait-timeout 180 "$@"
