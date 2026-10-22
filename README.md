<p align="center">
  <img src="./docs/images/SINTEF_logo-JPG-blue-RGB.jpg" alt="SINTEF logo" width="220">
</p>

<h1 align="center">Moose</h1>

<p align="center">
  <img src="./src/moose_api/static/moose-readme.png" alt="Moose demo" width="900">
</p>

Moose is an async API + Streamlit demo for:
- text annotation (NER),
- table column typing,
- BYO schema ingestion,
- privacy analysis with human/machine reports.

Use this README as a quick usage guide. For deep technical details, see the docs linked at the end.

## Quick Start

### Option A: Docker (recommended)

The same `docker-compose.yml` is used locally and in production.

```bash
cp .env.example .env
# Set MOOSE_API_KEY and MOOSE_SECRET_KEY in .env before starting.
./scripts/deploy.sh
```

The script validates the configuration, creates the proxy network
(`MOOSE_PROXY_NETWORK`, default `moose-proxy`) if it is missing, prepares the
data folders, then builds and starts all services and waits until they are
healthy. Rerun it after pulling updates. Check the services with
`docker compose ps`.

Default endpoints, all bound to loopback:
- API: `http://localhost:8000`
- Frontend (Streamlit): `http://localhost:8501`

Then set up the shared models as described in [Shared models](#shared-models).

### Production notes

- **Public access:** go through Cloudflare only, as described in
  [Cloudflare deployment](#cloudflare-deployment). Keep `MOOSE_BIND_ADDRESS` at
  `127.0.0.1` so nothing bypasses it.
- **Existing VM deployment:** the Compose project name is fixed to `moose` and
  MongoDB keeps using `./data/mongo`, so updates reuse the existing data.
  Preserve `data/` when updating the checkout.
- **Env file:** an existing `.env.moose` can be used with
  `MOOSE_ENV_FILE=.env.moose ./scripts/deploy.sh`. Set
  `MOOSE_PROXY_NETWORK=cloudflare_default` there to reuse the Cloudflare
  network; the tunnel itself stays managed separately on the VM.
- Users enter only their Moose API token in the Streamlit sidebar and pick a
  shared model or one of their own keys.

### Option B: Local API only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export MOOSE_API_KEY=test-dev-key
export MOOSE_SECRET_KEY=<random secret>
uvicorn moose_api.main:app --host 0.0.0.0 --port 8000
```

## Shared models

Shared models are the models every user may use, on the operator's own provider
key and at the operator's cost. Users can also bring their own keys (see
[Partners](#partners)); those are separate.

1. **Open the Admin page** of the dashboard with the admin key (`MOOSE_API_KEY`)
   in the sidebar.
2. **Add the provider key**, for example your OpenRouter key. Moose stores it
   encrypted with `MOOSE_SECRET_KEY` and only ever shows it masked. Set a credit
   limit on that key in OpenRouter's dashboard to cap Moose's spending.
3. **Choose the models**: search OpenRouter's catalogue (with prices), pick the
   models everyone may use, and choose the default. Changes apply immediately.

The same is available through the API: `GET /admin/shared-models`,
`PUT /admin/shared-models/key` and `PUT /admin/shared-models`. Requests pick a
shared model with `"llm": {"model": "<id>"}` (`GET /models` lists them); without
it, Moose uses the default. Any other model is rejected.

Every job response includes an `llm` block recording what actually ran:

```json
"llm": {
  "requested_model": "openai/gpt-oss-120b",
  "models_used": ["openai/gpt-oss-120b"],
  "calls": 1, "prompt_tokens": 812, "completion_tokens": 164,
  "cost_usd": 0.00021, "fallbacks": 0
}
```

With OpenRouter, provider keys you add in OpenRouter's own *BYOK* settings are
used first for the providers they cover, with OpenRouter's shared capacity as
fallback unless you disable it there; Moose needs no change for that.

## Cloudflare deployment

In production, everything public goes through the existing Cloudflare Tunnel
(`cloudflared` on `MOOSE_PROXY_NETWORK`). The tunnel only makes outbound
connections, and the containers publish ports on `127.0.0.1` only, so the VM
accepts no direct traffic for Moose.

| Hostname | Service | Who gets in |
| --- | --- | --- |
| Streamlit UI | `http://moose-demo:8501` | Cloudflare Access: allowed people only |
| API | `http://moose-api:8000` | Users with their vault API token, rate-limited by Cloudflare |

Set this up in the Cloudflare dashboard (Zero Trust):

1. **Access application.** Under *Access → Applications*, add a self-hosted
   application for the UI hostname with an *Allow* policy for the people or
   email domain that may use Moose. Note its *AUD tag*.
2. **Tunnel routes.** Under *Networks → Tunnels*, open the tunnel and add a
   public hostname for each row of the table. For the UI hostname, open
   *Additional application settings → Access*, turn on *Protect with Access*,
   and enter the team name and the application's AUD tag. `cloudflared`
   then refuses any request without a valid Access token, even if an Access
   application is later removed by mistake.
3. **API protection.** The API hostname has no Access application, so that
   partners can call it with their API key. Add a WAF *rate limiting rule* for
   that hostname (for example 60 requests per minute per IP), and under
   *SSL/TLS* use *Full* mode, *Always Use HTTPS* and minimum TLS 1.2.

Then, on the VM:

4. In the env file, keep `MOOSE_BIND_ADDRESS` at `127.0.0.1`.
5. Onboard partners with invitations as described in [Partners](#partners).

## Library use

The `moose` package can call an LLM without the API. Keys come
from your own code, an environment variable, or a callable that reads a keychain
or secrets manager on every request:

```python
import keyring
from moose.llm import OllamaClient, OpenRouterClient

llm = OllamaClient(host="http://localhost:11434", token=None, model="qwen3:8b", timeout=120)
llm = OpenRouterClient(api_key=lambda: keyring.get_password("moose", "openrouter"),
                       model="anthropic/claude-sonnet-4.5", timeout=120)
```

## Auth Headers

All API calls require:
- `X-API-Key: <your-key>`

`MOOSE_API_KEY` is the admin key. External users use their own vault API
token (see [Partners](#partners)).

## Partners

External users bring their own LLM provider keys to the hosted Moose. Each user
has a personal *vault*: their provider keys are encrypted with a data key that
only the user's API tokens or recovery code can unlock. Moose stores token
hashes, wrapped keys and ciphertext, so the operator cannot read users' keys or
run jobs on them without the user. The design and its limits are described in
[docs/design/partner-keys.md](docs/design/partner-keys.md).

1. **Invite.** The admin creates a one-time invitation and sends the code to
   the user:

   ```bash
   curl -sS -X POST http://localhost:8000/admin/invites \
     -H "X-API-Key: <admin-key>" -H "Content-Type: application/json" \
     -d '{"note": "Partner A"}'
   ```

2. **Register.** The user redeems it and receives their first API token and a
   recovery code, each shown once:

   ```bash
   curl -sS -X POST http://localhost:8000/vault/register \
     -H "Content-Type: application/json" \
     -d '{"invite_code": "<invite-code>", "name": "Partner A"}'
   ```

3. **Add a provider key.** Supported providers: `GET /vault/providers`
   (OpenRouter, OpenAI). Only a masked form is ever returned:

   ```bash
   curl -sS -X POST http://localhost:8000/vault/keys \
     -H "X-API-Key: <user-token>" -H "Content-Type: application/json" \
     -d '{"provider": "openrouter", "api_key": "<their key>", "label": "main",
          "default_model": "openai/gpt-oss-120b"}'
   ```

4. **Run jobs on it** with `"llm": {"key_id": "<key_id>"}` and optionally a
   `model`. Without `key_id`, users get the shared models (`GET /models`).

More: `GET /vault/me`, `POST /vault/tokens` (one token per script or machine),
`DELETE /vault/tokens/{id}`, `DELETE /vault/keys/{id}`, and `POST /vault/recover`
with the recovery code. Losing every token and the recovery code makes the
vault unreadable, by design.

### Account login (Keycloak)

With account login configured, users log in to the dashboard with their
Keycloak account instead of using an invitation. The **Account** page creates
their vault on first login (showing the first API token and recovery code
once), unlocks it with a token or the recovery code, and manages provider keys
and tokens. For OpenRouter keys it offers OpenRouter's model catalogue with
prices. Moose verifies each login token itself (signature, issuer, expiry,
client and the `moose-user` role) and links one vault to each account.

Production, with an existing Keycloak:

1. Create the realm from [`deploy/keycloak/moose-realm.json`](deploy/keycloak/moose-realm.json),
   changing the client's redirect URI and web origin to the dashboard's public
   URL (for example `https://moose-ui.example.org/oauth2callback`), and set a
   client secret.
2. Create users and give them the realm role `moose-user`.
3. In the env file set `MOOSE_OIDC_ISSUER` (for example
   `https://keycloak.example.org/realms/moose`), `MOOSE_DASHBOARD_URL`,
   `MOOSE_DASHBOARD_OIDC_METADATA_URL` (the issuer plus
   `/.well-known/openid-configuration`), `MOOSE_DASHBOARD_OIDC_CLIENT_ID`,
   `MOOSE_DASHBOARD_OIDC_CLIENT_SECRET` and a random
   `MOOSE_DASHBOARD_COOKIE_SECRET`, then rerun `./scripts/deploy.sh`.

Locally, `./scripts/dev_keycloak.sh` starts a stand-in Keycloak on
`http://localhost:8180` with this realm and a test user, and writes the
settings to `.env` (the test login goes to `data/dev-keycloak.txt`). It is
for development only: deployment never starts a Keycloak (`docker-compose.yml`
has none), and the script refuses to run when the env file already points to
another Keycloak.

- **During a job** the user's provider key is decrypted in memory only, in the
  process that accepted the job. If Moose restarts first, the job fails with a
  request to submit it again. Run the API as one process
  (`MOOSE_WEB_CONCURRENCY=1`).
- **Isolation.** Users see only their own jobs and keys; the admin cannot read
  users' jobs through the API, use their keys, or list their vault. Users
  cannot change schemas or create invitations.
- **Limit.** A modified, malicious server could capture keys while a job runs.
  Users reduce that risk with provider keys made only for Moose, with a small
  budget, the models they need, and an expiry.

## Core Usage (curl)

### 1) List available schemas

```bash
curl -sS "http://localhost:8000/schemas?include_type_count=true" \
  -H "X-API-Key: test-dev-key"
```

### 2) Text annotation (schema-specific)

```bash
curl -sS -X POST "http://localhost:8000/schemas/coarse/ner" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Maria Rossi works at Acme in Milan."
  }'
```

### 3) Table annotation (schema-specific)

```bash
curl -sS -X POST "http://localhost:8000/schemas/sti/tabular/annotate" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "table_id":"patients",
    "sampled_rows":[
      {"patient_name":"Alice Smith","email":"alice@example.com","medical_record_number":"MRN-100245"},
      {"patient_name":"Bob Jones","email":"bob@example.com","medical_record_number":"MRN-100246"}
    ]
  }'
```

### 4) BYO schema preview (dry run)

```bash
curl -sS -X POST "http://localhost:8000/schemas/ingest/preview" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"healthcare_pii",
    "label":"Healthcare PII",
    "source_type":"file",
    "source_path":"examples/schema_ingest_samples/schemas/healthcare_pii.json",
    "supports_text":true,
    "supports_table":true,
    "score_mode":"sparse"
  }'
```

### 5) BYO schema ingest (activate)

```bash
curl -sS -X POST "http://localhost:8000/schemas/ingest" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"healthcare_pii",
    "label":"Healthcare PII",
    "source_type":"file",
    "source_path":"examples/schema_ingest_samples/schemas/healthcare_pii.json",
    "supports_text":true,
    "supports_table":true,
    "score_mode":"sparse"
  }'
```

### 6) Privacy analysis

```bash
curl -sS -X POST "http://localhost:8000/privacy/analyze" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "profile":"balanced",
    "tasks":[
      {
        "kind":"text",
        "task_id":"t1",
        "text":"We collect patient names and emails and share with billing processor."
      }
    ]
  }'
```

### 7) Poll a job

```bash
curl -sS "http://localhost:8000/jobs/<job_id>" \
  -H "X-API-Key: test-dev-key"
```

### 8) Get machine-readable privacy report from a completed privacy job

```bash
curl -sS "http://localhost:8000/jobs/<job_id>/privacy-report" \
  -H "X-API-Key: test-dev-key"
```

## Frontend (Streamlit)

If running with Docker Compose, open:
- `http://localhost:8501`

The demo UI supports:
- text annotation,
- table annotation,
- BYO schema ingestion,
- privacy analysis,
- report downloads (human-readable `.md` and machine-readable `.json`).

## Detailed Documentation

- System blueprint: [docs/BLUEPRINT.md](docs/BLUEPRINT.md)
- Developer deep dive: [docs/DEVELOPER_DEEP_DIVE.md](docs/DEVELOPER_DEEP_DIVE.md)
- Privacy operation details: [docs/PRIVACY_OPERATION.md](docs/PRIVACY_OPERATION.md)
- BYO schema design + guardrails: [docs/BYO_SCHEMA_DESIGN.md](docs/BYO_SCHEMA_DESIGN.md)
