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
# Set MOOSE_API_KEY and the LITELLM_* secrets in .env before starting.
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
- LLM gateway admin UI (LiteLLM): `http://localhost:4000/ui`

Then connect an LLM provider as described in [LLM gateway](#llm-gateway).

### Production notes

- **Public access:** go through Cloudflare only, as described in
  [Cloudflare deployment](#cloudflare-deployment). Keep `MOOSE_BIND_ADDRESS` and
  `LITELLM_BIND_ADDRESS` at `127.0.0.1` so nothing bypasses it.
- **Existing VM deployment:** the Compose project name is fixed to `moose` and
  MongoDB keeps using `./data/mongo`, so updates reuse the existing data.
  Preserve `data/` when updating the checkout.
- **Env file:** an existing `.env.moose` can be used with
  `MOOSE_ENV_FILE=.env.moose ./scripts/deploy.sh`. Set
  `MOOSE_PROXY_NETWORK=cloudflare_default` there to reuse the Cloudflare
  network; the tunnel itself stays managed separately on the VM.
- Users enter only their Moose API key in the Streamlit sidebar and can pick
  one of the models the administrator made available; LLM provider keys stay
  in the gateway.

### Option B: Local API only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export MOOSE_API_KEY=test-dev-key
export MOOSE_LLM_BASE_URL=http://localhost:4000/v1   # LiteLLM gateway
export MOOSE_LLM_API_KEY=<gateway-virtual-key>
export MOOSE_MODEL=openai/gpt-oss-120b               # default model (gateway name)
uvicorn moose_api.main:app --host 0.0.0.0 --port 8000
```

## LLM gateway

LLM-backed operations go through a [LiteLLM](https://docs.litellm.ai/) gateway,
which runs as the `litellm` service (with its own Postgres) in Docker Compose.
Provider keys never pass through Moose requests and are never stored in Moose's
database.

1. **Log in to the admin UI** at `/ui` (locally `http://localhost:4000/ui`; in
   production the Cloudflare hostname) with `LITELLM_UI_USERNAME` /
   `LITELLM_UI_PASSWORD` from the env file. LiteLLM shows a warning that this
   shared login is enabled; that is expected here, because the UI is reachable
   only on localhost or behind Cloudflare Access. Use a long random password.
2. **Add models.** Under *Models + Endpoints → Add Model*, choose *Custom
   Model Name* and enter the provider route, for example
   `openrouter/openai/gpt-oss-120b`. Set the *Public Model Name* to the model's
   own ID without the provider prefix (`openai/gpt-oss-120b`); that is the name
   Moose shows. Select the stored OpenRouter key under *Existing Credentials*.
   For a model everyone may use, set the *Model Access Group* to `shared` when
   creating it (it cannot be added afterwards; delete and re-add the model
   instead). Keys are stored encrypted with `LITELLM_SALT_KEY`; never change that
   value after the first start.
3. **Create a virtual key for Moose.** Under *Virtual Keys*, create a key with a
   budget whose *Models* is the access group `shared`. Put it in `.env` as
   `MOOSE_LLM_API_KEY`, set `MOOSE_MODEL` to the default model's public name
   (for example `openai/gpt-oss-120b`), and rerun `./scripts/deploy.sh`.

**The administrator decides which models each caller can use.** Moose's own key
(used for admin requests) and every partner team get the `shared` group, so a
new shared model appears for everyone within a minute, with no restart and no
list to edit. `GET /models` lists a caller's models, the Streamlit sidebar
offers them, and requests may pick one with `"llm": {"model": "<name>"}`;
without it, Moose uses `MOOSE_MODEL`. Any other name is rejected, and the
gateway refuses it too. Each key's budget caps its spend.

Every job response includes an `llm` block recording what actually ran:

```json
"llm": {
  "requested_model": "openai/gpt-oss-120b",
  "models_used": ["openrouter/openai/gpt-oss-120b"],
  "calls": 1, "prompt_tokens": 812, "completion_tokens": 164,
  "cost_usd": 0.00021, "fallbacks": 0
}
```

Rotating or adding a provider key is done in the gateway UI and needs no Moose
restart. With OpenRouter, provider keys you add in OpenRouter's own *BYOK*
settings are used first for the providers they cover, with OpenRouter's shared
capacity as fallback unless you disable it there; Moose and LiteLLM need no
change for that. Prompt and response logging is turned off in
[`deploy/litellm/config.yaml`](deploy/litellm/config.yaml) because Moose sends
sensitive data through the gateway. The LiteLLM image is pinned by digest;
review release notes before changing it.

## Cloudflare deployment

In production, everything public goes through the existing Cloudflare Tunnel
(`cloudflared` on `MOOSE_PROXY_NETWORK`). The tunnel only makes outbound
connections, and the containers publish ports on `127.0.0.1` only, so the VM
accepts no direct traffic for Moose.

| Hostname | Service | Who gets in |
| --- | --- | --- |
| Streamlit UI | `http://moose-demo:8501` | Cloudflare Access: allowed people only |
| Gateway admin | `http://litellm:4000` | Cloudflare Access: gateway admins only, then the LiteLLM admin login |
| API | `http://moose-api:8000` | Users with their vault API token, rate-limited by Cloudflare |

Set this up in the Cloudflare dashboard (Zero Trust):

1. **Access applications.** Under *Access → Applications*, add a self-hosted
   application for the UI hostname and one for the gateway hostname. Give each
   an *Allow* policy: for the UI, the people or email domain that may use Moose;
   for the gateway, only the admins' own email addresses, with a short session
   (for example 8 hours). Note each application's *AUD tag*.
2. **Tunnel routes.** Under *Networks → Tunnels*, open the tunnel and add a
   public hostname for each row of the table. For the UI and gateway hostnames,
   open *Additional application settings → Access*, turn on *Protect with
   Access*, and enter the team name and that application's AUD tag. `cloudflared`
   then refuses any request without a valid Access token, even if an Access
   application is later removed by mistake.
3. **API protection.** The API hostname has no Access application, so that
   partners can call it with their API key. Add a WAF *rate limiting rule* for
   that hostname (for example 60 requests per minute per IP), and under
   *SSL/TLS* use *Full* mode, *Always Use HTTPS* and minimum TLS 1.2.

Then, on the VM:

4. In the env file, keep both bind addresses at `127.0.0.1`.
5. Onboard partners with invitations as described in [Partners](#partners).

## Library use

The `moose` package can call an LLM without the API or the gateway. Keys come
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
