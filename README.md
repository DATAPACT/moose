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
pip install -e ".[api,dev]"

export MOOSE_API_KEY=test-dev-key
export MOOSE_SECRET_KEY=<random secret>
uvicorn moose_api.main:app --host 0.0.0.0 --port 8000
```

## Shared models

Shared models are the models every user may use, on the operator's own provider
key and at the operator's cost. Users can also bring their own keys (see
[Partners](#partners)); those are separate.

1. **Open the Account page** of the dashboard and log in with an administrator
   account (a Keycloak account with the `moose-admin` role, see
   [Account login](#account-login-keycloak)). The page shows the *Shared models*
   section only to administrators.
2. **Add the provider key**, for example your OpenRouter key. Moose stores it
   encrypted with `MOOSE_SECRET_KEY` and only ever shows it masked. Set a credit
   limit on that key in OpenRouter's dashboard to cap Moose's spending.
3. **Choose the models**: search OpenRouter's catalogue (with prices), pick the
   models everyone may use, and choose the default. Changes apply immediately.

The same is available through the API (`GET /admin/shared-models`,
`PUT /admin/shared-models/key`, `PUT /admin/shared-models`) with an
administrator's login token as `Authorization: Bearer <token>`. Requests pick a
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

Moose is also a Python library, `moose-annotate` on PyPI, imported as `moose`. It runs on
your machine: LLM calls go straight to your provider with your key, and Laya runs locally.

```bash
pip install "moose-annotate[laya]"
```

```python
from moose import Annotator, OpenRouterClient

moose = Annotator(OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-oss-120b"))
moose.annotate_text("Thanks, Luca Bianchi. Your customer ID is USR 661 204.")
```

See [docs/library.md](docs/library.md) for tables, column relationships, privacy analysis,
your own schemas and async use.

## Auth Headers

All API calls require:
- `X-API-Key: <your-key>`

Two kinds of API keys exist, and neither grants administration:

- `MOOSE_API_KEY` uses the shared models. It may be given to external users.
  Everyone holding it shares one identity, so jobs submitted with it can be read
  by anyone else with the key who knows the job ID (IDs are random and not
  listable). Users who need private jobs or their own provider keys use a vault
  token.
- A vault API token belongs to one user (see [Partners](#partners)).

Administration (shared models, invitations, schema changes) needs an
administrator's login: a Keycloak account with the `moose-admin` role, sent as
`Authorization: Bearer <access token>`. The dashboard's Account and BYO Schema
pages do this after you log in.

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
     -H "Authorization: Bearer <administrator login token>" -H "Content-Type: application/json" \
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

1. Create the realm from [`deploy/keycloak/moose-realm-zooverse.json`](deploy/keycloak/moose-realm-zooverse.json)
   (*Create realm → Resource file*). It holds two clients: `moose-dashboard` for
   the deployed dashboard (`https://moose-ui.zooverse.dev`) and
   `moose-dashboard-dev` for a developer's local dashboard
   (`http://localhost:8502`); adjust the URLs for another host. Copy each
   client's secret from its *Credentials* tab.
2. Create users and give them the realm role `moose-user`; give administrators
   the role `moose-admin` (create it if the realm has none).
3. In the env file set `MOOSE_OIDC_ISSUER` (for example
   `https://keycloak.example.org/realms/moose`), `MOOSE_DASHBOARD_URL`,
   `MOOSE_DASHBOARD_OIDC_METADATA_URL` (the issuer plus
   `/.well-known/openid-configuration`), `MOOSE_DASHBOARD_OIDC_CLIENT_ID`,
   `MOOSE_DASHBOARD_OIDC_CLIENT_SECRET` and a random
   `MOOSE_DASHBOARD_COOKIE_SECRET`, then rerun `./scripts/deploy.sh`.

To develop against the real Keycloak, set `MOOSE_DASHBOARD_OIDC_CLIENT_ID=moose-dashboard-dev`
with that client's secret, and `MOOSE_OIDC_CLIENT_IDS=moose-dashboard,moose-dashboard-dev`.
### Login off for local development

Set `MOOSE_AUTH_ENABLED=false` to run without any login: every dashboard
visitor is a "Local developer" with both the `moose-user` and `moose-admin`
roles, so the Account page shows the vault and the shared models at once, and
admin API calls need no token. The dashboard shows a red banner while it is
off. Moose refuses to start this way unless `MOOSE_BIND_ADDRESS` is
`127.0.0.1`, `MOOSE_DASHBOARD_URL` is a `http://localhost` address and
`MOOSE_PROXY_NETWORK` is the default `moose-proxy` (a server uses its real
proxy's network, such as `cloudflare_default`), so it cannot be switched off on
a server by mistake. Set it back to `true` (the
default) to use Keycloak; deployment never starts a Keycloak of its own.

- **Jobs are encrypted.** A user's job input (text, table rows), its result and
  its error details are stored encrypted with their vault key; only the user's
  own tokens can read them back through `GET /jobs/{id}`. Job status,
  timestamps, model, tokens and cost stay readable for operations. Jobs made
  with `MOOSE_API_KEY` or by administrators are not encrypted.
- **During a job** the user's vault key, and their provider key for jobs on
  their own key, are held in memory only, in the process that accepted the job.
  If Moose restarts first, the job fails with a request to submit it again. The
  API therefore always runs as one process (`--workers 1` in `docker-compose.yml`
  and the `Dockerfile`): a second one would fail the first one's running jobs.
- **Isolation.** Users see only their own jobs and keys; administrators cannot
  read users' jobs (not even in the database), use their keys, or list their
  vault. Users cannot change schemas or create invitations.
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

#### Privacy annotation with DPV: two steps, decided by a System 1 model

For personal data, prefer the `dpv_privacy` schema over `dpv`, for text and for tables. It
holds 264 DPV concepts: personal-data categories, legal roles (controller, processor,
recipient, third party) and data-subject types. Every decision is a classification, and a
small "System 1" model, [Laya](https://huggingface.co/convaiinnovations/laya), makes it,
with probabilities. The LLM adds what it knows:

1. **Find and describe.** The LLM finds the mentions in a text, or the personal-data
   columns of a table, and describes each in a few words. The prompt has no type list.
2. **Retrieve candidates.** Moose matches each description against concept labels and
   definitions (BM25, no extra model). It widens the best hits with their direct parents
   and children.
3. **Decide.** Laya scores every candidate in one forward pass. Where its top probability
   is below 0.6, Moose asks the LLM, adds the LLM's choice to Laya's input, and Laya
   decides again. The reported `confidence` is always Laya's probability.

A request can choose who decides with `"linker"`:

| `linker` | Who decides | LLM calls |
|---|---|---|
| `system1+llm` (default) | Laya, consulting the LLM when unsure | 1, plus 1 when any item is unsure |
| `system1` | Laya alone | 1 |
| `llm` (default for `dpv`) | The LLM; Laya answers yes/no on its choice | 2 |

**Probabilities always come from the System 1 model, never from the LLM.** LLMs are
over-confident: on DPVCallBench, the LLM's stated confidence averaged 0.97 when it was
right 77% of the time. So the LLM is never asked for a number. Laya gives every probability:
- for its own decisions (`system1`, `system1+llm`): the chosen type's probability, exactly as
  it appears in `ranking`;
- for the LLM's (`llm`, and one-step schemas such as `sti` and `schemaorg_cta_v1`): Laya's
  P(yes) to a plain question, "Is the mention <type>?", with that type's definition and
  nothing about the LLM. `ranking` then shows Laya's own choice among the candidates, so
  the two can differ: a high P(yes) with the LLM's type low in `ranking` means Laya accepts
  the type but would rank another higher.

Nothing remaps these probabilities. A result says `"calibrated": true` only where the schema's
calibration file records that this linker's probabilities matched accuracy on labelled
data. Without Laya (the library without `moose[laya]`, or an image built
with `MOOSE_EXTRAS=""`), results have no probability (`"confidence": null`).

Each entity or column reports:
- `confidence`: the probability;
- `calibrated`;
- `decided_by`;
- `llm_suggestion`, when the LLM was consulted;
- `category`: the LLM's description;
- `candidates`;
- `ranking`: Laya's own top 3, plus the chosen type when the LLM chose one ranked lower.

Calibration was checked on DPVCallBench (train, dev and test pooled). Values are ECE, where
lower is better and ✓ means validated (0.10 or less):

| Schema | Default | `system1` | `system1+llm` | `llm` |
|---|---|---|---|---|
| `dpv_privacy` | system1+llm, consulting below 0.6 | 0.075 ✓ | 0.052 ✓ | 0.133 |
| `dpv_pd` | Laya alone (never consults) | 0.067 ✓ | 0.067 ✓ | 0.219 |
| `dpv` | llm | 0.254 | 0.238 | 0.176 |

The yes/no question replaced scoring the LLM's choice with the LLM's suggestion in Laya's
input, which was under-confident and barely told right from wrong (ECE 0.21–0.40). The
yes/no P(yes) separates them far better but runs high (about 0.9 on average), so `llm`
results are still reported as not calibrated.

Laya is zero-shot throughout. It chooses badly among `dpv`'s 8,000+ concepts (55–60%
right, against 78% for the LLM), so `dpv` lets the LLM decide, always in two steps. That
beats one-step `dpv` (span + concept F1 0.50 against 0.37) at about 1/11 of the cost.
Against one-step `dpv`, two-step costs about 1/20 as much on DPVCallBench dev (about 1,100
prompt tokens per text instead of 98,000).

Tables, `coarse` and `fine` have no labelled data yet: Laya still gives every probability,
marked `"calibrated": false` until a labelled set exists. Column relationships (CPA) and
cell-level NER work the same way as other LLM decisions: the LLM gives one relation or type,
and Laya's yes/no answer on it is the `confidence` ("Is the relationship between the subject
column and the target column <relation>?", "Is the mention <type>?"). In privacy reports,
items without a probability (no Laya on the server) are kept and flagged low-confidence.

The API image includes Laya by default (`MOOSE_EXTRAS=laya`, CPU PyTorch, about 1 GB). Its
model, about 800 MB, downloads into a Docker volume on first use, and runs on a GPU when one
is available. A schema uses another checkpoint through its `linker_calibration` file.

Any schema can use this mode through its registry entry:
- `"candidate_mode": "two_step"`;
- `"candidate_k"`: retrieved candidates per item, 8 by default;
- `"mention_guidance"`: what to look for, in plain words;
- `"linker"`;
- `"linker_calibration"`: the Laya checkpoint, the consult threshold, and per linker the
  tasks its probabilities were validated for.

It works best when the type source gives each type a label, a definition and optionally
its `parents`. `scripts/build_dpv_privacy_subset.py` rebuilds the DPV subset, and
`research/dpv_eval/fit_system1.py` refits the threshold, `verify_choice.py` caches Laya's yes/no
answers, and `check_calibration.py` rechecks each linker's calibration.

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

### 7) Anonymisation

Masks what could identify a person: names and ID numbers (DIRECT), and the dates, places,
employers and other details that could identify them together (QUASI). The result has the
mentions with their decision and `masked_text`, with one placeholder per entity.

```bash
curl -sS -X POST "http://localhost:8000/anonymise" \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text":"On 3 May 2021 Ms Anna Berg, a nurse at St Olav'"'"'s Hospital in Trondheim, complained to the Norwegian Data Protection Authority.",
    "protect":"Ms Berg"
  }'
```

`protect` says whom to protect; without it, every private individual in the text.

### 8) Poll a job

```bash
curl -sS "http://localhost:8000/jobs/<job_id>" \
  -H "X-API-Key: test-dev-key"
```

### 9) Get machine-readable privacy report from a completed privacy job

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
