from __future__ import annotations

import hmac
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import replace
from urllib.parse import urlsplit
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from moose.config import Settings, get_settings
from moose.ingest import delete_user_schema, preview_schema_payload, update_user_schema_metadata
from moose.pipelines import load_privacy_profiles
from moose.linkers import LINKERS, SYSTEM1_LINKERS, Calibration, system1_ready
from moose.ner import resolve_linker, two_step_active
from moose.schema import get_schema_config, list_schema_names, reload_schema_registry
from moose.privacy import get_machine_report_json_schema, list_policy_packs
from moose.vault import VaultError
from moose_api.job_sealing import is_sealed, open_value, seal_payload
from moose_api.oidc import OIDCError, OIDCVerifier
from moose_api.queue import JobRecord, WorkerPool, build_backends, utc_now
from moose_api.shared_models import SharedModelsError, SharedModelsStore
from moose_api.vault_store import PROVIDERS, ProviderRoute, Unlocked, VaultStore


class LLMOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None,
        description=(
            "Model to use. Without key_id: one of the shared models (see GET /models), defaulting "
            "to the server's default. With key_id: any model of that provider, defaulting to the "
            "key's default model."
        ),
    )
    key_id: str | None = Field(
        default=None,
        description="Run on one of your own provider keys (see GET /vault/keys) instead of a shared model.",
    )


# -----------------------
# Text NER request models
# -----------------------
LINKER_FIELD = Field(
    default=None,
    description=(
        "Two-step schemas only: who chooses each item's type (a mention, a column) among its candidates. "
        "'system1+llm': the System 1 model (Laya) decides and consults the LLM when unsure; "
        "'system1': Laya alone, no extra LLM call; 'llm': the LLM decides. With Laya, every "
        "confidence is Laya's probability. Defaults to system1+llm when Laya is installed."
    ),
)
Linker = Literal[LINKERS]


class BaseNERRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMOptions = Field(default_factory=LLMOptions)

    text: str = Field(min_length=1)
    linker: Linker | None = LINKER_FIELD

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty.")
        return value


class NERRequest(BaseNERRequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        description="Schema/vocabulary name to annotate against.",
        examples=["coarse", "fine", "dpv", "dpv_pd"],
    )

    @field_validator("schema_name")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# Tabular typing request models
# -----------------------------
class BaseTabularRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMOptions = Field(default_factory=LLMOptions)

    table_id: str | None = None
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)
    linker: Linker | None = LINKER_FIELD


class TabularRequest(BaseTabularRequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        description="Schema/vocabulary name to annotate against.",
        examples=["dpv", "dpv_pd", "sti", "schemaorg_cta_v1"],
    )

    @field_validator("schema_name")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# CPA (column relationship) request models
# -----------------------------
class BaseCPARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMOptions = Field(default_factory=LLMOptions)

    table_id: str | None = None
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)
    subject_column: str = Field(min_length=1, description="Subject column name (required).")

    # OPTIONAL: if known, helps filter schema.org predicates using domainIncludes + class hierarchy.
    # Example: "schema:Book"
    subject_class: str | None = Field(
        default=None,
        description="Optional schema.org class CURIE for the subject column (e.g., schema:Book). "
                    "If omitted and the CPA schema is schema.org-based, Moose will infer it using schemaorg_cta_v1 by default.",
    )

    target_columns: list[str] | None = None

    # Optional performance knob for deep CPA:
    # If enabled, Moose runs STI typing once and caches selection by STI type signature across columns.
    use_sti_signature_cache: bool = True

    # OPTIONAL: debug output (per task)
    debug: bool = False
    debug_preview_limit: int = Field(default=20, ge=0, le=200)


    @field_validator("subject_column")
    @classmethod
    def validate_subject_column(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("subject_column must be non-empty.")
        return value


class CPARequest(BaseCPARequest):
    schema_name: str = Field(
        alias="schema",
        serialization_alias="schema",
        default="cpa",
        description="CPA relationship schema name (must support CPA).",
        examples=["cpa", "schemaorg_cpa_v1"],
    )

    @field_validator("schema_name")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        try:
            get_schema_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


# -----------------------------
# Privacy analysis request models
# -----------------------------
class PrivacyTextTaskIn(BaseModel):
    kind: Literal["text"] = "text"
    task_id: str
    text: str
    context: dict[str, Any] | None = None


class PrivacyTableTaskIn(BaseModel):
    kind: Literal["table"] = "table"
    task_id: str
    table_id: str
    sampled_rows: list[dict[str, Any]] = Field(min_length=1)
    scan_columns: list[str] | None = None
    context: dict[str, Any] | None = None


PrivacyTaskIn = Annotated[Union[PrivacyTextTaskIn, PrivacyTableTaskIn], Field(discriminator="kind")]


class PrivacyAnalyzeRequest(BaseModel):
    # NEW: profile support (fast/balanced/deep)
    profile: str | None = None

    # These are now OPTIONAL; if omitted, profile defaults will apply in moose/privacy.py.
    policy_pack: str | None = None
    analysis_mode: Literal["rules", "hybrid"] | None = None
    text_schema: str | None = None
    table_schema: str | None = None
    scan_schema: str | None = None
    include_extraction: bool | None = None

    tasks: list[PrivacyTaskIn] = Field(min_length=1)
    llm: LLMOptions = Field(default_factory=LLMOptions)


# -----------------------
# Schema-specific wrappers
# -----------------------
class SchemaNERRequest(BaseNERRequest):
    pass


class SchemaTabularRequest(BaseTabularRequest):
    pass


class SchemaCPARequest(BaseCPARequest):
    pass


class DpvNERRequest(BaseNERRequest):
    pass


class DpvTabularRequest(BaseTabularRequest):
    pass


class SchemaIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMOptions = Field(default_factory=LLMOptions)

    name: str = Field(min_length=1, description="Schema registry name.")
    label: str | None = None
    description: str | None = None

    source_type: Literal["url", "github", "file"] = "url"
    source_url: str | None = None
    source_path: str | None = None
    github_repo: str | None = None
    github_ref: str | None = "main"
    github_path: str | None = None

    use_llm_fallback: bool = False

    score_mode: Literal["dense", "sparse"] = "sparse"
    prefilter_types: bool = False
    supports_text: bool = True
    supports_table: bool = True
    supports_cpa: bool = False
    text_intro: str | None = None
    table_intro: str | None = None
    cpa_intro: str | None = None
    expected_source_sha256: str | None = None
    expected_mapping_sha256: str | None = None


class SchemaPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    description: str | None = None
    score_mode: Literal["dense", "sparse"] | None = None
    prefilter_types: bool | None = None
    supports_text: bool | None = None
    supports_table: bool | None = None
    supports_cpa: bool | None = None
    text_intro: str | None = None
    table_intro: str | None = None
    cpa_intro: str | None = None


# -------------
# Job responses
# -------------
class JobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class SchemaIngestPreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(alias="schema", serialization_alias="schema")
    label: str
    description: str
    type_count: int
    sample_type_ids: list[str]
    source: dict[str, Any]
    source_sha256: str
    mapping_sha256: str
    mapping_strategy: Literal["deterministic", "llm_fallback"]
    mapping_confidence: float | None = None
    mapping: dict[str, Any]
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    can_activate: bool = True
    activated: bool = False


STATIC_DIR = Path(__file__).resolve().parent / "static"

TAG_METADATA = [
    {"name": "NER", "description": "Named entity recognition endpoints."},
    {"name": "Tabular", "description": "Tabular semantic typing endpoints."},
    {"name": "CPA", "description": "Column Property Annotation (CPA) / column relationship prediction."},
    {"name": "Schemas", "description": "Schema-specific annotation endpoints."},
    {"name": "DPV", "description": "DPV classification endpoints."},
    {"name": "Privacy", "description": "Privacy analysis orchestration endpoint."},
    {"name": "Metadata", "description": "Metadata endpoints used by frontends (schemas, policy packs, profiles, assets)."},
]

app = FastAPI(
    title="Moose API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_tags=TAG_METADATA,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# Callers of the annotation API:
# - shared key: MOOSE_API_KEY. Uses the shared models; may be given to external users, so
#   it grants no administration.
# - users: a vault API token (see docs/design/partner-keys.md). The token unlocks the user's
#   vault in memory for the request, so their own provider keys can be used; the server
#   stores only hashes, wrapped keys and ciphertext. Each caller sees only their own jobs.
# Administration (shared models, invitations, schema changes) needs a login token of an
# account with the admin role, see require_admin.
SHARED_KEY_CLIENT = "shared-key"
_NO_SHARED_MODELS = "No shared models are set up. Use one of your own keys (llm.key_id), or ask the administrator."

_current_client: ContextVar[str | None] = ContextVar("moose_api_client", default=None)
# The caller's unlocked vault (users only); never stored.
_current_vault: ContextVar[Unlocked | None] = ContextVar("moose_vault", default=None)


async def require_api_key(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str:
    shared_key = app.state.settings.MOOSE_API_KEY
    if not shared_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    token = api_key or (bearer.credentials if bearer else None) or ""
    if not token and not app.state.settings.MOOSE_AUTH_ENABLED:
        # Login disabled (local development): the caller is the local developer.
        return await _admin_client(_local_developer())
    if hmac.compare_digest(token.encode(), shared_key.encode()):
        _current_client.set(SHARED_KEY_CLIENT)
        _current_vault.set(None)
        return SHARED_KEY_CLIENT
    if not api_key and bearer is not None and token.count(".") == 2 and app.state.oidc is not None:
        # An administrator's login token (a JWT), e.g. to follow their own schema-ingest job.
        return await _admin_client(await require_admin(bearer))
    try:
        unlocked = await app.state.vault.unlock(token)
    except VaultError:
        unlocked = None
    if unlocked is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    client = f"user:{unlocked.user_id}"
    _current_client.set(client)
    _current_vault.set(unlocked)
    return client


async def require_user(client: str = Depends(require_api_key)) -> Unlocked:
    unlocked = _current_vault.get()
    if unlocked is None:
        raise HTTPException(status_code=403, detail="This operation needs a user API token.")
    return unlocked


LOCAL_DEVELOPER_SUB = "local-developer"
# Compose's default proxy network; servers set their real proxy's network instead.
LOCAL_PROXY_NETWORK = "moose-proxy"


def _local_developer() -> dict[str, Any]:
    """The identity everyone gets when login is disabled: a user and an administrator."""
    settings: Settings = app.state.settings
    return {
        "sub": LOCAL_DEVELOPER_SUB,
        "name": "Local developer",
        "email": None,
        "realm_access": {"roles": [settings.MOOSE_OIDC_USER_ROLE, settings.MOOSE_OIDC_ADMIN_ROLE]},
    }


def check_auth_settings(settings: Settings) -> None:
    """Refuse to run without login anywhere but on a developer's machine."""
    if settings.MOOSE_AUTH_ENABLED:
        return
    url = urlsplit(settings.MOOSE_DASHBOARD_URL or "")
    local_url = url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
    local_bind = settings.MOOSE_BIND_ADDRESS in ("127.0.0.1", "localhost", "::1")
    # A server publishes Moose through a shared proxy network (e.g. cloudflare_default), and
    # the proxy reaches the containers there whatever the bind address is.
    local_network = settings.MOOSE_PROXY_NETWORK == LOCAL_PROXY_NETWORK
    if not (local_url and local_bind and local_network):
        raise RuntimeError(
            "MOOSE_AUTH_ENABLED=false is only allowed locally: MOOSE_BIND_ADDRESS=127.0.0.1, "
            f"MOOSE_DASHBOARD_URL a http://localhost address and MOOSE_PROXY_NETWORK={LOCAL_PROXY_NETWORK} "
            "(not a server's proxy network). Otherwise enable login."
        )


async def _verify_login(bearer: HTTPAuthorizationCredentials | None, role: str) -> dict[str, Any]:
    """Claims of a verified account login token (Keycloak access token) with `role`."""
    if not app.state.settings.MOOSE_AUTH_ENABLED:
        return _local_developer()
    verifier: OIDCVerifier | None = app.state.oidc
    if verifier is None:
        raise HTTPException(status_code=503, detail="Account login is not configured on this server.")
    if bearer is None:
        raise HTTPException(status_code=401, detail="Missing login token.")
    try:
        return await verifier.verify(bearer.credentials, role)
    except OIDCError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


async def require_admin(bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme)) -> dict[str, Any]:
    """An administrator's login: an account with the admin role. API keys never qualify."""
    claims = await _verify_login(bearer, app.state.settings.MOOSE_OIDC_ADMIN_ROLE)
    await _admin_client(claims)
    return claims


async def _admin_client(claims: dict[str, Any]) -> str:
    client = f"admin:{claims['sub']}"
    _current_client.set(client)
    _current_vault.set(None)
    return client


def _ensure_schema_supported(
    schema: str,
    require_text: bool = False,
    require_table: bool = False,
    require_cpa: bool = False,
    linker: str | None = None,
) -> None:
    try:
        config = get_schema_config(schema)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if linker is not None and config.candidate_mode == "all":
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' annotates in one step; 'linker' applies only to two-step schemas.")
    if linker in SYSTEM1_LINKERS and not system1_ready(Calibration.load(config.linker_calibration)):
        raise HTTPException(status_code=400, detail='The System 1 model (Laya) is not available on this server; use "linker": "llm".')

    if require_text and not config.supports_text:
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support text annotation.")
    if require_table and not config.supports_table:
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support tabular annotation.")
    if require_cpa and not getattr(config, "supports_cpa", False):
        raise HTTPException(status_code=400, detail=f"Schema '{schema}' does not support CPA annotation.")


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    if not settings.MOOSE_API_KEY:
        raise RuntimeError("MOOSE_API_KEY is required for the API service")
    check_auth_settings(settings)

    job_store, queue_backend, info = await build_backends(settings)
    mongo_client = info.get("mongo_client")
    vault_store = (
        await VaultStore.mongo(mongo_client[settings.MOOSE_MONGO_DB]) if mongo_client else VaultStore.in_memory()
    )

    shared_models = (
        SharedModelsStore.mongo(mongo_client[settings.MOOSE_MONGO_DB], settings.MOOSE_SECRET_KEY)
        if mongo_client
        else SharedModelsStore.in_memory(settings.MOOSE_SECRET_KEY)
    )

    app.state.settings = settings
    app.state.vault = vault_store
    app.state.oidc = (
        OIDCVerifier(
            issuer=settings.MOOSE_OIDC_ISSUER,
            jwks_url=settings.MOOSE_OIDC_JWKS_URL,
            client_ids=[c.strip() for c in settings.MOOSE_OIDC_CLIENT_IDS.split(",") if c.strip()],
        )
        if settings.MOOSE_AUTH_ENABLED and settings.MOOSE_OIDC_ISSUER
        else None
    )
    app.state.shared_models = shared_models
    app.state.job_store = job_store
    app.state.queue_backend = queue_backend
    app.state.queue_info = info

    worker_pool = WorkerPool(queue_backend, job_store, shared_models, settings)
    await worker_pool.start()
    app.state.worker_pool = worker_pool


@app.on_event("shutdown")
async def shutdown() -> None:
    worker_pool: WorkerPool = app.state.worker_pool
    await worker_pool.stop()

    queue_info = app.state.queue_info
    mongo_client = queue_info.get("mongo_client")
    if mongo_client is not None:
        mongo_client.close()


def _new_job_id() -> str:
    # Compact UUID form for easier copy/paste while keeping enough entropy.
    return uuid.uuid4().hex


async def _shared_models() -> tuple[list[str], str | None]:
    """Shared models set up by the administrator, and the default one (one read)."""
    models, default = await app.state.shared_models.models()
    return models, default if default in models else (models[0] if models else None)


async def _resolve_model(requested: str | None) -> str:
    models, default = await _shared_models()
    model = requested or default
    if model is None:
        raise HTTPException(status_code=503, detail=_NO_SHARED_MODELS)
    if model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. Choose one of: {', '.join(models)} (see GET /models).",
        )
    return model


async def _resolve_llm(llm: LLMOptions) -> tuple[dict[str, Any], ProviderRoute | None]:
    """Payload `llm` block (no secrets) and, for a user's own key, the in-memory route."""
    if not llm.key_id:
        return {"model": await _resolve_model(llm.model)}, None
    unlocked = _current_vault.get()
    if unlocked is None:
        raise HTTPException(status_code=400, detail="key_id needs a user API token.")
    try:
        route, key = await app.state.vault.route(unlocked, llm.key_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such key in your vault (see GET /vault/keys).") from None
    model = llm.model or key.get("default_model")
    if not model:
        raise HTTPException(status_code=400, detail="Choose llm.model, or set a default model on this key.")
    return {"model": model, "key_id": llm.key_id, "provider": route.provider}, route


async def _enqueue_job(endpoint_type: str, payload: dict, llm: LLMOptions, needs_llm: bool = True):
    settings: Settings = app.state.settings
    queue_backend = app.state.queue_backend
    job_store = app.state.job_store

    route: ProviderRoute | None = None
    if needs_llm:
        payload["llm"], route = await _resolve_llm(llm)

    worker_pool = app.state.worker_pool
    queue_size = await queue_backend.size() + worker_pool.local_queue_size()
    if queue_size >= settings.MOOSE_QUEUE_MAXSIZE:
        raise HTTPException(status_code=429, detail="Queue is full, try again later")

    job_id = _new_job_id()
    now = utc_now()
    # A user's input is stored encrypted with their vault key; the job then runs in this
    # process, which holds that key in memory until the job is done.
    unlocked = _current_vault.get()
    job = JobRecord(
        job_id=job_id,
        endpoint_type=endpoint_type,
        payload=seal_payload(unlocked.data_key, job_id, payload) if unlocked else payload,
        status="queued",
        created_at=now,
        updated_at=now,
        retries=0,
        client=_current_client.get(),
    )
    await job_store.put_job(job)
    try:
        if unlocked is not None:
            worker_pool.enqueue_local(job_id, route, unlocked.data_key)
        elif route is not None:
            worker_pool.enqueue_local(job_id, route)
        else:
            await queue_backend.enqueue(job_id)
    except Exception as exc:  # noqa: BLE001
        await job_store.update_job(
            job_id,
            status="failed",
            updated_at=utc_now(),
            error=f"Failed to enqueue job: {exc}",
        )
        raise HTTPException(status_code=429, detail="Queue is full, try again later")
    return job_id


def _validate_schema_ingest_request(request: SchemaIngestRequest) -> None:
    if request.source_type == "url" and not request.source_url:
        raise HTTPException(status_code=400, detail="source_url is required when source_type='url'.")
    if request.source_type == "file" and not request.source_path:
        raise HTTPException(status_code=400, detail="source_path is required when source_type='file'.")
    if request.source_type == "github" and (not request.github_repo or not request.github_path):
        raise HTTPException(
            status_code=400,
            detail="github_repo and github_path are required when source_type='github'.",
        )


def _build_schema_ingest_payload(request: SchemaIngestRequest) -> dict[str, Any]:
    return {
        "name": request.name,
        "label": request.label,
        "description": request.description,
        "source_type": request.source_type,
        "source_url": request.source_url,
        "source_path": request.source_path,
        "github_repo": request.github_repo,
        "github_ref": request.github_ref,
        "github_path": request.github_path,
        "use_llm_fallback": request.use_llm_fallback,
        "score_mode": request.score_mode,
        "prefilter_types": request.prefilter_types,
        "supports_text": request.supports_text,
        "supports_table": request.supports_table,
        "supports_cpa": request.supports_cpa,
        "text_intro": request.text_intro,
        "table_intro": request.table_intro,
        "cpa_intro": request.cpa_intro,
        "expected_source_sha256": request.expected_source_sha256,
        "expected_mapping_sha256": request.expected_mapping_sha256,
    }


# -------------
# Text NER APIs
# -------------
@app.post("/ner", dependencies=[Depends(require_api_key)], tags=["NER"], response_model=JobQueuedResponse)
async def submit_ner(
    request: NERRequest,
):
    _ensure_schema_supported(request.schema_name, require_text=True, linker=request.linker)

    payload = {
        "schema": request.schema_name,
        "text": request.text,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("ner", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/ner", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_ner(
    schema: str,
    request: SchemaNERRequest,
):
    _ensure_schema_supported(schema, require_text=True, linker=request.linker)

    payload = {
        "schema": schema,
        "text": request.text,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("ner", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/dpv/ner", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
async def submit_dpv_ner(
    request: DpvNERRequest,
):
    _ensure_schema_supported("dpv", require_text=True, linker=request.linker)

    payload = {
        "schema": "dpv",
        "text": request.text,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("ner", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


# -----------------
# Tabular typing APIs
# -----------------
@app.post("/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["Tabular"], response_model=JobQueuedResponse)
async def submit_tabular(
    request: TabularRequest,
):
    _ensure_schema_supported(request.schema_name, require_table=True, linker=request.linker)

    payload = {
        "schema": request.schema_name,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("tabular", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular(
    schema: str,
    request: SchemaTabularRequest,
):
    _ensure_schema_supported(schema, require_table=True, linker=request.linker)

    payload = {
        "schema": schema,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("tabular", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/dpv/tabular/annotate", dependencies=[Depends(require_api_key)], tags=["DPV"], response_model=JobQueuedResponse, deprecated=True)
async def submit_dpv_tabular(
    request: DpvTabularRequest,
):
    _ensure_schema_supported("dpv", require_table=True, linker=request.linker)

    payload = {
        "schema": "dpv",
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
    }
    if request.linker:
        payload["linker"] = request.linker
    job_id = await _enqueue_job("tabular", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# CPA endpoints
# ------------------------
@app.post("/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["CPA"], response_model=JobQueuedResponse)
async def submit_tabular_cpa(
    request: CPARequest,
):
    _ensure_schema_supported(request.schema_name, require_cpa=True)

    payload = {
        "schema": request.schema_name,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "subject_column": request.subject_column,
        "subject_class": request.subject_class,
        "target_columns": request.target_columns,
        "use_sti_signature_cache": request.use_sti_signature_cache,
        "debug": request.debug,
        "debug_preview_limit": request.debug_preview_limit,
    }
    job_id = await _enqueue_job("cpa", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/{schema}/tabular/cpa", dependencies=[Depends(require_api_key)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_tabular_cpa(
    schema: str,
    request: SchemaCPARequest,
):
    _ensure_schema_supported(schema, require_cpa=True)

    payload = {
        "schema": schema,
        "table_id": request.table_id,
        "sampled_rows": request.sampled_rows,
        "subject_column": request.subject_column,
        "subject_class": request.subject_class,
        "target_columns": request.target_columns,
        "use_sti_signature_cache": request.use_sti_signature_cache,
        "debug": request.debug,
        "debug_preview_limit": request.debug_preview_limit,
    }
    job_id = await _enqueue_job("cpa", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


# ------------------------
# Privacy analysis API
# ------------------------
@app.post("/privacy/analyze", dependencies=[Depends(require_api_key)], tags=["Privacy"], response_model=JobQueuedResponse)
async def submit_privacy_analyze(
    request: PrivacyAnalyzeRequest,
):
    """
    NOTE: We do not validate text_schema/table_schema/scan_schema here anymore,
    because they can be omitted and resolved by `profile` defaults in moose/privacy.py.
    """

    payload = {
        "profile": request.profile,
        "policy_pack": request.policy_pack,
        "analysis_mode": request.analysis_mode,
        "text_schema": request.text_schema,
        "table_schema": request.table_schema,
        "scan_schema": request.scan_schema,
        "include_extraction": request.include_extraction,
        "tasks": [t.model_dump() for t in request.tasks],
    }
    job_id = await _enqueue_job("privacy_analyze", payload, request.llm)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post("/schemas/ingest", dependencies=[Depends(require_admin)], tags=["Schemas"], response_model=JobQueuedResponse)
async def submit_schema_ingest(
    request: SchemaIngestRequest,
):
    _validate_schema_ingest_request(request)

    payload = _build_schema_ingest_payload(request)
    job_id = await _enqueue_job("schema_ingest", payload, request.llm, needs_llm=request.use_llm_fallback)
    return JobQueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/schemas/ingest/preview",
    dependencies=[Depends(require_admin)],
    tags=["Schemas"],
    response_model=SchemaIngestPreviewResponse,
)
async def preview_schema_ingest(
    request: SchemaIngestRequest,
):
    _validate_schema_ingest_request(request)

    llm_client = None
    if request.use_llm_fallback:
        try:
            llm_client = await app.state.worker_pool.shared_client(await _resolve_model(request.llm.model))
        except SharedModelsError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = _build_schema_ingest_payload(request)
    try:
        preview = await preview_schema_payload(payload, llm_client=llm_client)
        return SchemaIngestPreviewResponse.model_validate(preview)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if llm_client is not None:
            await llm_client.close()


@app.patch("/schemas/{schema_name}", dependencies=[Depends(require_admin)], tags=["Schemas"])
async def patch_user_schema(schema_name: str, request: SchemaPatchRequest) -> dict[str, Any]:
    updates = request.model_dump()
    if not any(value is not None for value in updates.values()):
        raise HTTPException(status_code=400, detail="At least one field must be provided.")
    try:
        return {"status": "ok", **update_user_schema_metadata(schema_name, updates)}
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc


@app.delete("/schemas/{schema_name}", dependencies=[Depends(require_admin)], tags=["Schemas"])
async def delete_schema(schema_name: str, remove_files: bool = True) -> dict[str, Any]:
    try:
        result = delete_user_schema(schema_name, remove_files=remove_files)
        return {"status": "ok", **result}
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc


# ----------
# Metadata / jobs
# ----------
async def _get_visible_job(job_id: str) -> JobRecord:
    """The caller's own job, with a user's encrypted result and error opened in memory."""
    job = await app.state.job_store.get_job(job_id)
    client = _current_client.get()
    # Each caller sees only their own jobs. Older jobs (no client, or the former "admin")
    # were made with MOOSE_API_KEY.
    owner = job.client if job and job.client not in (None, "admin") else SHARED_KEY_CLIENT
    if not job or owner != client:
        raise HTTPException(status_code=404, detail="Job not found")
    unlocked = _current_vault.get()
    if unlocked is not None:
        try:
            if is_sealed(job.result):
                job = replace(job, result=open_value(unlocked.data_key, job.job_id, "result", job.result))
            if is_sealed(job.error):
                job = replace(job, error=open_value(unlocked.data_key, job.job_id, "error", job.error))
        except VaultError:
            raise HTTPException(status_code=500, detail="The job's stored data cannot be decrypted.") from None
    return job


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_job(job_id: str):
    job = await _get_visible_job(job_id)

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.llm:
        response["llm"] = job.llm
    if job.status == "completed":
        response["result"] = job.result
    if job.status == "failed":
        response["error"] = job.error
    return response


@app.get("/jobs/{job_id}/privacy-report", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_job_privacy_report(job_id: str) -> dict[str, Any]:
    job = await _get_visible_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed (status={job.status}).")

    result = job.result if isinstance(job.result, dict) else {}
    reports = result.get("reports")
    if not isinstance(reports, dict):
        raise HTTPException(status_code=404, detail="No reports found for this job.")

    machine = reports.get("machine_readable")
    if not isinstance(machine, dict):
        raise HTTPException(status_code=404, detail="Machine-readable report is not available for this job.")

    content = machine.get("content")
    if not isinstance(content, dict):
        raise HTTPException(status_code=404, detail="Machine-readable report content is missing or invalid.")

    return content


@app.get("/schemas", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_schemas(include_type_count: bool = False) -> dict[str, Any]:
    schemas = []
    for name in list_schema_names():
        cfg = get_schema_config(name)
        item: dict[str, Any] = {
            "name": cfg.name,
            "label": cfg.label,
            "description": cfg.description,
            "supports_text": bool(cfg.supports_text),
            "supports_table": bool(cfg.supports_table),
            "supports_cpa": bool(getattr(cfg, "supports_cpa", False)),
            "prefilter_types": bool(getattr(cfg, "prefilter_types", False)),
            "score_mode": "dense" if cfg.require_all_scores else "sparse",
            "candidate_mode": cfg.candidate_mode,
        }
        if cfg.candidate_mode != "all":
            # How this server annotates with the schema: "auto" schemas are two-step only when
            # the System 1 model (Laya) is installed. System 1 linkers need Laya and a calibration.
            calibration = Calibration.load(cfg.linker_calibration)
            item["candidate_mode"] = "two_step" if two_step_active(cfg) else "all"
            if item["candidate_mode"] == "two_step":
                item["linker"] = resolve_linker(None, calibration, cfg.linker)
                item["linkers"] = list(LINKERS) if system1_ready(calibration) else ["llm"]
                # Tasks ("text", "table") on which its probabilities were checked against labelled data.
                item["system1_validated_for"] = list(calibration.validated_for) if calibration else []
        if include_type_count:
            try:
                item["type_count"] = len(cfg.load_type_ids())
            except Exception:  # noqa: BLE001
                item["type_count"] = None
        schemas.append(item)
    return {"schemas": schemas}


@app.post("/schemas/reload", dependencies=[Depends(require_admin)], tags=["Metadata"])
async def reload_schemas() -> dict[str, Any]:
    names = reload_schema_registry()
    return {"status": "ok", "schema_count": len(names), "schemas": names}


@app.get("/policy-packs", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_policy_packs_endpoint() -> dict[str, Any]:
    return {"policy_packs": list_policy_packs()}


@app.get("/privacy/profiles", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_privacy_profiles_endpoint() -> dict[str, Any]:
    """
    Return pipelines/privacy_profiles.json so frontends can show available profiles.
    """
    return load_privacy_profiles()


@app.get("/privacy/reports/schema", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_privacy_report_schema_endpoint() -> dict[str, Any]:
    return get_machine_report_json_schema()


@app.get("/health", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def health():
    settings: Settings = app.state.settings
    queue_info = app.state.queue_info
    models, default = await app.state.shared_models.models()
    return {
        "status": "ok",
        "shared_models": len(models),
        "default_model": default,
        "worker_count": settings.MOOSE_WORKER_COUNT,
        "queue_backend": queue_info.get("queue_backend"),
    }


def _load_user_schema_assets(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "user_vocabularies.json"
    if not registry_path.exists():
        return []
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        type_source = entry.get("type_source")
        type_count: int | None = None
        if isinstance(type_source, str) and type_source.strip():
            type_path = Path(type_source.strip())
            if not type_path.is_absolute():
                type_path = data_dir / type_path
            try:
                payload = json.loads(type_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    type_count = len(payload)
            except Exception:  # noqa: BLE001
                type_count = None

        manifest_path = data_dir / "user" / name / "manifest.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except Exception:  # noqa: BLE001
                manifest = None

        item: dict[str, Any] = {
            "name": name,
            "label": entry.get("label"),
            "description": entry.get("description"),
            "type_source": entry.get("type_source"),
            "score_mode": entry.get("score_mode"),
            "supports_text": bool(entry.get("supports_text", True)),
            "supports_table": bool(entry.get("supports_table", True)),
            "supports_cpa": bool(entry.get("supports_cpa", False)),
            "prefilter_types": bool(entry.get("prefilter_types", False)),
            "type_count": type_count,
            "manifest": str(manifest_path.relative_to(data_dir).as_posix()) if manifest_path.exists() else None,
        }
        if manifest:
            item["source"] = manifest.get("source")
            item["generated_at"] = manifest.get("generated_at")
            counts = manifest.get("counts")
            if isinstance(counts, dict) and isinstance(counts.get("types"), int):
                item["type_count"] = counts["types"]
        out.append(item)

    return sorted(out, key=lambda x: x["name"])


@app.get("/models", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def list_models() -> dict[str, Any]:
    """Shared models (set up by the administrator) and, for users, their own provider keys."""
    models, default = await _shared_models()
    response: dict[str, Any] = {"default_model": default, "models": models}
    unlocked = _current_vault.get()
    if unlocked is not None:
        response["own_keys"] = await app.state.vault.list_secrets(unlocked.user_id)
    return response


# -----------------
# Vault: user accounts and their own provider keys
# -----------------
class InviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None, description="For your records, e.g. the partner's name.")
    valid_days: int = Field(default=7, ge=1, le=90)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invite_code: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)


class RecoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recovery_code: str = Field(min_length=1)
    label: str = Field(default="recovered", max_length=100)


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=100)


class ProviderKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(description="One of the supported providers (see GET /vault/providers).")
    api_key: str = Field(min_length=8, repr=False)
    label: str = Field(min_length=1, max_length=100)
    default_model: str | None = Field(default=None, description="Model used when a job names this key but no model.")


async def require_account(
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> dict[str, Any]:
    """Claims of a verified user login (an account with the user role)."""
    return await _verify_login(bearer, app.state.settings.MOOSE_OIDC_USER_ROLE)


def _account_name(claims: dict[str, Any]) -> str:
    return str(claims.get("name") or claims.get("preferred_username") or claims.get("email") or claims["sub"])


@app.get("/account", tags=["Vault"])
async def get_account(bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme)) -> dict[str, Any]:
    """The logged-in account, what its roles allow, and whether it already has a vault.
    Used by the dashboard to show only the sections an account may use; every
    protected endpoint still checks the role itself."""
    settings: Settings = app.state.settings
    claims = await _verify_login(bearer, None)
    roles = (claims.get("realm_access") or {}).get("roles") or []
    account: dict[str, Any] = {
        "name": _account_name(claims),
        "email": claims.get("email"),
        "can_use_vault": settings.MOOSE_OIDC_USER_ROLE in roles,
        "is_admin": settings.MOOSE_OIDC_ADMIN_ROLE in roles,
    }
    found = await app.state.vault.find_account(claims["sub"]) if account["can_use_vault"] else None
    account["has_vault"] = found is not None
    if found:
        account["user_id"] = found["user_id"]
    return account


@app.post("/account/vault", tags=["Vault"])
async def create_account_vault(claims: dict[str, Any] = Depends(require_account)) -> dict[str, str]:
    """Create the logged-in account's vault. Returns the first API token and the recovery
    code, each shown only once; Moose stores neither."""
    try:
        return await app.state.vault.create_account_vault(claims["sub"], _account_name(claims))
    except VaultError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class SharedKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(default="openrouter", description="One of GET /vault/providers.")
    api_key: str = Field(min_length=8, repr=False)


class SharedModelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] = Field(description="Model IDs everyone may use, e.g. openai/gpt-oss-120b.")
    default_model: str | None = None


@app.get("/admin/shared-models", dependencies=[Depends(require_admin)], tags=["Admin"])
async def get_shared_models() -> dict[str, Any]:
    """The shared provider key (masked only) and the models everyone may use."""
    return await app.state.shared_models.describe()


@app.put("/admin/shared-models/key", dependencies=[Depends(require_admin)], tags=["Admin"])
async def set_shared_key(request: SharedKeyRequest) -> dict[str, Any]:
    """Store the operator's provider key, encrypted with MOOSE_SECRET_KEY. Jobs on shared models use it."""
    try:
        return await app.state.shared_models.set_key(request.provider, request.api_key)
    except SharedModelsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/admin/shared-models", dependencies=[Depends(require_admin)], tags=["Admin"])
async def set_shared_models(request: SharedModelsRequest) -> dict[str, Any]:
    try:
        return await app.state.shared_models.set_models(request.models, request.default_model)
    except SharedModelsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/invites", dependencies=[Depends(require_admin)], tags=["Vault"])
async def create_invite(request: InviteRequest) -> dict[str, str]:
    """One-time invitation for a new user. Send the code to the user; it works once."""
    return await app.state.vault.create_invite(request.note, request.valid_days)


@app.post("/vault/register", tags=["Vault"])
async def register(request: RegisterRequest) -> dict[str, str]:
    """Redeem an invitation. Returns the first API token and a recovery code, each shown
    only once. Moose stores neither, so they cannot be shown again or recovered by the operator."""
    try:
        return await app.state.vault.register(request.invite_code, request.name)
    except VaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/vault/recover", tags=["Vault"])
async def recover(request: RecoverRequest) -> dict[str, str]:
    """Create a new API token with the recovery code, e.g. after losing all tokens."""
    try:
        return await app.state.vault.recover(request.recovery_code, request.label)
    except VaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/vault/providers", tags=["Vault"])
async def list_providers() -> dict[str, list[str]]:
    return {"providers": sorted(PROVIDERS)}


@app.get("/vault/me", tags=["Vault"])
async def vault_me(unlocked: Unlocked = Depends(require_user)) -> dict[str, Any]:
    store: VaultStore = app.state.vault
    return {
        "user_id": unlocked.user_id,
        "current_token_id": unlocked.token_id,
        "tokens": await store.list_tokens(unlocked.user_id),
        "keys": await store.list_secrets(unlocked.user_id),
    }


@app.post("/vault/tokens", tags=["Vault"])
async def create_token(request: TokenRequest, unlocked: Unlocked = Depends(require_user)) -> dict[str, str]:
    """Another API token for the same vault (e.g. one per script or machine), shown once."""
    return await app.state.vault.create_token(unlocked, request.label)


@app.delete("/vault/tokens/{token_id}", tags=["Vault"])
async def revoke_token(token_id: str, unlocked: Unlocked = Depends(require_user)) -> dict[str, bool]:
    if not await app.state.vault.revoke_token(unlocked.user_id, token_id):
        raise HTTPException(status_code=404, detail="Token not found")
    return {"revoked": True}


@app.get("/vault/keys", tags=["Vault"])
async def list_provider_keys(unlocked: Unlocked = Depends(require_user)) -> dict[str, Any]:
    return {"keys": await app.state.vault.list_secrets(unlocked.user_id)}


@app.post("/vault/keys", tags=["Vault"])
async def add_provider_key(request: ProviderKeyRequest, unlocked: Unlocked = Depends(require_user)) -> dict[str, Any]:
    """Store a provider key, encrypted with your vault's data key. Only a masked form is ever returned."""
    try:
        return await app.state.vault.add_secret(
            unlocked, request.provider, request.api_key, request.label, request.default_model
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/vault/keys/{key_id}", tags=["Vault"])
async def delete_provider_key(key_id: str, unlocked: Unlocked = Depends(require_user)) -> dict[str, bool]:
    if not await app.state.vault.delete_secret(unlocked.user_id, key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"deleted": True}


@app.get("/assets", dependencies=[Depends(require_api_key)], tags=["Metadata"])
async def get_assets_index() -> dict:
    from moose.schema import DATA_DIR
    path = DATA_DIR / "assets_index.json"
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"assets_index.json not found at {path}")
    assets = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(assets, dict):
        raise HTTPException(status_code=500, detail=f"assets_index.json is invalid at {path}")

    assets.setdefault("registries", {})
    if isinstance(assets["registries"], dict):
        assets["registries"]["user_vocabularies"] = "user_vocabularies.json"

    assets.setdefault("assets", {})
    if isinstance(assets["assets"], dict):
        assets["assets"]["user_schemas"] = _load_user_schema_assets(DATA_DIR)

    return assets


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Moose API Docs",
        swagger_favicon_url="/static/moose-logo.png",
        swagger_css_url="/static/docs.css",
    )
