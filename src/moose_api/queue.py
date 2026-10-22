from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from moose.config import Settings
from moose.cpa import run_table_cpa
from moose.ingest import ingest_schema_payload
from moose.llm import LLMClient, OpenAICompatibleClient, track_llm_usage
from moose.ner import run_table_annotate, run_tabular_ner, run_text_ner
from moose.privacy import run_privacy_analyze
from moose_api.job_sealing import SEALED_ERROR, is_sealed, open_payload, seal_value


@dataclass
class JobRecord:
    job_id: str
    endpoint_type: str
    payload: dict
    status: str
    created_at: str
    updated_at: str
    retries: int
    result: dict | None = None
    error: str | None = None
    # API client (key name) that submitted the job.
    client: str | None = None
    # Model requested, models that actually answered, tokens and cost (see moose.llm.usage).
    llm: dict | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SINGLE_TASK_ID = "__single__"
_SINGLE_TABLE_ID = "__single_table__"
INTERRUPTED_PARTNER_JOB_ERROR = "Moose restarted before this job finished. Please submit it again."
_NO_LLM_MODEL_ERROR = "No LLM is available for this job: shared models are not set up and no own key was given."

# Credential fields that older Moose versions stored inside job payloads.
_LEGACY_LLM_CREDENTIAL_FIELDS = (
    "openrouter_api_key",
    "ollama_token",
    "deepinfra_api_key",
    "deepseek_api_key",
    "endpoint",
)


def _strip_warning_ids(warnings: Any, keys: set[str]) -> list[Any]:
    if not isinstance(warnings, list):
        return []
    out: list[Any] = []
    for warning in warnings:
        if isinstance(warning, dict):
            out.append({k: v for k, v in warning.items() if k not in keys})
        else:
            out.append(warning)
    return out


def _unwrap_single_ner_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"entities": item.get("entities", [])}
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_tabular_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"columns": item.get("columns", [])}
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_tabular_ner_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {"rows": item.get("rows", [])}
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id", "table_task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


def _unwrap_single_cpa_result(result: dict[str, Any]) -> dict[str, Any]:
    item = (result.get("results") or [{}])[0]
    out: dict[str, Any] = {
        "subject_column": item.get("subject_column"),
        "relationships": item.get("relationships", []),
    }
    table_id = item.get("table_id")
    if isinstance(table_id, str) and table_id and table_id != _SINGLE_TABLE_ID:
        out["table_id"] = table_id
    if isinstance(item.get("debug"), dict):
        out["debug"] = item["debug"]
    warnings = _strip_warning_ids(result.get("warnings"), {"task_id"})
    if warnings:
        out["warnings"] = warnings
    return out


class JobStore:
    async def get_job(self, job_id: str) -> JobRecord | None:
        raise NotImplementedError

    async def put_job(self, job: JobRecord) -> None:
        raise NotImplementedError

    async def update_job(self, job_id: str, **fields: Any) -> None:
        raise NotImplementedError

    async def fail_interrupted_partner_jobs(self) -> int:
        return 0


class InMemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def get_job(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def put_job(self, job: JobRecord) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job

    async def update_job(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            self._jobs[job_id] = job


class MongoJobStore(JobStore):
    def __init__(self, db) -> None:
        self._jobs = db["jobs"]

    async def ensure_indexes(self) -> None:
        await self._jobs.create_index("status")
        await self._jobs.create_index("created_at")
        await self._jobs.create_index("job_id", unique=True)

    async def fail_interrupted_partner_jobs(self) -> int:
        result = await self._jobs.update_many(
            # Users' jobs ran only in the old process: their vault key (to decrypt the input)
            # and any own provider key lived in its memory. Other jobs are in the durable queue.
            {"status": {"$in": ["queued", "processing"]}, "client": {"$regex": "^user:"}},
            {"$set": {"status": "failed", "updated_at": utc_now(), "error": INTERRUPTED_PARTNER_JOB_ERROR}},
        )
        return int(result.modified_count)

    async def scrub_legacy_llm_credentials(self) -> int:
        """Remove LLM provider keys that older versions persisted in job payloads."""
        fields = {f"payload.llm.{name}": "" for name in _LEGACY_LLM_CREDENTIAL_FIELDS}
        query = {"$or": [{key: {"$exists": True}} for key in fields]}
        result = await self._jobs.update_many(query, {"$unset": fields})
        return int(result.modified_count)

    def _doc_to_record(self, doc: dict) -> JobRecord:
        if not doc:
            raise ValueError("Empty job document")
        job_id = doc.get("job_id") or doc.get("_id")
        return JobRecord(
            job_id=job_id,
            endpoint_type=doc["endpoint_type"],
            payload=doc["payload"],
            status=doc["status"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            retries=doc.get("retries", 0),
            result=doc.get("result"),
            error=doc.get("error"),
            client=doc.get("client"),
            llm=doc.get("llm"),
        )

    async def get_job(self, job_id: str) -> JobRecord | None:
        doc = await self._jobs.find_one({"_id": job_id})
        if not doc:
            return None
        return self._doc_to_record(doc)

    async def put_job(self, job: JobRecord) -> None:
        doc = {"_id": job.job_id, **job.__dict__}
        await self._jobs.replace_one({"_id": job.job_id}, doc, upsert=True)

    async def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        await self._jobs.update_one({"_id": job_id}, {"$set": fields})


class QueueBackend:
    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError

    async def dequeue(self) -> str:
        raise NotImplementedError

    async def size(self) -> int:
        raise NotImplementedError

    def set_shutdown_event(self, event: asyncio.Event) -> None:
        return None


class InMemoryQueue(QueueBackend):
    def __init__(self, maxsize: int) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)

    async def enqueue(self, job_id: str) -> None:
        self._queue.put_nowait(job_id)

    async def dequeue(self) -> str:
        return await self._queue.get()

    async def size(self) -> int:
        return self._queue.qsize()


class MongoQueue(QueueBackend):
    def __init__(self, db, poll_interval: float = 0.2) -> None:
        self._queue = db["queue"]
        self._poll_interval = poll_interval
        self._shutdown: asyncio.Event | None = None

    async def ensure_indexes(self) -> None:
        await self._queue.create_index([("created_at", 1)])

    def set_shutdown_event(self, event: asyncio.Event) -> None:
        self._shutdown = event

    async def enqueue(self, job_id: str) -> None:
        from pymongo.errors import DuplicateKeyError

        doc = {"_id": job_id, "job_id": job_id, "created_at": utc_now()}
        try:
            await self._queue.insert_one(doc)
        except DuplicateKeyError:
            return None

    async def dequeue(self) -> str:
        while True:
            if self._shutdown and self._shutdown.is_set():
                raise asyncio.CancelledError
            doc = await self._queue.find_one_and_delete({}, sort=[("created_at", 1)])
            if doc:
                return doc["job_id"]
            await asyncio.sleep(self._poll_interval)

    async def size(self) -> int:
        return int(await self._queue.count_documents({}))


class WorkerPool:
    def __init__(
        self,
        queue: QueueBackend,
        store: JobStore,
        shared_models: Any,
        settings: Settings,
    ) -> None:
        self._queue = queue
        self._store = store
        self._settings = settings
        # shared_models.SharedModelsStore: the operator's key for shared models.
        self._shared_models = shared_models
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._queue.set_shutdown_event(self._shutdown)
        # Jobs on a user's own provider key, with that key, kept in this process's memory only.
        self._local_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=settings.MOOSE_QUEUE_MAXSIZE)

    async def start(self) -> None:
        # User keys lived only in the memory of the previous process, so its unfinished
        # jobs on user keys can never run.
        await self._store.fail_interrupted_partner_jobs()
        for idx in range(self._settings.MOOSE_WORKER_COUNT):
            self._tasks.append(asyncio.create_task(self._worker_loop(idx)))
            self._tasks.append(asyncio.create_task(self._local_worker_loop(idx)))

    def enqueue_local(self, job_id: str, route: Any, data_key: bytes | None = None) -> None:
        """Queue a user's job with what only this process may hold: `route` (a
        vault_store.ProviderRoute) for a job on their own key, and `data_key` to decrypt
        its input and encrypt its result."""
        self._local_queue.put_nowait((job_id, route, data_key))

    def local_queue_size(self) -> int:
        return self._local_queue.qsize()

    async def stop(self) -> None:
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _client(self, route: Any, model: str) -> LLMClient:
        return OpenAICompatibleClient(
            base_url=route.base_url,
            api_key=route.api_key,
            model=model,
            timeout=self._settings.MOOSE_TIMEOUT_SECS,
            provider=route.provider,
            extra_headers={"X-Title": "Moose"} if route.provider == "openrouter" else None,
        )

    async def shared_client(self, model: str) -> LLMClient:
        """A client on the operator's key for a shared model. The caller closes it."""
        return self._client(await self._shared_models.route(), model)

    async def _worker_loop(self, worker_id: int) -> None:
        while not self._shutdown.is_set():
            job_id = await self._queue.dequeue()
            await self._process_job(job_id)

    async def _local_worker_loop(self, worker_id: int) -> None:
        while not self._shutdown.is_set():
            job_id, route, data_key = await self._local_queue.get()
            await self._process_job(job_id, route, data_key)
            del route, data_key

    async def _process_job(self, job_id: str, route: Any = None, data_key: bytes | None = None) -> None:
        job = await self._store.get_job(job_id)
        if not job:
            return
        if is_sealed(job.payload):
            if data_key is None:
                await self._store.update_job(
                    job_id, status="failed", updated_at=utc_now(), error=INTERRUPTED_PARTNER_JOB_ERROR
                )
                return
            # Decrypted in memory only; the stored job keeps the sealed input.
            job = replace(job, payload=open_payload(data_key, job_id, job.payload))

        await self._store.update_job(job_id, status="processing", updated_at=utc_now())

        model = (job.payload.get("llm") or {}).get("model")
        llm_client: LLMClient | None = None
        no_llm_reason = _NO_LLM_MODEL_ERROR
        if model:
            if route is None:
                # Shared model: the operator's key, decrypted in memory for this job.
                try:
                    route = await self._shared_models.route()
                except Exception as exc:  # noqa: BLE001 - e.g. no key, or stored under another secret
                    route = None
                    no_llm_reason = f"Shared models cannot be used: {exc}"
            # Otherwise a user's own key: their models, their bill.
            if route is not None:
                llm_client = self._client(route, model)

        with track_llm_usage(model) as usage:
            try:
                if job.endpoint_type == "ner":
                    if llm_client is None:
                        raise RuntimeError(no_llm_reason)
                    single_task = {"task_id": _SINGLE_TASK_ID, "text": job.payload["text"]}
                    raw = await run_text_ner(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_ner_result(raw)

                elif job.endpoint_type == "tabular":
                    if llm_client is None:
                        raise RuntimeError(no_llm_reason)
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                    }
                    raw = await run_table_annotate(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_tabular_result(raw)

                elif job.endpoint_type == "tabular_ner":
                    if llm_client is None:
                        raise RuntimeError(no_llm_reason)
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                        "target_columns": job.payload["target_columns"],
                        "strings_only": job.payload.get("strings_only", True),
                        "skip_structured_literals": job.payload.get("skip_structured_literals", True),
                    }
                    raw = await run_tabular_ner(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_tabular_ner_result(raw)

                elif job.endpoint_type == "privacy_analyze":
                    if llm_client is None:
                        raise RuntimeError(no_llm_reason)
                    result = await run_privacy_analyze(
                        tasks=job.payload["tasks"],
                        policy_pack=job.payload.get("policy_pack"),
                        profile=job.payload.get("profile"),
                        analysis_mode=job.payload.get("analysis_mode"),
                        text_schema=job.payload.get("text_schema"),
                        table_schema=job.payload.get("table_schema"),
                        scan_schema=job.payload.get("scan_schema"),
                        include_extraction=job.payload.get("include_extraction"),
                        llm_client=llm_client,
                        settings=self._settings,
                    )

                elif job.endpoint_type == "cpa":
                    if llm_client is None:
                        raise RuntimeError(no_llm_reason)
                    single_task = {
                        "task_id": _SINGLE_TASK_ID,
                        "table_id": job.payload.get("table_id") or _SINGLE_TABLE_ID,
                        "sampled_rows": job.payload["sampled_rows"],
                        "subject_column": job.payload["subject_column"],
                        "subject_class": job.payload.get("subject_class"),
                        "target_columns": job.payload.get("target_columns"),
                        "use_sti_signature_cache": job.payload.get("use_sti_signature_cache", True),
                        "debug": job.payload.get("debug", False),
                        "debug_preview_limit": job.payload.get("debug_preview_limit", 20),
                    }
                    raw = await run_table_cpa(
                        [single_task],
                        job.payload["schema"],
                        llm_client,
                        settings=self._settings,
                    )
                    result = _unwrap_single_cpa_result(raw)

                elif job.endpoint_type == "schema_ingest":
                    result = await ingest_schema_payload(job.payload, llm_client=llm_client)

                else:
                    raise ValueError(f"Unknown endpoint_type: {job.endpoint_type}")

                await self._store.update_job(
                    job_id,
                    status="completed",
                    updated_at=utc_now(),
                    result=seal_value(data_key, job_id, "result", result) if data_key else result,
                    llm=usage.as_dict(),
                )

            except Exception as exc:  # noqa: BLE001
                # A user's error details can quote their input, so they are sealed too.
                await self._store.update_job(
                    job_id,
                    status="failed",
                    updated_at=utc_now(),
                    error=SEALED_ERROR if data_key else str(exc),
                    result=seal_value(data_key, job_id, "error", str(exc)) if data_key else None,
                    llm=usage.as_dict(),
                )
        if llm_client is not None:
            await llm_client.close()



async def build_backends(settings: Settings) -> tuple[JobStore, QueueBackend, dict]:
    info: dict[str, Any] = {"queue_backend": "memory"}
    if settings.MOOSE_MONGO_URL:
        try:
            import motor.motor_asyncio as motor
        except ImportError:
            return InMemoryJobStore(), InMemoryQueue(settings.MOOSE_QUEUE_MAXSIZE), info

        timeout_ms = max(1, int(settings.MOOSE_MONGO_TIMEOUT_SECS * 1000))
        client = motor.AsyncIOMotorClient(
            settings.MOOSE_MONGO_URL,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
        )
        try:
            await client.admin.command("ping")
            db = client[settings.MOOSE_MONGO_DB]
            store = MongoJobStore(db)
            queue = MongoQueue(db)
            await store.ensure_indexes()
            info["scrubbed_legacy_credentials"] = await store.scrub_legacy_llm_credentials()
            await queue.ensure_indexes()
            info["queue_backend"] = "mongo"
            return store, queue, {"mongo_client": client, **info}
        except Exception:  # noqa: BLE001
            client.close()

    return InMemoryJobStore(), InMemoryQueue(settings.MOOSE_QUEUE_MAXSIZE), info
