"""Moose as a Python library: annotate text and tables on your own machine.

LLM calls go straight from this process to your provider with your key, and the System 1
model (Laya, `pip install "moose-annotate[laya]"`) runs locally; nothing passes through a
Moose server. Results have the same shape as the REST API's job results.

    from moose import Annotator, OpenRouterClient

    moose = Annotator(OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-oss-120b"))
    moose.annotate_text("Thanks, Luca Bianchi. Your customer ID is USR 661 204.")
    moose.last_usage  # LLM calls, tokens and cost of the last call

Every method has an async twin with an `a` prefix (`aannotate_text`, ...). The sync
methods also work inside a running event loop, such as a Jupyter notebook. Use either the
sync or the async methods with one LLM client: its connections belong to one event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Literal, TypeVar

from moose.config import Settings
from moose.cpa import run_table_cpa
from moose.llm.base import LLMClient
from moose.llm.usage import LLMUsage, track_llm_usage
from moose.ner import run_table_annotate, run_text_ner
from moose.privacy import run_privacy_analyze
from moose.schema import list_schema_names

T = TypeVar("T")

Linker = Literal["system1+llm", "system1", "llm"]


class _Loop:
    """One event loop in a background thread, for the sync methods: the LLM client's
    connections belong to the loop they were opened on, so every sync call uses the same
    one. This also works where a loop is already running, such as a Jupyter notebook."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def run(self, coro: Awaitable[T]) -> T:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(target=self._loop.run_forever, name="moose-annotator", daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()  # type: ignore[arg-type]


class Annotator:
    """Annotates text and tables with one LLM client (`moose.OpenRouterClient`,
    `moose.OllamaClient`, `moose.OpenAICompatibleClient`, or your own `moose.LLMClient`)."""

    def __init__(self, llm: LLMClient, *, max_retries: int = 2) -> None:
        self.llm = llm
        self.settings = Settings(MOOSE_MAX_RETRIES=max_retries)
        # LLM calls, tokens and cost of the most recent call on this annotator.
        self.last_usage: LLMUsage | None = None
        self._loop = _Loop()

    async def _tracked(self, coro: Awaitable[T]) -> T:
        with track_llm_usage(getattr(self.llm, "model", None)) as usage:
            try:
                return await coro
            finally:
                self.last_usage = usage

    # Text ---------------------------------------------------------------------------

    async def aannotate_texts(self, texts: list[str], schema: str = "dpv_privacy", linker: Linker | None = None) -> list[dict[str, Any]]:
        """Entities in each text: span, type_id, confidence (the System 1 model's), and how
        the type was decided."""
        tasks = [{"task_id": str(i), "text": text} for i, text in enumerate(texts)]
        out = await self._tracked(run_text_ner(tasks, schema, self.llm, self.settings, linker=linker))
        return [{k: v for k, v in r.items() if k != "task_id"} for r in out["results"]]

    async def aannotate_text(self, text: str, schema: str = "dpv_privacy", linker: Linker | None = None) -> dict[str, Any]:
        return (await self.aannotate_texts([text], schema, linker))[0]

    def annotate_texts(self, texts: list[str], schema: str = "dpv_privacy", linker: Linker | None = None) -> list[dict[str, Any]]:
        return self._loop.run(self.aannotate_texts(texts, schema, linker))

    def annotate_text(self, text: str, schema: str = "dpv_privacy", linker: Linker | None = None) -> dict[str, Any]:
        return self._loop.run(self.aannotate_text(text, schema, linker))

    # Tables -------------------------------------------------------------------------

    async def aannotate_table(
        self, rows: list[dict[str, Any]], schema: str = "dpv_privacy", table_id: str = "table", linker: Linker | None = None
    ) -> dict[str, Any]:
        """A type for each column, from a sample of rows (dicts of column -> value)."""
        tasks = [{"task_id": "0", "table_id": table_id, "sampled_rows": rows}]
        out = await self._tracked(run_table_annotate(tasks, schema, self.llm, self.settings, linker=linker))
        return {k: v for k, v in out["results"][0].items() if k != "task_id"}

    def annotate_table(
        self, rows: list[dict[str, Any]], schema: str = "dpv_privacy", table_id: str = "table", linker: Linker | None = None
    ) -> dict[str, Any]:
        return self._loop.run(self.aannotate_table(rows, schema, table_id, linker))

    async def acolumn_relationships(
        self,
        rows: list[dict[str, Any]],
        subject_column: str,
        schema: str = "cpa",
        target_columns: list[str] | None = None,
        subject_class: str | None = None,
        table_id: str = "table",
    ) -> dict[str, Any]:
        """The relationship between the subject column and each target column (CPA)."""
        task: dict[str, Any] = {"task_id": "0", "table_id": table_id, "sampled_rows": rows, "subject_column": subject_column}
        if target_columns is not None:
            task["target_columns"] = target_columns
        if subject_class is not None:
            task["subject_class"] = subject_class
        out = await self._tracked(run_table_cpa([task], schema, self.llm, self.settings))
        return {k: v for k, v in out["results"][0].items() if k != "task_id"}

    def column_relationships(
        self,
        rows: list[dict[str, Any]],
        subject_column: str,
        schema: str = "cpa",
        target_columns: list[str] | None = None,
        subject_class: str | None = None,
        table_id: str = "table",
    ) -> dict[str, Any]:
        return self._loop.run(self.acolumn_relationships(rows, subject_column, schema, target_columns, subject_class, table_id))

    # Privacy ------------------------------------------------------------------------

    async def aanalyze_privacy(
        self,
        text: str | None = None,
        rows: list[dict[str, Any]] | None = None,
        *,
        profile: str | None = None,
        policy_pack: str | None = None,
        scan_columns: list[str] | None = None,
        include_extraction: bool | None = None,
    ) -> dict[str, Any]:
        """Privacy findings and recommended actions for a text or a table (pass one)."""
        if (text is None) == (rows is None):
            raise ValueError("Pass either text or rows.")
        if text is not None:
            task: dict[str, Any] = {"kind": "text", "task_id": "0", "text": text}
        else:
            task = {"kind": "table", "task_id": "0", "table_id": "table", "sampled_rows": rows}
            if scan_columns is not None:
                task["scan_columns"] = scan_columns
        return await self._tracked(run_privacy_analyze(
            [task], policy_pack, self.llm, profile=profile, include_extraction=include_extraction, settings=self.settings,
        ))

    def analyze_privacy(
        self,
        text: str | None = None,
        rows: list[dict[str, Any]] | None = None,
        *,
        profile: str | None = None,
        policy_pack: str | None = None,
        scan_columns: list[str] | None = None,
        include_extraction: bool | None = None,
    ) -> dict[str, Any]:
        return self._loop.run(self.aanalyze_privacy(
            text, rows, profile=profile, policy_pack=policy_pack, scan_columns=scan_columns, include_extraction=include_extraction,
        ))

    # Schemas ------------------------------------------------------------------------

    @staticmethod
    def schemas() -> list[str]:
        """Names of the available schemas, built in and your own."""
        return list_schema_names()

    async def aadd_schema(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add your own schema (bring your own schema), as the API's /schemas/ingest does. It
        is stored under MOOSE_USER_DATA_DIR (default ~/.moose for an installed package)."""
        from moose.ingest import ingest_schema_payload

        return await self._tracked(ingest_schema_payload(payload, self.llm))

    def add_schema(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._loop.run(self.aadd_schema(payload))

    @staticmethod
    def remove_schema(name: str) -> dict[str, Any]:
        """Remove one of your own schemas and its files."""
        from moose.ingest import delete_user_schema

        return delete_user_schema(name)
