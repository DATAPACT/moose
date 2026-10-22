from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st


def _headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"X-API-Key": api_key} if api_key else {}
    if extra:
        h.update(extra)
    return h


def api_get(base_url: str, api_key: str, path: str, extra_headers: dict[str, str] | None = None, params: dict[str, Any] | None = None) -> Any:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{base_url.rstrip('/')}{path}", headers=_headers(api_key, extra_headers), params=params)
        r.raise_for_status()
        return r.json()


def api_post(base_url: str, api_key: str, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> Any:
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{base_url.rstrip('/')}{path}", headers=_headers(api_key, extra_headers), json=payload)
        r.raise_for_status()
        return r.json()


def poll_job(base_url: str, api_key: str, job_id: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return api_get(base_url, api_key, f"/jobs/{job_id}", extra_headers=headers)


def wait_for_job(
    base_url: str,
    api_key: str,
    job_id: str,
    timeout_secs: int = 240,
    poll_interval_secs: float = 1.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.time()
    placeholder = st.empty()

    while True:
        data = poll_job(base_url, api_key, job_id, headers)
        status = data.get("status")
        elapsed = time.time() - start
        placeholder.info(f"Job status: {status} (elapsed: {elapsed:.1f}s)")

        if status in {"completed", "failed"}:
            placeholder.empty()
            return data

        if elapsed > timeout_secs:
            placeholder.empty()
            return {"job_id": job_id, "status": status, "error": f"Timed out after {timeout_secs}s"}

        time.sleep(poll_interval_secs)