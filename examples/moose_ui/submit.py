from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from .api import api_post, wait_for_job
from .config import llm_options
from .render import render_job
from .state import add_job_history


def _render_api_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        status_code = response.status_code if response is not None else "?"
        detail = str(exc)
        if response is not None:
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("detail"):
                    detail = str(payload["detail"])
            except Exception:
                pass
        st.error(f"Request failed ({status_code}): {detail}")
        if response is not None:
            with st.expander("API error response", expanded=False):
                try:
                    st.json(response.json())
                except Exception:
                    st.code(response.text)
        return
    st.error(str(exc))


def submit_and_render_job(
    *,
    cfg: dict[str, Any],
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    label: str,
    auto_poll: bool,
    persist_result_key: str | None = None,
) -> None:
    if "llm" not in payload and llm_options(cfg):
        payload = {**payload, "llm": llm_options(cfg)}
    try:
        resp = api_post(cfg["base_url"], cfg["api_key"], path, payload, headers)
    except Exception as exc:  # noqa: BLE001
        _render_api_error(exc)
        return
    job_id = resp.get("job_id", "")
    add_job_history(job_id, label)

    st.success("Job submitted.")
    st.json(resp)

    if auto_poll and job_id:
        final = wait_for_job(cfg["base_url"], cfg["api_key"], job_id)
        if persist_result_key:
            st.session_state[persist_result_key] = final
        render_job(
            final,
            show_raw=cfg.get("show_raw", False),
            show_legal_refs=cfg.get("show_legal_refs", True),
            show_legal_detail=cfg.get("show_legal_detail", True),
            show_debug=cfg.get("show_debug", False),
        )
