from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from moose_ui.api import api_post, wait_for_job
from moose_ui.account import admin_headers, auth_disabled_banner, logged_in, login_configured
from moose_ui.config import llm_options, sidebar, validate_common
from moose_ui.metadata import clear_metadata_caches, fetch_schemas
from moose_ui.render import render_job
from moose_ui.state import add_job_history

_PREVIEW_RESULT_KEY = "byo_schema_preview_result"
_PREVIEW_FINGERPRINT_KEY = "byo_schema_preview_fingerprint"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_source_path(path_value: str) -> str:
    raw = (path_value or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.exists():
        return str(p)

    # Rewrite host-specific absolute paths (e.g. /Users/.../examples/...) to the current repo root.
    marker = "/examples/"
    raw_norm = raw.replace("\\", "/")
    if marker in raw_norm:
        suffix = raw_norm.split(marker, 1)[1]
        candidate = _repo_root() / "examples" / suffix
        if candidate.exists():
            return str(candidate)

    return raw


def _load_request_presets() -> dict[str, dict[str, Any]]:
    req_dir = Path(__file__).resolve().parents[1] / "schema_ingest_samples" / "requests"
    out: dict[str, dict[str, Any]] = {}
    if not req_dir.exists():
        return out
    for file_path in sorted(req_dir.glob("ingest_*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            if str(data.get("source_type")) == "file" and isinstance(data.get("source_path"), str):
                data["source_path"] = _resolve_source_path(data["source_path"])
            out[file_path.name] = data
    return out


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_payload(
    *,
    name: str,
    label: str,
    description: str,
    source_type: str,
    source_path: str,
    source_url: str,
    github_repo: str,
    github_ref: str,
    github_path: str,
    supports_text: bool,
    supports_table: bool,
    supports_cpa: bool,
    prefilter_types: bool,
    score_mode: str,
    use_llm_fallback: bool,
    text_intro: str,
    table_intro: str,
    cpa_intro: str,
    cfg: dict[str, Any],
    err: str | None,
    expected_source_sha256: str | None = None,
    expected_mapping_sha256: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not name.strip():
        return None, "Schema name is required."
    if source_type == "file" and not source_path.strip():
        return None, "source_path is required for source_type=file."
    if source_type == "url" and not source_url.strip():
        return None, "source_url is required for source_type=url."
    if source_type == "github" and (not github_repo.strip() or not github_path.strip()):
        return None, "github_repo and github_path are required for source_type=github."
    if use_llm_fallback and err:
        return None, err

    payload: dict[str, Any] = {
        "name": name.strip(),
        "source_type": source_type,
        "supports_text": supports_text,
        "supports_table": supports_table,
        "supports_cpa": supports_cpa,
        "prefilter_types": prefilter_types,
        "score_mode": score_mode,
        "use_llm_fallback": use_llm_fallback,
    }
    if label.strip():
        payload["label"] = label.strip()
    if description.strip():
        payload["description"] = description.strip()
    if source_type == "file":
        payload["source_path"] = _resolve_source_path(source_path.strip())
    elif source_type == "url":
        payload["source_url"] = source_url.strip()
    else:
        payload["github_repo"] = github_repo.strip()
        payload["github_ref"] = github_ref.strip() or "main"
        payload["github_path"] = github_path.strip()
    if text_intro.strip():
        payload["text_intro"] = text_intro.strip()
    if table_intro.strip():
        payload["table_intro"] = table_intro.strip()
    if cpa_intro.strip():
        payload["cpa_intro"] = cpa_intro.strip()
    if use_llm_fallback and llm_options(cfg):
        payload["llm"] = llm_options(cfg)
    if expected_source_sha256:
        payload["expected_source_sha256"] = expected_source_sha256
    if expected_mapping_sha256:
        payload["expected_mapping_sha256"] = expected_mapping_sha256
    return payload, None


st.title("BYO Schema")

cfg = sidebar()
err = validate_common(cfg)
if err:
    st.warning(err)

# Changing schemas is administration: it needs a login with the admin role.
if not login_configured():
    st.error("Account login is not configured on this server, so schemas cannot be changed here.")
    st.stop()
auth_disabled_banner()
if not logged_in():
    st.write("Log in with an administrator account to add or change schemas.")
    st.button("Log in", type="primary", on_click=st.login)
    st.stop()

st.caption("Workflow: preview mapping and extracted types first, then activate schema.")

presets = _load_request_presets()

top_left, top_right = st.columns([2, 1])
with top_left:
    preset_name = st.selectbox("Ingest preset (optional)", ["(none)"] + list(presets.keys()))
with top_right:
    auto_poll = st.checkbox("Auto-poll", value=cfg["auto_poll_default"])

preset_payload = presets.get(preset_name, {}) if preset_name != "(none)" else {}

name = st.text_input("Schema name", value=str(preset_payload.get("name") or ""))
label = st.text_input("Label (optional)", value=str(preset_payload.get("label") or ""))
description = st.text_area("Description (optional)", value=str(preset_payload.get("description") or ""), height=90)

source_type = st.selectbox(
    "source_type",
    ["file", "url", "github"],
    index=["file", "url", "github"].index(str(preset_payload.get("source_type") or "file")),
)

source_url = ""
source_path = ""
github_repo = ""
github_ref = "main"
github_path = ""

if source_type == "file":
    source_path = st.text_input(
        "source_path",
        value=_resolve_source_path(str(preset_payload.get("source_path") or "")),
    )
elif source_type == "url":
    source_url = st.text_input("source_url", value=str(preset_payload.get("source_url") or ""))
else:
    github_repo = st.text_input("github_repo", value=str(preset_payload.get("github_repo") or ""))
    github_ref = st.text_input("github_ref", value=str(preset_payload.get("github_ref") or "main"))
    github_path = st.text_input("github_path", value=str(preset_payload.get("github_path") or ""))

col1, col2, col3, col4 = st.columns(4)
with col1:
    supports_text = st.checkbox("supports_text", value=bool(preset_payload.get("supports_text", True)))
with col2:
    supports_table = st.checkbox("supports_table", value=bool(preset_payload.get("supports_table", True)))
with col3:
    supports_cpa = st.checkbox("supports_cpa", value=bool(preset_payload.get("supports_cpa", False)))
with col4:
    prefilter_types = st.checkbox("prefilter_types", value=bool(preset_payload.get("prefilter_types", False)))

score_mode = st.selectbox(
    "score_mode",
    ["sparse", "dense"],
    index=["sparse", "dense"].index(str(preset_payload.get("score_mode") or "sparse")),
)
use_llm_fallback = st.checkbox("use_llm_fallback", value=bool(preset_payload.get("use_llm_fallback", False)))

st.markdown("Optional prompt intros")
text_intro = st.text_area("text_intro", value=str(preset_payload.get("text_intro") or ""), height=80)
table_intro = st.text_area("table_intro", value=str(preset_payload.get("table_intro") or ""), height=80)
cpa_intro = st.text_area("cpa_intro", value=str(preset_payload.get("cpa_intro") or ""), height=80)

payload, payload_error = _build_payload(
    name=name,
    label=label,
    description=description,
    source_type=source_type,
    source_path=source_path,
    source_url=source_url,
    github_repo=github_repo,
    github_ref=github_ref,
    github_path=github_path,
    supports_text=supports_text,
    supports_table=supports_table,
    supports_cpa=supports_cpa,
    prefilter_types=prefilter_types,
    score_mode=score_mode,
    use_llm_fallback=use_llm_fallback,
    text_intro=text_intro,
    table_intro=table_intro,
    cpa_intro=cpa_intro,
    cfg=cfg,
    err=err,
)

current_fingerprint = _payload_fingerprint(payload) if payload else ""
preview = st.session_state.get(_PREVIEW_RESULT_KEY)
preview_fingerprint = st.session_state.get(_PREVIEW_FINGERPRINT_KEY, "")
preview_matches_current = bool(preview) and preview_fingerprint == current_fingerprint

action_left, action_right = st.columns([1, 1])
with action_left:
    if st.button("1) Preview schema"):
        if payload_error:
            st.error(payload_error)
            st.stop()
        try:
            preview_resp = api_post(
                cfg["base_url"],
                "",
                "/schemas/ingest/preview",
                payload,
                admin_headers(),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()
        st.session_state[_PREVIEW_RESULT_KEY] = preview_resp
        st.session_state[_PREVIEW_FINGERPRINT_KEY] = current_fingerprint
        preview = preview_resp
        preview_matches_current = True
        st.success("Preview ready. Review results, then activate.")

with action_right:
    activate_disabled = (not preview_matches_current) or not bool((preview or {}).get("can_activate", False))
    if st.button("2) Activate schema", type="primary", disabled=activate_disabled):
        if payload_error:
            st.error(payload_error)
            st.stop()
        activate_payload, activate_payload_error = _build_payload(
            name=name,
            label=label,
            description=description,
            source_type=source_type,
            source_path=source_path,
            source_url=source_url,
            github_repo=github_repo,
            github_ref=github_ref,
            github_path=github_path,
            supports_text=supports_text,
            supports_table=supports_table,
            supports_cpa=supports_cpa,
            prefilter_types=prefilter_types,
            score_mode=score_mode,
            use_llm_fallback=use_llm_fallback,
            text_intro=text_intro,
            table_intro=table_intro,
            cpa_intro=cpa_intro,
            cfg=cfg,
            err=err,
            expected_source_sha256=str((preview or {}).get("source_sha256") or ""),
            expected_mapping_sha256=str((preview or {}).get("mapping_sha256") or ""),
        )
        if activate_payload_error or not activate_payload:
            st.error(activate_payload_error or "Activation payload invalid.")
            st.stop()
        try:
            resp = api_post(
                cfg["base_url"],
                "",
                "/schemas/ingest",
                activate_payload,
                admin_headers(),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

        job_id = str(resp.get("job_id") or "")
        add_job_history(job_id, f"Schema ingest ({name.strip()})")
        st.success("Schema ingest job submitted.")
        st.json(resp)

        if auto_poll and job_id:
            final = wait_for_job(cfg["base_url"], "", job_id, headers=admin_headers())
            render_job(
                final,
                show_raw=cfg.get("show_raw", False),
                show_legal_refs=cfg.get("show_legal_refs", True),
                show_legal_detail=cfg.get("show_legal_detail", True),
                show_debug=cfg.get("show_debug", False),
            )
            if final.get("status") == "completed":
                clear_metadata_caches()
                st.success("Metadata cache refreshed. New schema should now appear in Text/Tables/Privacy.")

if preview and isinstance(preview, dict):
    st.divider()
    st.subheader("Preview result")
    st.markdown(
        f"**schema:** `{preview.get('schema')}`  \n"
        f"**type_count:** `{preview.get('type_count')}`  \n"
        f"**mapping_strategy:** `{preview.get('mapping_strategy')}`  \n"
        f"**mapping_confidence:** `{preview.get('mapping_confidence')}`  \n"
        f"**can_activate:** `{preview.get('can_activate')}`"
    )
    sample_ids = preview.get("sample_type_ids")
    if isinstance(sample_ids, list):
        st.caption(f"sample_type_ids ({len(sample_ids)} shown):")
        st.code("\n".join(str(x) for x in sample_ids), language="text")

    with st.expander("Mapping details", expanded=False):
        st.json(preview.get("mapping"))
    with st.expander("Warnings", expanded=False):
        st.json(preview.get("warnings") or [])
    with st.expander("Guardrails", expanded=False):
        st.json(preview.get("guardrails") or {})

    if not preview_matches_current:
        st.warning("Form values changed since preview. Run Preview again before activating.")
    elif not bool(preview.get("can_activate")):
        st.warning("Preview guardrails did not pass. Fix findings under Guardrails before activation.")

st.divider()
st.subheader("Current schemas")
if st.button("Refresh and list schemas"):
    clear_metadata_caches()
    try:
        schemas = fetch_schemas(cfg["base_url"], cfg["api_key"])
        st.json(schemas)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch schemas: {exc}")
