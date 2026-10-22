from __future__ import annotations

import json
from typing import Any

import streamlit as st

from moose_ui.config import sidebar, validate_common
from moose_ui.metadata import fetch_policy_packs, fetch_privacy_profiles, fetch_schemas, schemas_supporting
from moose_ui.render import render_job
from moose_ui.samples import DEFAULT_TABLE_SAMPLE
from moose_ui.submit import submit_and_render_job


def _infer_table_columns(sampled_rows: Any) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    if not isinstance(sampled_rows, list):
        return cols
    for row in sampled_rows:
        if not isinstance(row, dict):
            continue
        for k in row.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)
    return cols


def _get_profile_defaults(profiles_payload: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profiles_obj = profiles_payload.get("profiles") or {}
    if not isinstance(profiles_obj, dict):
        return {}
    p = profiles_obj.get(profile_name) or {}
    if not isinstance(p, dict):
        return {}
    d = p.get("defaults") or {}
    return d if isinstance(d, dict) else {}


def _resolve_effective(
    profile_defaults: dict[str, Any],
    policy_pack: str,
    advanced: bool,
    analysis_mode: str,
    include_extraction: str,
    text_schema: str,
    table_schema: str,
    scan_schema: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Returns (effective_config, source_map) where source_map says 'profile' or 'override'.
    """
    effective = dict(profile_defaults)
    source_map = {k: "profile" for k in effective.keys()}

    # policy_pack is always set explicitly by this UI
    effective["policy_pack"] = policy_pack
    source_map["policy_pack"] = "override"

    # Overrides (only when user picked non-default)
    if advanced:
        if analysis_mode != "(profile default)":
            effective["analysis_mode"] = analysis_mode
            source_map["analysis_mode"] = "override"

        if include_extraction != "(profile default)":
            effective["include_extraction"] = (include_extraction == "true")
            source_map["include_extraction"] = "override"

        if text_schema != "(profile default)":
            effective["text_schema"] = text_schema
            source_map["text_schema"] = "override"

        if table_schema != "(profile default)":
            effective["table_schema"] = table_schema
            source_map["table_schema"] = "override"

        if scan_schema != "(profile default)":
            effective["scan_schema"] = scan_schema
            source_map["scan_schema"] = "override"

    # Fill missing keys with sensible placeholders (shouldn't happen if profiles are correct)
    effective.setdefault("analysis_mode", None)
    effective.setdefault("text_schema", None)
    effective.setdefault("table_schema", None)
    effective.setdefault("scan_schema", None)
    effective.setdefault("include_extraction", None)

    return effective, source_map


st.title("Privacy")

cfg = sidebar()
err = validate_common(cfg)
if err:
    st.warning(err)

if not cfg.get("api_key"):
    st.info("Enter your Moose API key in the sidebar.")
    st.stop()

PERSIST_RESULT_KEY = "privacy_last_job"

# Metadata
try:
    policy_packs = fetch_policy_packs(cfg["base_url"], cfg["api_key"]) or ["gdpr_basic"]
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch policy packs: {exc}")
    policy_packs = ["gdpr_basic"]

try:
    profiles_payload = fetch_privacy_profiles(cfg["base_url"], cfg["api_key"])
    profiles = list((profiles_payload.get("profiles") or {}).keys())
    default_profile = profiles_payload.get("default_profile") or "balanced"
    if default_profile not in profiles and profiles:
        default_profile = profiles[0]
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch privacy profiles: {exc}")
    profiles = ["fast", "balanced", "deep"]
    default_profile = "balanced"
    profiles_payload = {}

# Schemas only needed for advanced overrides
schemas = []
try:
    schemas = fetch_schemas(cfg["base_url"], cfg["api_key"])
except Exception:
    schemas = []

text_schema_names = schemas_supporting(schemas, text=True)
table_schema_names = schemas_supporting(schemas, table=True)

st.subheader("Settings")

profile = st.selectbox("profile", profiles, index=(profiles.index(default_profile) if default_profile in profiles else 0))

# policy_pack override (always explicit for now)
policy_pack = st.selectbox(
    "policy_pack",
    policy_packs,
    index=(policy_packs.index("gdpr_basic") if "gdpr_basic" in policy_packs else 0),
)

auto_poll = st.checkbox("Auto-poll", value=cfg["auto_poll_default"])

advanced = cfg.get("advanced_mode", False)
if advanced:
    st.markdown("### Advanced overrides")
    analysis_mode = st.selectbox("analysis_mode override", ["(profile default)", "rules", "hybrid"], index=0)
    include_extraction = st.selectbox("include_extraction override", ["(profile default)", "true", "false"], index=0)

    # schema overrides
    text_schema = st.selectbox("text_schema override", ["(profile default)"] + (text_schema_names or ["dpv_pd", "dpv"]), index=0)
    table_schema = st.selectbox("table_schema override", ["(profile default)"] + (table_schema_names or ["sti"]), index=0)
    scan_schema = st.selectbox("scan_schema override", ["(profile default)"] + (text_schema_names or ["dpv_pd", "dpv"]), index=0)
else:
    analysis_mode = "(profile default)"
    include_extraction = "(profile default)"
    text_schema = "(profile default)"
    table_schema = "(profile default)"
    scan_schema = "(profile default)"

# ------------------------
# Profile summary + Effective config summary (compact)
# ------------------------
profile_defaults = _get_profile_defaults(profiles_payload, profile)
effective_cfg, cfg_source = _resolve_effective(
    profile_defaults,
    policy_pack,
    advanced,
    analysis_mode,
    include_extraction,
    text_schema,
    table_schema,
    scan_schema,
)

# Small 1-line summary (non-technical)
mode_label = effective_cfg.get("analysis_mode") or "unknown"
text_label = effective_cfg.get("text_schema") or "unknown"
table_label = effective_cfg.get("table_schema") or "unknown"
scan_label = effective_cfg.get("scan_schema") or "unknown"
extract_label = "on" if effective_cfg.get("include_extraction") else "off"

st.info(
    f"This run will use: Reasoning **{mode_label}** | Text **{text_label}** | "
    f"Table **{table_label}** | Scan **{scan_label}** | Extraction **{extract_label}**"
)

# Optional detailed expander (still small)
with st.expander("Profile & configuration details", expanded=False):
    # Profile description (short)
    prof_obj = (profiles_payload.get("profiles") or {}).get(profile, {}) if isinstance(profiles_payload.get("profiles"), dict) else {}
    prof_desc = prof_obj.get("description") if isinstance(prof_obj, dict) else None
    if isinstance(prof_desc, str) and prof_desc.strip():
        st.caption(f"Profile **{profile}**: {prof_desc.strip()}")

    # Compact table of effective settings
    rows = []
    for k in ["policy_pack", "analysis_mode", "include_extraction", "text_schema", "table_schema", "scan_schema"]:
        rows.append(
            {
                "Setting": k,
                "Value": effective_cfg.get(k),
                "Source": "Override" if cfg_source.get(k) == "override" else "Profile default",
            }
        )
    st.table(rows)

    st.caption(
        "Tip: Most users can ignore Advanced overrides. Profiles pick a sensible balance of speed vs accuracy."
    )

tab_text, tab_table = st.tabs(["Text privacy", "Table privacy"])
ran_submission = False

with tab_text:
    st.subheader("Text privacy analysis")
    text = st.text_area(
        "Text",
        value="We collect email addresses and IP addresses and share them with our payment processor.",
        height=130,
    )

    if st.button("Run privacy analysis (text)", type="primary"):
        if err:
            st.error(err)
            st.stop()
        if not text.strip():
            st.error("Text is empty.")
            st.stop()

        payload: dict[str, Any] = {
            "profile": profile,
            "policy_pack": policy_pack,
            "tasks": [{"kind": "text", "task_id": "pt1", "text": text}],
        }

        if advanced:
            if analysis_mode != "(profile default)":
                payload["analysis_mode"] = analysis_mode
            if include_extraction != "(profile default)":
                payload["include_extraction"] = (include_extraction == "true")
            if text_schema != "(profile default)":
                payload["text_schema"] = text_schema
            if table_schema != "(profile default)":
                payload["table_schema"] = table_schema
            if scan_schema != "(profile default)":
                payload["scan_schema"] = scan_schema

        submit_and_render_job(
            cfg=cfg,
            path="/privacy/analyze",
            payload=payload,
            headers={},
            label=f"Privacy analyze (text) [{profile}]",
            auto_poll=auto_poll,
            persist_result_key=PERSIST_RESULT_KEY,
        )
        ran_submission = True

with tab_table:
    st.subheader("Table privacy analysis")
    table_id = st.text_input("table_id", value="customers")
    sampled_rows_raw = st.text_area("sampled_rows JSON", value=DEFAULT_TABLE_SAMPLE, height=200)

    sampled_rows: Any = None
    cols: list[str] = []
    json_error = None
    try:
        sampled_rows = json.loads(sampled_rows_raw)
        cols = _infer_table_columns(sampled_rows)
    except Exception as exc:  # noqa: BLE001
        json_error = str(exc)

    if json_error:
        st.error(f"Invalid JSON: {json_error}")
    else:
        st.caption(f"Detected columns: {cols or '(none)'}")

    scan_columns = st.multiselect("scan_columns (optional)", options=cols, default=[])
    st.caption("If left empty, privacy scan will automatically analyze all detected columns.")

    if st.button("Run privacy analysis (table)", type="primary"):
        if err:
            st.error(err)
            st.stop()
        if not isinstance(sampled_rows, list) or not all(isinstance(r, dict) for r in sampled_rows):
            st.error("sampled_rows must be a JSON array of objects.")
            st.stop()

        task: dict[str, Any] = {"kind": "table", "task_id": "tbl1", "table_id": table_id, "sampled_rows": sampled_rows}
        if scan_columns:
            task["scan_columns"] = scan_columns

        payload: dict[str, Any] = {
            "profile": profile,
            "policy_pack": policy_pack,
            "tasks": [task],
        }

        if advanced:
            if analysis_mode != "(profile default)":
                payload["analysis_mode"] = analysis_mode
            if include_extraction != "(profile default)":
                payload["include_extraction"] = (include_extraction == "true")
            if text_schema != "(profile default)":
                payload["text_schema"] = text_schema
            if table_schema != "(profile default)":
                payload["table_schema"] = table_schema
            if scan_schema != "(profile default)":
                payload["scan_schema"] = scan_schema

        submit_and_render_job(
            cfg=cfg,
            path="/privacy/analyze",
            payload=payload,
            headers={},
            label=f"Privacy analyze (table) [{profile}]",
            auto_poll=auto_poll,
            persist_result_key=PERSIST_RESULT_KEY,
        )
        ran_submission = True

if not ran_submission:
    saved_job = st.session_state.get(PERSIST_RESULT_KEY)
    if isinstance(saved_job, dict) and saved_job:
        st.markdown("---")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.subheader("Last privacy result (persisted)")
        with c2:
            if st.button("Clear saved result"):
                st.session_state.pop(PERSIST_RESULT_KEY, None)
                st.rerun()
        render_job(
            saved_job,
            show_raw=cfg.get("show_raw", False),
            show_legal_refs=cfg.get("show_legal_refs", True),
            show_legal_detail=cfg.get("show_legal_detail", True),
            show_debug=cfg.get("show_debug", False),
        )
