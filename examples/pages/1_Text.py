from __future__ import annotations

import streamlit as st

from moose_ui.config import sidebar, validate_common
from moose_ui.metadata import clear_metadata_caches, fetch_schemas, schemas_supporting
from moose_ui.samples import DEFAULT_TEXT_SAMPLE
from moose_ui.submit import submit_and_render_job

st.title("Text")

cfg = sidebar()
err = validate_common(cfg)
if err:
    st.warning(err)

if not cfg.get("api_key"):
    st.info("Enter your Moose API key in the sidebar.")
    st.stop()

if st.button("Refresh schema list", key="text_refresh_schemas"):
    clear_metadata_caches()
    st.rerun()

# Metadata
try:
    schemas = fetch_schemas(cfg["base_url"], cfg["api_key"])
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch schemas: {exc}")
    st.stop()

schema_names = schemas_supporting(schemas, text=True)
if not schema_names:
    st.error("No text-capable schemas returned from API.")
    st.stop()

# Prefer dpv_pd then dpv
preferred = "dpv_pd" if "dpv_pd" in schema_names else ("dpv" if "dpv" in schema_names else schema_names[0])
schema = st.selectbox("Schema", schema_names, index=schema_names.index(preferred))

raw_text = st.text_area("Text (one sentence per line)", value=DEFAULT_TEXT_SAMPLE, height=160)

auto_poll = st.checkbox("Auto-poll", value=cfg["auto_poll_default"])

if st.button("Run NER", type="primary"):
    if err:
        st.error(err)
        st.stop()

    text = raw_text.strip()
    if not text:
        st.error("Provide non-empty text.")
        st.stop()

    payload = {
        "text": text,
    }

    submit_and_render_job(
        cfg=cfg,
        path=f"/schemas/{schema}/ner",
        payload=payload,
        headers={},
        label=f"Text NER ({schema})",
        auto_poll=auto_poll,
    )
