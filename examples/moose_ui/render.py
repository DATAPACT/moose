from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st


def _expander_json(title: str, data: Any, *, show: bool) -> None:
    if not show:
        return
    with st.expander(title, expanded=False):
        st.json(data)


def render_job(job: dict[str, Any], *, show_raw: bool, show_legal_refs: bool, show_legal_detail: bool, show_debug: bool) -> None:
    st.markdown("## Job")
    st.markdown(
        f"**job_id:** `{job.get('job_id')}`  \n"
        f"**status:** `{job.get('status')}`  \n"
        f"**created_at:** `{job.get('created_at', '')}`  \n"
        f"**updated_at:** `{job.get('updated_at', '')}`"
    )
    llm = job.get("llm") or {}
    if llm:
        used = ", ".join(llm.get("models_used") or []) or "none"
        cost = llm.get("cost_usd")
        st.caption(
            f"Model: {llm.get('requested_model')} → {used} · {llm.get('calls', 0)} calls · "
            f"{llm.get('prompt_tokens', 0)} + {llm.get('completion_tokens', 0)} tokens"
            + (f" · ${cost:.6f}" if cost is not None else "")
            + (f" · {llm['fallbacks']} fallback(s)" if llm.get("fallbacks") else "")
        )

    if job.get("status") == "failed":
        st.error(job.get("error") or "Job failed (no error message).")
        _expander_json("Raw job JSON", job, show=show_raw)
        return

    if job.get("status") != "completed":
        _expander_json("Raw job JSON", job, show=show_raw)
        return

    result = job.get("result") or {}
    _render_result(result, show_raw=show_raw, show_legal_refs=show_legal_refs, show_legal_detail=show_legal_detail, show_debug=show_debug)
    _expander_json("Raw job JSON", job, show=show_raw)


def _detect_kind(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if "schema" in result and "mapping_strategy" in result and "activated" in result:
        return "schema_ingest"
    if "entities" in result:
        return "text_ner"
    if "columns" in result:
        return "tabular_annotate"
    if "relationships" in result:
        return "cpa"
    if "rows" in result:
        return "tabular_ner"
    if "action_catalog" in result and "results" in result:
        rs = result.get("results")
        if isinstance(rs, list) and rs and isinstance(rs[0], dict) and "findings" in rs[0]:
            return "privacy"
    rs = result.get("results")
    if not isinstance(rs, list) or not rs:
        return "unknown"
    first = rs[0]
    if not isinstance(first, dict):
        return "unknown"
    if "entities" in first:
        return "text_ner"
    if "columns" in first:
        return "tabular_annotate"
    if "relationships" in first:
        return "cpa"
    if "rows" in first:
        return "tabular_ner"
    return "unknown"


def _render_result(result: dict[str, Any], *, show_raw: bool, show_legal_refs: bool, show_legal_detail: bool, show_debug: bool) -> None:
    kind = _detect_kind(result)
    if kind == "schema_ingest":
        render_schema_ingest_result(result, show_raw=show_raw)
        return
    if kind == "text_ner":
        render_text_ner_result(result, show_raw=show_raw)
        return
    if kind == "privacy":
        render_privacy_result(result, show_legal_refs=show_legal_refs, show_legal_detail=show_legal_detail, show_raw=show_raw)
        return
    if kind == "cpa":
        render_cpa_result(result, show_debug=show_debug, show_raw=show_raw)
        return
    if kind == "tabular_annotate":
        render_tabular_annotate_result(result, show_raw=show_raw)
        return
    if kind == "tabular_ner":
        render_tabular_ner_result(result, show_raw=show_raw)
        return

    # fallback: keep your existing simple behavior
    st.subheader("Result")
    st.json(result) if show_raw else st.write(result)


def render_schema_ingest_result(result: dict[str, Any], *, show_raw: bool) -> None:
    st.subheader("Schema ingest")
    st.markdown(
        f"**schema:** `{result.get('schema')}`  \n"
        f"**label:** `{result.get('label')}`  \n"
        f"**type_count:** `{result.get('type_count')}`  \n"
        f"**mapping_strategy:** `{result.get('mapping_strategy')}`  \n"
        f"**activated:** `{result.get('activated')}`"
    )
    if result.get("activated"):
        st.success("Schema is active and should now be visible in schema selectors.")
    source = result.get("source")
    if source:
        _expander_json("Source details", source, show=True)
    _expander_json("Raw schema ingest result", result, show=show_raw)


def _entity_type_group(type_id: str | None) -> str:
    t = str(type_id or "")
    tl = t.lower()
    if tl.startswith("dpv-pd:") or tl in {"ne:person", "ext:email", "ext:phone"}:
        return "direct_identifier"
    if "ip" in tl or tl in {"ext:ipv4", "ext:ipv6"}:
        return "network_identifier"
    if "recipient" in tl or "processor" in tl or tl.startswith("dpv:recipient") or tl.startswith("dpv:dataprocessor"):
        return "recipient"
    if "date" in tl or "age" in tl or "postal" in tl:
        return "quasi_identifier"
    return "other"


def _group_color(group: str) -> str:
    return {
        "direct_identifier": "#f8d7da",
        "network_identifier": "#fde2c8",
        "recipient": "#d9edf7",
        "quasi_identifier": "#fff3cd",
        "other": "#f1f3f5",
    }.get(group, "#f1f3f5")


def render_text_ner_result(result: dict[str, Any], *, show_raw: bool) -> None:
    st.subheader("Text entities")

    items = result.get("results")
    if not isinstance(items, list):
        items = [result]

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        entities = item.get("entities", []) or []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "text": ent.get("text"),
                    "type_id": ent.get("type_id"),
                    "type_group": _entity_type_group(ent.get("type_id")),
                    "coarse_type_id": ent.get("coarse_type_id"),
                    "confidence": ent.get("confidence"),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        if not df.empty and "type_group" in df.columns:
            st.dataframe(
                df.style.apply(
                    lambda r: [
                        # Fixed dark text: the theme's text colour is near-white in dark mode,
                        # which is unreadable on these pastel backgrounds.
                        f"background-color: {_group_color(str(r.get('type_group', 'other')))}; "
                        "color: #1f2328; font-weight: 600"
                        if c in {"type_id", "type_group"}
                        else ""
                        for c in df.columns
                    ],
                    axis=1,
                ),
                use_container_width=True,
            )
            st.caption(
                "Color legend: red=direct identifier, orange=network identifier, blue=recipient, yellow=quasi identifier."
            )
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No entities returned.")

    if result.get("warnings"):
        _expander_json("Warnings", result.get("warnings"), show=True)
    _expander_json("Raw text NER result", result, show=show_raw)


def render_cpa_result(result: dict[str, Any], *, show_debug: bool, show_raw: bool) -> None:
    st.subheader("CPA relationships")
    items = result.get("results", [])
    if not isinstance(items, list):
        items = [result]

    for item in items:
        st.markdown(
            f"**table_id:** `{item.get('table_id')}`  \n"
            f"**subject_column:** `{item.get('subject_column')}`"
        )
        rels = [
            {
                "target_column": r.get("target_column"),
                "relation_id": r.get("relation_id"),
                "confidence": r.get("confidence"),
            }
            for r in item.get("relationships", []) or []
        ]
        if rels:
            st.dataframe(rels, use_container_width=True)
        else:
            st.info("No relationships returned.")

        if show_debug and "debug" in item:
            _expander_json("Debug", item.get("debug"), show=True)

    _expander_json("Raw CPA result", result, show=show_raw)


def render_tabular_annotate_result(result: dict[str, Any], *, show_raw: bool) -> None:
    st.subheader("Column typing")
    items = result.get("results")
    if not isinstance(items, list):
        items = [result]

    for item in items:
        if not isinstance(item, dict):
            continue
        table_id = item.get("table_id")
        if isinstance(table_id, str) and table_id:
            st.markdown(f"**table_id:** `{table_id}`")

        columns = item.get("columns", []) or []
        rows: list[dict[str, Any]] = []
        for col in columns:
            if not isinstance(col, dict):
                continue
            rows.append(
                {
                    "column": col.get("column"),
                    "type_id": col.get("type_id"),
                    "coarse_type_id": col.get("coarse_type_id"),
                    "fine_type_id": col.get("fine_type_id"),
                    "confidence": col.get("confidence"),
                    "fine_confidence": col.get("fine_confidence"),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True)
            if any(row.get("fine_type_id") for row in rows):
                st.caption("For STI NE columns, both coarse_type_id and fine_type_id are shown.")
        else:
            st.info("No columns returned.")

    if result.get("warnings"):
        _expander_json("Warnings", result.get("warnings"), show=True)
    _expander_json("Raw tabular typing result", result, show=show_raw)


def render_tabular_ner_result(result: dict[str, Any], *, show_raw: bool) -> None:
    st.subheader("Cell NER")
    items = result.get("results")
    if not isinstance(items, list):
        items = [result]

    for item in items:
        if not isinstance(item, dict):
            continue
        table_id = item.get("table_id")
        if isinstance(table_id, str) and table_id:
            st.markdown(f"**table_id:** `{table_id}`")

        rows = item.get("rows", []) or []
        flat_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_index = row.get("row_index")
            for cell in row.get("cells", []) or []:
                if not isinstance(cell, dict):
                    continue
                column = cell.get("column")
                entities = cell.get("entities", []) or []
                if not entities:
                    continue
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    flat_rows.append(
                        {
                            "row_index": row_index,
                            "column": column,
                            "text": ent.get("text"),
                            "type_id": ent.get("type_id"),
                            "coarse_type_id": ent.get("coarse_type_id"),
                            "confidence": ent.get("confidence"),
                        }
                    )

        if flat_rows:
            st.dataframe(flat_rows, use_container_width=True)
        else:
            st.info("No entities returned for selected cells/columns.")

    if result.get("warnings"):
        _expander_json("Warnings", result.get("warnings"), show=True)
    _expander_json("Raw tabular NER result", result, show=show_raw)


def _render_privacy_extraction(task: dict[str, Any], *, show_raw: bool) -> None:
    extraction = task.get("extraction")
    if not isinstance(extraction, dict):
        return

    kind = task.get("kind")
    if kind == "text":
        ents = extraction.get("entities", []) or []
        if ents:
            rows: list[dict[str, Any]] = []
            for ent in ents:
                if not isinstance(ent, dict):
                    continue
                rows.append(
                    {
                        "text": ent.get("text"),
                        "type_id": ent.get("type_id"),
                        "confidence": ent.get("confidence"),
                        "start": ent.get("start"),
                        "end": ent.get("end"),
                    }
                )
            if rows:
                with st.expander(f"Extraction entities ({len(rows)})", expanded=False):
                    st.dataframe(rows, use_container_width=True)
    elif kind == "table":
        cols = extraction.get("columns", []) or []
        if cols:
            rows = []
            for col in cols:
                if not isinstance(col, dict):
                    continue
                rows.append(
                    {
                        "column": col.get("column"),
                        "type_id": col.get("type_id"),
                        "coarse_type_id": col.get("coarse_type_id"),
                        "fine_type_id": col.get("fine_type_id"),
                        "confidence": col.get("confidence"),
                    }
                )
            if rows:
                with st.expander(f"Extraction typed columns ({len(rows)})", expanded=False):
                    st.dataframe(rows, use_container_width=True)

    _expander_json("Extraction (raw)", extraction, show=show_raw)


def _render_privacy_reports(result: dict[str, Any]) -> None:
    reports = result.get("reports")
    if not isinstance(reports, dict):
        return

    human = reports.get("human_readable") if isinstance(reports.get("human_readable"), dict) else {}
    machine = reports.get("machine_readable") if isinstance(reports.get("machine_readable"), dict) else {}

    human_content = str(human.get("content") or "")
    machine_content = machine.get("content") if isinstance(machine.get("content"), dict) else {}

    if not human_content and not machine_content:
        return

    st.markdown("### Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download human report (.md)",
            data=human_content or "# Privacy Analysis Report\n",
            file_name="privacy_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download machine report (.json)",
            data=json.dumps(machine_content or {}, indent=2, ensure_ascii=True),
            file_name="privacy_report.json",
            mime="application/json",
            use_container_width=True,
        )

    if human_content:
        with st.expander("Human report preview", expanded=False):
            st.markdown(human_content)
    if machine_content:
        with st.expander("Machine report preview", expanded=False):
            st.json(machine_content)


def render_privacy_result(result: dict[str, Any], *, show_legal_refs: bool, show_legal_detail: bool, show_raw: bool) -> None:
    st.subheader("Privacy findings")
    st.caption(f"profile: {result.get('profile')} | policy_pack: {result.get('policy_pack')}")
    _render_privacy_reports(result)

    for task in result.get("results", []) or []:
        st.markdown(f"### {task.get('kind')} task `{task.get('task_id')}`")

        findings = task.get("findings", []) or []
        if not findings:
            st.info("No findings.")
            continue

        severity_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for f in findings:
            sev = str(f.get("severity") or "unknown")
            stat = str(f.get("status") or "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            status_counts[stat] = status_counts.get(stat, 0) + 1

        sev_summary = ", ".join(f"{k}:{v}" for k, v in severity_counts.items()) or "none"
        status_summary = ", ".join(f"{k}:{v}" for k, v in status_counts.items()) or "none"
        st.caption(f"{len(findings)} finding(s) | severity [{sev_summary}] | status [{status_summary}]")

        summary_rows = []
        for f in findings:
            row = {
                "severity": f.get("severity"),
                "status": f.get("status"),
                "assessment": f.get("assessment", "compliance_risk"),
                "rule_id": f.get("rule_id"),
                "issue": f.get("issue"),
                "actions": ", ".join(f.get("recommended_actions") or []),
            }
            if show_legal_refs:
                row["legal_ref_count"] = len(f.get("legal_refs") or [])
            summary_rows.append(row)

        st.dataframe(summary_rows, use_container_width=True)

        for idx, f in enumerate(findings, start=1):
            sev = str(f.get("severity") or "unknown").upper()
            stat = str(f.get("status") or "unknown")
            title = f"{idx}. [{sev}] [{stat}] {f.get('rule_id')} - {f.get('issue')}"
            with st.expander(title, expanded=False):
                st.caption(
                    f"assessment={f.get('assessment', 'compliance_risk')} | "
                    f"severity={f.get('severity')} | status={f.get('status')}"
                )
                st.write(f.get("rationale"))

                actions = f.get("recommended_actions") or []
                if actions:
                    st.write("Recommended actions:", ", ".join(str(a) for a in actions))
                if f.get("violation_reasons"):
                    st.write("Violation signals:", "; ".join(str(x) for x in (f.get("violation_reasons") or [])))

                mitigation = f.get("mitigation_plan") or []
                if mitigation:
                    rows = []
                    for step in mitigation:
                        if not isinstance(step, dict):
                            continue
                        rows.append(
                            {
                                "action_id": step.get("action_id"),
                                "label": step.get("label"),
                                "priority": step.get("priority"),
                                "targets": ", ".join(step.get("targets") or []),
                            }
                        )
                    if rows:
                        st.dataframe(rows, use_container_width=True)

                if f.get("evidence"):
                    _expander_json("Evidence", f.get("evidence"), show=True)

                if show_legal_refs:
                    if show_legal_detail:
                        details = f.get("legal_refs_detail") or []
                        if isinstance(details, list) and details:
                            detail_rows = []
                            for d in details:
                                if not isinstance(d, dict):
                                    continue
                                detail_rows.append(
                                    {
                                        "id": d.get("id"),
                                        "label": d.get("label"),
                                        "kind": d.get("kind"),
                                        "number": d.get("number"),
                                    }
                                )
                            if detail_rows:
                                st.dataframe(detail_rows, use_container_width=True)
                    else:
                        refs = f.get("legal_refs") or []
                        if refs:
                            st.write("Legal refs:", ", ".join(str(x) for x in refs))

        _render_privacy_extraction(task, show_raw=show_raw)

    if result.get("warnings"):
        _expander_json("Warnings", result.get("warnings"), show=True)

    _expander_json("Raw privacy result", result, show=show_raw)
