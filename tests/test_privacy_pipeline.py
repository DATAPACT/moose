from __future__ import annotations

from types import SimpleNamespace

import pytest

import moose.privacy as privacy
from moose.privacy import (
    FindingCandidate,
    PolicyAction,
    PolicyPack,
    _build_finding_out,
    _resolve_scan_columns,
    build_table_facts,
)


def _pack_for_tests() -> PolicyPack:
    return PolicyPack.model_validate(
        {
            "name": "unit_pack",
            "actions": [{"action_id": "mask_logs", "label": "Mask in logs"}],
            "type_categories": {
                "dpv-pd:EmailAddress": ["DIRECT_IDENTIFIER"],
            },
            "rules": [],
        }
    )


def test_resolve_scan_columns_defaults_to_all_detected_columns() -> None:
    cols = _resolve_scan_columns(
        {
            "sampled_rows": [
                {"email": "a@example.test", "country": "IT"},
                {"email": "b@example.test", "postal_code": "00100"},
            ]
        }
    )
    assert cols == ["email", "country", "postal_code"]


def test_build_table_facts_includes_scan_entities_in_category_counts() -> None:
    pack = _pack_for_tests()
    facts = build_table_facts(
        pack=pack,
        task_id="t1",
        table_id="tbl",
        sampled_rows=[{"email": "a@example.test"}],
        columns=[],
        scan_result={
            "rows": [
                {
                    "cells": [
                        {
                            "column": "email",
                            "entities": [
                                {"type_id": "dpv-pd:EmailAddress", "confidence": 1.0, "text": "a@example.test"},
                                {"type_id": "dpv-pd:EmailAddress", "confidence": 1.0, "text": "b@example.test"},
                            ],
                        }
                    ]
                }
            ]
        },
    )
    assert facts.category_counts_any["DIRECT_IDENTIFIER"] == 2
    assert facts.category_counts_confident["DIRECT_IDENTIFIER"] == 2


def test_build_finding_out_marks_possible_violation_from_negative_context_flags() -> None:
    pack = _pack_for_tests()
    candidate = FindingCandidate(
        rule_id="direct_identifier_present",
        issue="Direct identifier detected",
        severity="high",
        default_actions=["mask_logs"],
        evidence=[],
    )
    finding = _build_finding_out(
        pack=pack,
        task={"task_id": "t1", "context": {"has_lawful_basis": False}},
        candidate=candidate,
        status="confirmed",
        severity="high",
        recommended_actions=["mask_logs"],
        rationale="test rationale",
    )

    assert finding.assessment == "possible_violation"
    assert finding.violation_reasons
    assert finding.mitigation_plan
    assert finding.mitigation_plan[0].action_id == "mask_logs"


@pytest.mark.asyncio
async def test_run_privacy_analyze_continues_when_initial_table_scan_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    pack = _pack_for_tests()

    monkeypatch.setattr(
        privacy,
        "resolve_privacy_defaults",
        lambda profile, overrides: {
            "_profile": "balanced",
            "_profile_config": {},
            "policy_pack": "unit_pack",
            "analysis_mode": "rules",
            "text_schema": "coarse",
            "table_schema": "coarse",
            "scan_schema": "coarse",
            "include_extraction": True,
        },
    )
    monkeypatch.setattr(privacy, "load_policy_pack", lambda name: pack)
    monkeypatch.setattr(privacy, "get_schema_config", lambda name: SimpleNamespace(supports_text=True, supports_table=True))

    async def _fake_run_text_ner(tasks, schema, llm_client, settings):  # noqa: ANN001
        return {"results": [], "warnings": []}

    async def _fake_run_table_annotate(tasks, schema, llm_client, settings):  # noqa: ANN001
        return {
            "results": [
                {
                    "task_id": "tbl1",
                    "table_id": "t1",
                    "columns": [{"column": "email", "type_id": "dpv-pd:EmailAddress", "confidence": 1.0}],
                }
            ],
            "warnings": [],
        }

    async def _fake_run_tabular_ner(tasks, schema, llm_client, settings):  # noqa: ANN001
        raise RuntimeError("scan parser failed")

    async def _fake_compute_findings_for_table_task(**kwargs):  # noqa: ANN003
        return []

    monkeypatch.setattr(privacy, "run_text_ner", _fake_run_text_ner)
    monkeypatch.setattr(privacy, "run_table_annotate", _fake_run_table_annotate)
    monkeypatch.setattr(privacy, "run_tabular_ner", _fake_run_tabular_ner)
    monkeypatch.setattr(privacy, "_compute_findings_for_table_task", _fake_compute_findings_for_table_task)

    out = await privacy.run_privacy_analyze(
        tasks=[
            {
                "kind": "table",
                "task_id": "tbl1",
                "table_id": "t1",
                "sampled_rows": [{"email": "a@example.test"}],
            }
        ],
        policy_pack="unit_pack",
        llm_client=object(),
        settings=SimpleNamespace(),
    )

    assert out["results"][0]["task_id"] == "tbl1"
    assert out["results"][0]["kind"] == "table"
    assert out["results"][0]["findings"] == []
    assert out["results"][0]["extraction"]["columns"][0]["column"] == "email"
    assert any(w.get("code") == "table_scan_failed" for w in out.get("warnings", []))
    reports = out.get("reports") or {}
    assert "human_readable" in reports
    assert "machine_readable" in reports
    machine = reports["machine_readable"]["content"]
    assert machine["schema_id"] == "moose.privacy.machine_report.v1"
    assert machine["summary"]["task_count"] == 1


def test_items_without_a_probability_are_kept_as_low_confidence() -> None:
    from moose.privacy import is_ignored, is_low_confidence

    pack = _pack_for_tests()
    assert not is_ignored(pack, None) and is_ignored(pack, 0.0)
    assert is_low_confidence(pack, None, [])[0]
