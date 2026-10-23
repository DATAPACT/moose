"""The dashboard's Text and Tables pages with two-step schemas, run by Streamlit's test runner against a fake API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import moose_ui.metadata as metadata  # noqa: E402
import moose_ui.submit as submit  # noqa: E402

SCHEMAS = [
    {"name": "coarse", "supports_text": True, "candidate_mode": "all"},
    {"name": "dpv_pd", "supports_text": True, "candidate_mode": "two_step", "linker": "system1+llm",
     "linkers": ["system1+llm", "system1", "llm"], "system1_validated_for": []},
    {"name": "dpv_privacy", "supports_text": True, "supports_table": True, "candidate_mode": "two_step",
     "linker": "system1+llm", "linkers": ["system1+llm", "system1", "llm"], "system1_validated_for": ["text"]},
]
RESULT = {
    "job_id": "j1",
    "status": "completed",
    "result": {"entities": [
        {"start": 0, "end": 12, "text": "Luca Bianchi", "type_id": "dpv-pd:Name", "confidence": 0.91,
         "calibrated": True, "decided_by": "system1+llm", "llm_suggestion": "Name", "category": "person name",
         "candidates": ["dpv-pd:Name", "dpv-pd:Identifier"],
         "ranking": [{"type_id": "dpv-pd:Name", "probability": 0.7}, {"type_id": "dpv-pd:Identifier", "probability": 0.2}]},
    ]},
}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    posted: list[dict] = []
    monkeypatch.setattr(metadata, "fetch_schemas", lambda base_url, api_key: SCHEMAS)
    monkeypatch.setattr(metadata, "fetch_models", lambda base_url, api_key: {"models": ["x/y"], "default_model": "x/y"})
    monkeypatch.setattr(submit, "api_post", lambda base_url, api_key, path, payload, headers: posted.append({"path": path, **payload}) or {"job_id": "j1"})
    monkeypatch.setattr(submit, "wait_for_job", lambda base_url, api_key, job_id: RESULT)
    def make(page: str) -> AppTest:
        at = AppTest.from_file(str(EXAMPLES / "pages" / page), default_timeout=30)
        at.session_state["api_key"] = "test-key"
        at.posted = posted
        return at

    return make


def test_two_step_schema_offers_the_server_linkers_and_sends_the_choice(app) -> None:
    page = app("1_Text.py")
    page.run()
    assert page.selectbox[0].value == "dpv_privacy"
    [linker] = page.radio
    assert linker.value == "system1+llm" and len(linker.options) == 3
    assert linker.options[0].startswith("System 1 + LLM: Laya decides")
    linker.set_value("system1").run()
    page.button[-1].click().run()  # Run NER

    assert page.posted[-1]["path"] == "/schemas/dpv_privacy/ner" and page.posted[-1]["linker"] == "system1"
    table = page.dataframe[0].value
    assert {"confidence", "calibrated", "decided_by", "llm_suggestion", "category", "ranking"} <= set(table.columns)
    assert table.iloc[0]["ranking"] == "dpv-pd:Name 0.70 · dpv-pd:Identifier 0.20"
    assert any("always from the System 1 model (Laya), never the LLM" in c.value for c in page.caption)


def test_one_step_schema_has_no_linker(app) -> None:
    page = app("1_Text.py")
    page.run()
    page.selectbox[0].set_value("coarse").run()
    assert not page.radio
    assert any("One-step schema" in c.value for c in page.caption)
    page.button[-1].click().run()
    assert "linker" not in page.posted[-1]


TABLE_RESULT = {
    "job_id": "j2",
    "status": "completed",
    "result": {"table_id": "table-1", "unannotated_columns": ["Plan"], "columns": [
        {"column": "Customer", "type_id": "dpv-pd:Name", "confidence": 0.9, "calibrated": False,
         "decided_by": "system1", "category": "customer names", "candidates": ["dpv-pd:Name"],
         "ranking": [{"type_id": "dpv-pd:Name", "probability": 0.9}]},
    ]},
}


def test_tables_page_types_columns_with_the_chosen_linker(app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "wait_for_job", lambda base_url, api_key, job_id: TABLE_RESULT)
    page = app("2_Tables.py")
    page.run()
    schema = next(s for s in page.selectbox if s.label == "Schema")
    assert schema.value == "dpv_privacy"
    [_, linker] = page.radio  # the first radio is the operation (column typing or CPA)
    linker.set_value("llm").run()
    next(b for b in page.button if b.label == "Run column typing").click().run()

    assert page.posted[-1]["path"] == "/schemas/dpv_privacy/tabular/annotate" and page.posted[-1]["linker"] == "llm"
    table = page.dataframe[0].value
    assert {"column", "type_id", "decided_by", "category", "ranking"} <= set(table.columns)
    assert "fine_type_id" not in table.columns  # empty STI columns are dropped
    assert any("No personal-data type for: Plan" in c.value for c in page.caption)


def test_unvalidated_system1_schema_says_its_probabilities_are_not_calibrated(app) -> None:
    page = app("1_Text.py")
    page.run()
    page.selectbox[0].set_value("dpv_pd").run()
    assert page.radio[0].value == "system1+llm"
    assert any("not been checked against labelled data" in c.value for c in page.caption)


def test_tables_page_says_when_table_probabilities_are_not_validated(app) -> None:
    page = app("2_Tables.py")
    page.run()
    # dpv_privacy is validated for text only.
    assert any("column annotations" not in c.value and "table annotations have not been checked" in c.value for c in page.caption)
