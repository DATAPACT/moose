from __future__ import annotations

import pytest
from pydantic import ValidationError

from moose_api.main import (
    SchemaCPARequest,
    SchemaNERRequest,
    SchemaTabularRequest,
)


def test_schema_ner_accepts_single_payload_shape() -> None:
    request = SchemaNERRequest.model_validate({"text": "Alice works at Moose."})
    assert request.text == "Alice works at Moose."


def test_schema_ner_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaNERRequest.model_validate(
            {
                "text": "Alice works at Moose.",
                "tasks": [{"task_id": "t1", "text": "Alice works at Moose."}],
            }
        )


def test_schema_ner_rejects_removed_debug_flags() -> None:
    with pytest.raises(ValidationError):
        SchemaNERRequest.model_validate(
            {
                "text": "Alice works at Moose.",
                "include_scores": True,
                "strict_offsets": True,
            }
        )


def test_schema_tabular_accepts_single_payload_shape() -> None:
    request = SchemaTabularRequest.model_validate(
        {"sampled_rows": [{"name": "Alice"}]}
    )
    assert request.sampled_rows == [{"name": "Alice"}]


def test_schema_tabular_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaTabularRequest.model_validate(
            {
                "sampled_rows": [{"name": "Alice"}],
                "tasks": [{"task_id": "t1", "sampled_rows": [{"name": "Alice"}]}],
            }
        )

def test_schema_cpa_accepts_single_payload_shape() -> None:
    request = SchemaCPARequest.model_validate(
        {
            "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
            "subject_column": "subject",
        }
    )
    assert request.subject_column == "subject"


def test_schema_cpa_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaCPARequest.model_validate(
            {
                "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
                "subject_column": "subject",
                "tasks": [
                    {
                        "task_id": "t1",
                        "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
                        "subject_column": "subject",
                    }
                ],
            }
        )


def test_llm_model_is_optional_and_strict() -> None:
    assert SchemaNERRequest.model_validate({"text": "Alice works at Moose."}).llm.model is None
    chosen = SchemaNERRequest.model_validate({"text": "Alice works at Moose.", "llm": {"model": "moose-strong"}})
    assert chosen.llm.model == "moose-strong"
    # Provider settings or keys cannot be passed through requests.
    with pytest.raises(ValidationError):
        SchemaNERRequest.model_validate({"text": "Alice", "llm": {"model": "m", "provider": "openrouter"}})
