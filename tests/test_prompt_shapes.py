from __future__ import annotations

from moose.prompts import (
    build_cpa_prompt,
    build_table_prompt,
    build_tabular_cell_ner_prompt,
    build_text_ner_prompt,
    build_type_selection_prompt,
)
from moose.schema import get_schema_config


def test_text_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_text_ner_prompt(schema, [{"task_id": "t1", "text": "Alice"}], schema.load_type_ids())
    assert "task_id" not in prompt


def test_table_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_table_prompt(
        schema,
        [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}]}],
        schema.load_type_ids(),
    )
    assert "task_id" not in prompt


def test_table_prompt_includes_observed_columns_contract() -> None:
    schema = get_schema_config("coarse")
    prompt = build_table_prompt(
        schema,
        [
            {
                "task_id": "t1",
                "table_id": "tbl",
                "sampled_rows": [{"patient_name": "Alice", "mrn": "1"}],
            }
        ],
        schema.load_type_ids(),
    )
    assert '"observed_columns": ["patient_name", "mrn"]' in prompt
    assert "Column names in output MUST match observed_columns exactly" in prompt


def test_tabular_cell_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_tabular_cell_ner_prompt(
        schema,
        [{"task_id": "t1", "table_id": "tbl", "row_index": 0, "column": "name", "text": "Alice"}],
        schema.load_type_ids(),
    )
    assert "task_id" not in prompt


def test_type_selection_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("cpa")
    prompt = build_type_selection_prompt(
        schema,
        [
            {
                "task_id": "sel-1",
                "table_id": "tbl",
                "subject_column": "subject",
                "target_column": "target",
                "sampled_rows": [{"subject": "A", "target": "B"}],
            }
        ],
        ["CPA:NONE"],
        mode="cpa",
    )
    assert "task_id" not in prompt


def test_cpa_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("cpa")
    prompt = build_cpa_prompt(
        schema,
        {
            "task_id": "cpa-1",
            "table_id": "tbl",
            "sampled_rows": [{"subject": "A", "target": "B"}],
            "subject_column": "subject",
            "target_columns": ["target"],
        },
        ["CPA:NONE", "CPA:OTHER"],
    )
    assert "task_id" not in prompt


def test_dpv_pd_text_prompt_follows_contract_sections() -> None:
    schema = get_schema_config("dpv_pd")
    prompt = build_text_ner_prompt(
        schema,
        [{"task_id": "t1", "text": "We process email addresses and use an AI system."}],
        schema.load_type_ids(),
    )
    assert "TASK" in prompt
    assert "OUTPUT (API CONTRACT)" in prompt
    assert "ALLOWED LABEL SPACE" in prompt
    assert "SPAN RULES (NER BEST PRACTICES)" in prompt
    assert "SCORING RULES" in prompt
    assert "INPUT" in prompt
    assert "OUTPUT" in prompt
    assert "personal data categories (DPV-PD)" in prompt
    assert "AI-related concepts/activities (DPV-AI)" in prompt


def test_text_prompt_single_input_uses_single_object_contract() -> None:
    schema = get_schema_config("dpv_pd")
    prompt = build_text_ner_prompt(
        schema,
        [{"task_id": "t1", "text": "We process email addresses."}],
        schema.load_type_ids(),
    )
    assert "Input text JSON:" in prompt
    assert "Input texts JSON:" not in prompt
    assert "Top-level output MUST be a JSON object for this single input text." in prompt
    assert "Top-level output MUST be a JSON array" not in prompt


def test_text_prompt_multi_input_uses_array_contract() -> None:
    schema = get_schema_config("coarse")
    prompt = build_text_ner_prompt(
        schema,
        [{"task_id": "t1", "text": "Alice"}, {"task_id": "t2", "text": "Bob"}],
        schema.load_type_ids(),
    )
    assert "Input texts JSON:" in prompt
    assert "Top-level output MUST be a JSON array with length exactly equal to the number of input texts." in prompt


def test_prompts_say_speaker_labels_are_not_mentions() -> None:
    from moose.prompts import build_mention_prompt

    one_step = build_text_ner_prompt(get_schema_config("coarse"), [{"task_id": "t", "text": "Agent: Hi."}], ["PERSON"])
    two_step = build_mention_prompt(get_schema_config("dpv_privacy"), "Agent: Hi.")
    assert 'speaker labels that only mark who is talking (e.g. "Agent:"' in one_step
    assert 'speaker labels that only mark who is talking (e.g. "Agent:"' in two_step


def test_small_one_step_prompts_define_their_types_and_large_ones_do_not() -> None:
    from moose.schema import COARSE_TYPES

    coarse = get_schema_config("coarse")
    small = build_text_ner_prompt(coarse, [{"task_id": "t", "text": "Maria"}], list(COARSE_TYPES))
    assert "TYPE DEFINITIONS" in small and "- PERSON (person): A named person" in small
    dpv = get_schema_config("dpv")
    large = build_text_ner_prompt(dpv, [{"task_id": "t", "text": "x"}], dpv.load_type_ids())
    assert "TYPE DEFINITIONS" not in large
