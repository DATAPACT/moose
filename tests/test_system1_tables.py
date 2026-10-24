from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from moose.cpa import run_table_cpa
from moose.ner import run_tabular_ner
from moose.schema import get_schema_config


class _LLM:
    def __init__(self, response: object) -> None:
        self.response, self.prompts = json.dumps(response), []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class _Laya:
    """Answers every yes/no question with the same P(yes), and records the questions."""

    def __init__(self, p_yes: float) -> None:
        self.p_yes, self.asked = p_yes, []

    async def verify(self, context, items, records, prompt=None):
        self.asked += [(item["text"], record["id"], prompt.verify_question) for item, record in zip(items, records)]
        return [self.p_yes] * len(items)


def _use(monkeypatch: pytest.MonkeyPatch, laya: _Laya) -> None:
    import moose.linkers as linkers

    async def get(model):
        return laya

    monkeypatch.setattr(linkers, "laya_available", lambda: True)
    monkeypatch.setattr(linkers, "get_scorer", get)


SETTINGS = SimpleNamespace(MOOSE_MAX_RETRIES=0, MOOSE_MAX_CHARS_PER_PROMPT=100_000)


@pytest.mark.asyncio
async def test_cell_entities_take_their_probability_from_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    laya = _Laya(0.8)
    _use(monkeypatch, laya)
    llm = _LLM([{"row_index": 0, "column": "name", "entities": [{"start": 0, "end": 5, "text": "Alice", "type_id": "PERSON"}]}])
    tasks = [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}], "target_columns": ["name"]}]
    out = await run_tabular_ner(tasks, "coarse", llm, settings=SETTINGS)
    [entity] = out["results"][0]["rows"][0]["cells"][0]["entities"]
    assert (entity["type_id"], entity["confidence"], entity["calibrated"], entity["decided_by"]) == ("PERSON", 0.8, False, "llm")
    assert laya.asked == [("Alice", "PERSON", "Is the mention {label}?")]
    assert '"scores"' not in llm.prompts[0] and "Do not give scores" in llm.prompts[0]


@pytest.mark.asyncio
async def test_cell_entities_have_no_probability_without_laya() -> None:
    llm = _LLM([{"row_index": 0, "column": "name", "entities": [{"start": 0, "end": 5, "text": "Alice", "type_id": "PERSON"}]}])
    tasks = [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}], "target_columns": ["name"]}]
    out = await run_tabular_ner(tasks, "coarse", llm, settings=SETTINGS)
    [entity] = out["results"][0]["rows"][0]["cells"][0]["entities"]
    assert (entity["type_id"], entity["confidence"]) == ("PERSON", None)  # never the LLM's number


@pytest.mark.asyncio
async def test_column_relationships_take_their_probability_from_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    laya = _Laya(0.6)
    _use(monkeypatch, laya)
    relation = next(r for r in get_schema_config("cpa").load_type_ids() if not r.endswith(("NONE", "OTHER")))
    llm = _LLM([{"table_id": "tbl", "subject_column": "name", "relationships": [{"target_column": "city", "relation_id": relation}]}])
    tasks = [{"task_id": "t1", "table_id": "tbl", "subject_column": "name",
              "sampled_rows": [{"name": "Alice", "city": "Oslo"}, {"name": "Bob", "city": "Milan"}]}]
    out = await run_table_cpa(tasks, "cpa", llm, settings=SETTINGS)
    [rel] = out["results"][0]["relationships"]
    assert (rel["relation_id"], rel["confidence"], rel["calibrated"], rel["decided_by"]) == (relation, 0.6, False, "llm")
    assert [(text, rid) for text, rid, _ in laya.asked] == [("name -> city", relation)]
    assert '"scores"' not in llm.prompts[0] and '"relation_id"' in llm.prompts[0]


def test_cell_outputs_without_table_id_match_by_row_and_column() -> None:
    from moose.ner import _coerce_cell_ner_output_with_task_ids

    tasks = [{"task_id": "a", "table_id": "t1", "row_index": 0, "column": "name", "text": "x"},
             {"task_id": "b", "table_id": "t2", "row_index": 0, "column": "city", "text": "y"}]
    raw = json.dumps([{"row_index": 0, "column": "name", "entities": []}, {"row_index": 0, "column": "city", "entities": []}])
    assert [i["task_id"] for i in json.loads(_coerce_cell_ner_output_with_task_ids(raw, tasks))] == ["a", "b"]
    # Two tables with the same cell: without a table_id the answer is ambiguous.
    clash = [dict(tasks[0]), {**tasks[1], "column": "name"}]
    with pytest.raises(ValueError, match="Unknown cell"):
        _coerce_cell_ner_output_with_task_ids(json.dumps([{"row_index": 0, "column": "name", "entities": []}] * 2), clash)
