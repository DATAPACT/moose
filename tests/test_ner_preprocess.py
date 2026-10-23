from __future__ import annotations

from types import SimpleNamespace

import pytest

from moose.ner import (
    make_cell_task_id,
    run_table_annotate,
    run_tabular_ner,
    run_text_ner,
)


class _FakeLLM:
    def __init__(self, response: str | list[str]) -> None:
        if isinstance(response, list):
            if not response:
                raise ValueError("Fake LLM requires at least one response.")
            self._responses = response
        else:
            self._responses = [response]
        self._cursor = 0
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._cursor < len(self._responses):
            response = self._responses[self._cursor]
            self._cursor += 1
            return response
        return self._responses[-1]


@pytest.mark.asyncio
async def test_run_text_ner_uses_single_prompt_for_all_tasks():
    tasks = [{"task_id": "t1", "text": "Alice"}, {"task_id": "t2", "text": "Bob"}]
    llm = _FakeLLM('[{"task_id":"t1","entities":[]},{"task_id":"t2","entities":[]}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_text_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert [item["task_id"] for item in out["results"]] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_run_text_ner_prompt_keeps_raw_text_unchanged():
    tasks = [{"task_id": "t1", "text": "hello\x00world"}]
    llm = _FakeLLM('[{"task_id":"t1","entities":[]}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    await run_text_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert "\\u0000" in llm.calls[0]


@pytest.mark.asyncio
async def test_run_text_ner_accepts_single_object_output_shape():
    tasks = [{"task_id": "t1", "text": "Alice"}]
    llm = _FakeLLM('{"entities":[]}')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_text_ner(tasks, "coarse", llm, settings=settings)
    assert out["results"][0]["task_id"] == "t1"
    assert out["results"][0]["entities"] == []


@pytest.mark.asyncio
async def test_run_tabular_ner_prompt_keeps_raw_cell_text_unchanged():
    table_task_id = "t1"
    cell_task_id = make_cell_task_id(table_task_id, 0, "name")
    tasks = [
        {
            "task_id": table_task_id,
            "table_id": "tbl",
            "sampled_rows": [{"name": "  Alice\x00"}],
            "target_columns": ["name"],
            "strings_only": True,
            "skip_structured_literals": False,
        }
    ]
    llm = _FakeLLM(f'[{{"task_id":"{cell_task_id}","entities":[]}}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_tabular_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert '"text": "  Alice\\u0000"' in llm.calls[0]
    assert out["results"][0]["rows"][0]["cells"][0]["entities"] == []


@pytest.mark.asyncio
async def test_run_text_ner_does_not_return_distribution_field():
    tasks = [{"task_id": "t1", "text": "Alice"}]
    llm = _FakeLLM(
        '[{"task_id":"t1","entities":[{"start":0,"end":5,"text":"Alice","scores":{"PERSON":1.0,"ORGANIZATION":0.0,"LOCATION":0.0,"EVENT":0.0,"WORK":0.0,"PRODUCT":0.0,"CONCEPT":0.0,"MISC":0.0}}]}]'
    )
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_text_ner(tasks, "coarse", llm, settings=settings)
    entity = out["results"][0]["entities"][0]
    assert entity["type_id"] == "PERSON"
    assert "distribution" not in entity


@pytest.mark.asyncio
async def test_run_table_annotate_does_not_return_distribution_field():
    tasks = [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}]}]
    llm = _FakeLLM(
        '[{"task_id":"t1","table_id":"tbl","columns":[{"column":"name","scores":{"PERSON":1.0,"ORGANIZATION":0.0,"LOCATION":0.0,"EVENT":0.0,"WORK":0.0,"PRODUCT":0.0,"CONCEPT":0.0,"MISC":0.0}}]}]'
    )
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_table_annotate(tasks, "coarse", llm, settings=settings)
    column = out["results"][0]["columns"][0]
    assert column["type_id"] == "PERSON"
    assert "distribution" not in column


@pytest.mark.asyncio
async def test_run_table_annotate_sti_returns_coarse_and_fine_for_ne_columns_only():
    tasks = [
        {
            "task_id": "t1",
            "table_id": "tbl",
            "sampled_rows": [{"name": "Alice", "movie": "Inception", "email": "alice@example.com"}],
        }
    ]
    llm = _FakeLLM(
        [
            '[{"task_id":"t1","table_id":"tbl","columns":[{"column":"name","scores":{"NE:PERSON":1.0}},{"column":"movie","scores":{"NE:OTHER":1.0}},{"column":"email","scores":{"ext:email":1.0}}]}]',
            '[{"task_id":"t1","table_id":"tbl","columns":[{"column":"name","scores":{"PERSON":1.0}}]}]',
            '[{"task_id":"t1","table_id":"tbl","columns":[{"column":"movie","scores":{"FILM":1.0}}]}]',
        ]
    )
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)

    out = await run_table_annotate(tasks, "sti", llm, settings=settings)

    by_col = {c["column"]: c for c in out["results"][0]["columns"]}
    assert by_col["name"]["coarse_type_id"] == "NE:PERSON"
    assert by_col["name"]["fine_type_id"] == "PERSON"
    # The LLM's numbers are never used as probabilities; without the System 1 model there is none.
    assert "fine_confidence" not in by_col["name"] and by_col["name"]["confidence"] is None
    assert by_col["movie"]["coarse_type_id"] == "NE:OTHER"
    assert by_col["movie"]["fine_type_id"] == "FILM"
    assert by_col["email"]["coarse_type_id"] == "LIT:STRING"
    assert "fine_type_id" not in by_col["email"]


@pytest.mark.asyncio
async def test_run_table_annotate_sti_keeps_base_output_if_fine_enrichment_fails():
    tasks = [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}]}]
    llm = _FakeLLM(
        [
            '[{"task_id":"t1","table_id":"tbl","columns":[{"column":"name","scores":{"NE:PERSON":1.0}}]}]',
            "not-json",
        ]
    )
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)

    out = await run_table_annotate(tasks, "sti", llm, settings=settings)

    column = out["results"][0]["columns"][0]
    assert column["type_id"] == "NE:PERSON"
    assert column["coarse_type_id"] == "NE:PERSON"
    assert "fine_type_id" not in column
