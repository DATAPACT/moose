from __future__ import annotations

import json
import subprocess
import sys

import pytest

from moose import Annotator, LLMClient

TEXT = "Agent: Thanks, Luca Bianchi."


class _LLM(LLMClient):
    """Answers each prompt with the next scripted response."""

    def __init__(self, *responses: object) -> None:
        super().__init__(provider="test", model="test-model")
        self.responses = [json.dumps(r) for r in responses]

    async def generate(self, prompt: str) -> str:
        return self.responses.pop(0)


def _script() -> _LLM:
    start = TEXT.index("Luca Bianchi")
    mention = {"start": start, "end": start + 12, "text": "Luca Bianchi", "category": "person name"}
    return _LLM({"mentions": [mention]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]})


def test_annotate_text_returns_the_api_result_shape() -> None:
    result = Annotator(_script()).annotate_text(TEXT)
    [entity] = result["entities"]
    assert "task_id" not in result and (entity["text"], entity["type_id"]) == ("Luca Bianchi", "dpv-pd:Name")
    assert entity["confidence"] is None  # no Laya in tests: never the LLM's number


def test_sync_calls_work_inside_a_running_event_loop() -> None:
    import asyncio

    async def notebook_cell() -> dict:
        return Annotator(_script()).annotate_text(TEXT)  # sync call while a loop runs, as in Jupyter

    assert asyncio.run(notebook_cell())["entities"][0]["type_id"] == "dpv-pd:Name"


@pytest.mark.asyncio
async def test_async_calls_and_usage() -> None:
    annotator = Annotator(_script())
    result = await annotator.aannotate_text(TEXT)
    assert result["entities"][0]["type_id"] == "dpv-pd:Name"
    assert annotator.last_usage is not None and annotator.last_usage.requested_model == "test-model"


def test_schemas_and_argument_checks() -> None:
    assert {"dpv_privacy", "dpv_pd", "dpv", "cpa"} <= set(Annotator.schemas())
    with pytest.raises(ValueError, match="either text or rows"):
        Annotator(_LLM()).analyze_privacy()


def test_user_schemas_go_to_the_user_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import moose.schema as schema_mod

    monkeypatch.setenv("MOOSE_USER_DATA_DIR", str(tmp_path / "mine"))
    assert schema_mod.user_data_dir() == tmp_path / "mine"
    monkeypatch.delenv("MOOSE_USER_DATA_DIR")
    monkeypatch.setattr(schema_mod, "DATA_DIR", tmp_path / "lib" / "site-packages" / "moose" / "data")
    assert schema_mod.user_data_dir().name == ".moose"  # never inside an installed package


def test_the_library_needs_no_server_dependencies() -> None:
    code = "import sys, moose; print(sorted(m for m in ('fastapi', 'motor', 'cryptography', 'jwt', 'uvicorn') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_sync_calls_share_one_event_loop() -> None:
    import asyncio

    class _LoopBound(_LLM):
        """Like an httpx client: fails if used from a different event loop than the first."""

        loop = None

        async def generate(self, prompt: str) -> str:
            self.loop = self.loop or asyncio.get_running_loop()
            assert self.loop is asyncio.get_running_loop()
            return await super().generate(prompt)

    start = TEXT.index("Luca Bianchi")
    mention = {"start": start, "end": start + 12, "text": "Luca Bianchi", "category": "person name"}
    llm = _LoopBound(*[{"mentions": [mention]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]}] * 2)
    annotator = Annotator(llm)
    assert annotator.annotate_text(TEXT)["entities"] and annotator.annotate_text(TEXT)["entities"]
