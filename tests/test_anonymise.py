from __future__ import annotations

import json

import pytest

from moose.anonymise import CATEGORIES, _parse_decisions, anonymise, build_identifier_mention_prompt, find_identifiers

TEXT = "Mr John Smith, a British national born in 1961, worked for Acme Ltd in Leeds until 3 May 1994."


class _ScriptedLLM:
    def __init__(self, *responses: object) -> None:
        self.responses = [json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_the_prompt_lists_every_category_and_asks_for_text_not_offsets() -> None:
    prompt = build_identifier_mention_prompt(TEXT)
    assert all(f"- {name}: " in prompt for name in CATEGORIES)
    assert "whether or not it" in prompt and "do not give character offsets" in prompt
    assert '"five years"' in prompt and '"Mr John Smith"' in prompt
    assert json.dumps(TEXT) in prompt


@pytest.mark.asyncio
async def test_mentions_are_placed_in_the_text_and_unknown_categories_become_misc() -> None:
    llm = _ScriptedLLM({"mentions": [
        {"text": "Mr John Smith", "category": "PERSON"},
        {"text": "British", "category": "dem"},
        {"text": "1961", "category": "DATETIME"},
        {"text": "Acme Ltd", "category": "ORG"},
        {"text": "Leeds", "category": "LOC"},
        {"text": "3 May 1994", "category": "DATE"},
    ]})
    found = await find_identifiers(TEXT, llm)
    assert [(m["text"], m["category"]) for m in found] == [
        ("Mr John Smith", "PERSON"), ("British", "DEM"), ("1961", "DATETIME"),
        ("Acme Ltd", "ORG"), ("Leeds", "LOC"), ("3 May 1994", "MISC"),
    ]
    assert all(TEXT[m["start"] : m["end"]] == m["text"] for m in found) and len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_each_entity_is_decided_once_for_all_its_mentions_and_masked_with_one_placeholder() -> None:
    text = "Mr Smith met Mr Jones in Leeds. In 1994 Mr Smith left the Home Office and Leeds."
    llm = _ScriptedLLM(
        {"mentions": [
            {"text": "Mr Smith", "category": "PERSON"}, {"text": "Mr Jones", "category": "PERSON"},
            {"text": "Leeds", "category": "LOC"}, {"text": "1994", "category": "DATETIME"},
            {"text": "Mr Smith", "category": "PERSON"}, {"text": "Home Office", "category": "ORG"},
            {"text": "Leeds", "category": "LOC"},
        ]},
        # Entities in order of first appearance: Mr Smith, Mr Jones, Leeds, 1994, Home Office.
        # 1994 is left out of the answer: it is masked, to be safe.
        {"decisions": [
            {"entity": 0, "identifier_type": "DIRECT"}, {"entity": 1, "identifier_type": "quasi"},
            {"entity": 2, "identifier_type": "QUASI"}, {"entity": 4, "identifier_type": "NO_MASK"},
        ]},
    )
    result = await anonymise(text, llm, protect="Mr Smith")

    assert result["masked_text"] == (
        "[PERSON_1] met [PERSON_2] in [LOC_1]. In [DATETIME_1] [PERSON_1] left the Home Office and [LOC_1]."
    )
    kinds = [(m["text"], m["identifier_type"], m["entity"]) for m in result["entities"]]
    assert kinds[0] == ("Mr Smith", "DIRECT", 0) and kinds[4] == ("Mr Smith", "DIRECT", 0)
    assert ("Home Office", "NO_MASK", 4) in kinds and ("1994", "QUASI", 3) in kinds
    assert all(m["confidence"] is None and m["calibrated"] is False for m in result["entities"])
    # The decision call sees each entity once, the person to protect, and the whole text.
    decision_prompt = llm.prompts[1]
    assert decision_prompt.count('"text": "Mr Smith"') == 1 and '"mentions": 2' in decision_prompt
    assert "so that Mr Smith cannot be identified" in decision_prompt and json.dumps(text) in decision_prompt


@pytest.mark.asyncio
async def test_with_laya_each_entity_gets_the_fitted_confidence_of_its_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import moose.anonymise as anon

    calibration = tmp_path / "laya_calibration.json"
    calibration.write_text(json.dumps({
        "model": "fake", "weights": {"PERSON:mask": 2.0, "ORG:leave": 0.0}, "laya_weight": 1.0, "validated_for": ["text"],
    }))
    asked: list[tuple[str, dict]] = []

    class _Laya:
        async def ask(self, inputs):
            asked.extend(inputs)
            return [{"yes": 0.5, "no": 0.5} for _ in inputs]

    async def scorer(model: str) -> _Laya:
        return _Laya()

    monkeypatch.setattr(anon, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(anon, "system1_ready", lambda cal: True)
    monkeypatch.setattr(anon, "get_scorer", scorer)
    llm = _ScriptedLLM(
        {"mentions": [{"text": "Mr Smith", "category": "PERSON"}, {"text": "Home Office", "category": "ORG"},
                      {"text": "Mr Smith", "category": "PERSON"}]},
        {"decisions": [{"entity": 0, "identifier_type": "DIRECT"}, {"entity": 1, "identifier_type": "NO_MASK"}]},
    )
    result = await anonymise("Mr Smith left the Home Office. Mr Smith", llm, protect="Mr Smith")

    # Laya is asked once per entity, on its first mention; P(yes) 0.5 adds nothing, so the
    # confidence is the weight of the category and decision: sigmoid(2) and sigmoid(0).
    assert len(asked) == 2 and "protect Mr Smith" in asked[0][1]["mask"]["instructions"]
    conf = {(m["text"], m["start"]): (m["confidence"], m["calibrated"]) for m in result["entities"]}
    assert conf[("Mr Smith", 0)] == conf[("Mr Smith", 31)] == (0.8808, True)
    assert conf[("Home Office", 18)] == (0.5, True)


def test_decisions_must_name_listed_entities_and_known_types() -> None:
    assert _parse_decisions('{"decisions": [{"entity": 1, "identifier_type": "no_mask"}]}', 2) == {1: "NO_MASK"}
    with pytest.raises(ValueError, match="listed entity"):
        _parse_decisions('{"decisions": [{"entity": 2, "identifier_type": "QUASI"}]}', 2)
    with pytest.raises(ValueError, match="must be one of"):
        _parse_decisions('{"decisions": [{"entity": 0, "identifier_type": "MAYBE"}]}', 2)


@pytest.mark.asyncio
async def test_the_api_worker_runs_an_anonymise_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from moose.config import Settings
    from moose.llm import OpenAICompatibleClient
    from moose_api.queue import InMemoryJobStore, InMemoryQueue, JobRecord, WorkerPool
    from moose_api.vault_store import ProviderRoute

    answers = [
        {"mentions": [{"text": "Mr Smith", "category": "PERSON"}]},
        {"decisions": [{"entity": 0, "identifier_type": "DIRECT"}]},
    ]

    async def fake_generate(self, prompt: str) -> str:
        assert ("so that the applicant cannot be identified" in prompt) == (len(answers) == 1)
        return json.dumps(answers.pop(0))

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)
    store = InMemoryJobStore()
    pool = WorkerPool(InMemoryQueue(10), store, None, Settings(_env_file=None, MOOSE_WORKER_COUNT=0, MOOSE_MAX_RETRIES=0))
    await store.put_job(JobRecord(
        job_id="a1", endpoint_type="anonymise",
        payload={"text": "Mr Smith appealed.", "protect": "the applicant", "llm": {"model": "x/y", "key_id": "k1"}},
        status="queued", created_at="now", updated_at="now", retries=0, client="user:u1",
    ))
    route = ProviderRoute(provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key="sk-test", secret_id="k1")
    await pool._process_job("a1", route)

    job = await store.get_job("a1")
    assert job.status == "completed", job.error
    assert job.result["masked_text"] == "[PERSON_1] appealed." and job.result["entities"][0]["identifier_type"] == "DIRECT"
