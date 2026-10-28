from __future__ import annotations

import json
import math

import pytest

from moose.candidates import candidate_index
from moose.ner import run_text_ner
from moose.schema import get_schema_config

TEXT = "Agent: Thanks, Luca Bianchi. Your customer ID is USR 661 204 and I have your email as luca@example.com."


class _ScriptedLLM:
    """Answers each prompt with the next scripted response, and records the prompts."""

    def __init__(self, *responses: object) -> None:
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _mention(text: str, category: str) -> dict:
    start = TEXT.index(text)
    return {"start": start, "end": start + len(text), "text": text, "category": category}


MENTIONS = {
    "mentions": [
        _mention("Luca Bianchi", "person name"),
        _mention("USR 661 204", "customer identifier"),
        _mention("luca@example.com", "email address"),
    ]
}


def test_privacy_schema_is_two_step_with_definitions() -> None:
    schema = get_schema_config("dpv_privacy")
    assert schema.candidate_mode == "two_step" and schema.candidate_k == 8
    records = {r["id"]: r for r in schema.load_type_records()}
    assert {"dpv-pd:Name", "dpv-pd:EmailAddress", "dpv:DataProcessor", "dpv:ThirdParty"} <= records.keys()
    assert records["dpv-pd:EmailAddress"]["definition"]


def test_candidates_come_from_labels_and_definitions() -> None:
    index = candidate_index(get_schema_config("dpv_privacy"))
    assert "dpv-pd:EmailAddress" in [c["id"] for c in index.top_k("email address luca@example.com", 8)]
    assert "dpv-pd:Name" in [c["id"] for c in index.top_k("person name Luca Bianchi", 8)]
    assert "dpv:DataProcessor" in [c["id"] for c in index.top_k("company processing data on our behalf", 8)]


def test_candidates_include_parents_and_children_of_the_best_hits() -> None:
    index = candidate_index(get_schema_config("dpv_privacy"))
    found = [c["id"] for c in index.top_k("medical condition asthma", 8)]
    # The description matches MedicalHealth best; its child Health is what usually fits.
    assert {"dpv-pd:MedicalHealth", "dpv-pd:Health"} <= set(found) and len(found) == len(set(found))


@pytest.mark.asyncio
async def test_two_step_finds_mentions_then_chooses_among_their_candidates() -> None:
    llm = _ScriptedLLM(
        MENTIONS,
        {"choices": [
            {"mention": 0, "type_id": "dpv-pd:Name", "confidence": 0.9},
            {"mention": 1, "type_id": "dpv-pd:Identifier", "confidence": 0.8},
            {"mention": 2, "type_id": "none", "confidence": 0.2},
        ]},
    )
    result = await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm)

    entities = result["results"][0]["entities"]
    assert [(e["text"], e["type_id"]) for e in entities] == [("Luca Bianchi", "dpv-pd:Name"), ("USR 661 204", "dpv-pd:Identifier")]
    # No System 1 model in this test: the LLM decided, and there is no probability (never the LLM's own).
    assert entities[0]["confidence"] is None and entities[1]["category"] == "customer identifier"
    # Up to candidate_k retrieved, plus as many parents and children of the best hits.
    assert "dpv-pd:Identifier" in entities[1]["candidates"] and len(entities[1]["candidates"]) <= 16
    # The first prompt lists no type ids; the second shows only each mention's candidates.
    assert "dpv-pd:" not in llm.prompts[0]
    assert "USR 661 204" in llm.prompts[1] and "dpv-pd:Identifier (Identifier)" in llm.prompts[1]
    assert len(llm.prompts[1]) < 8000


@pytest.mark.asyncio
async def test_choice_outside_a_mentions_candidates_is_retried() -> None:
    llm = _ScriptedLLM(
        {"mentions": [_mention("Luca Bianchi", "person name")]},
        {"choices": [{"mention": 0, "type_id": "dpv-pd:Health"}]},
        {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]},
    )
    result = await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm)
    assert [e["type_id"] for e in result["results"][0]["entities"]] == ["dpv-pd:Name"]
    assert "is not one of its candidates" in llm.prompts[2]


@pytest.mark.asyncio
async def test_mentions_not_in_the_text_are_dropped_and_offsets_repaired() -> None:
    wrong_offsets = {**_mention("Luca Bianchi", "person name"), "start": 0, "end": 3}
    invented = {"start": 0, "end": 4, "text": "Mario Rossi", "category": "person name"}
    llm = _ScriptedLLM({"mentions": [wrong_offsets, invented]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]})
    result = await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm)
    [entity] = result["results"][0]["entities"]
    assert TEXT[entity["start"]:entity["end"]] == "Luca Bianchi"


@pytest.mark.asyncio
async def test_no_mentions_needs_one_call() -> None:
    llm = _ScriptedLLM({"mentions": []})
    result = await run_text_ner([{"task_id": "t", "text": "Hello, how can I help?"}], "dpv_privacy", llm)
    assert result["results"][0]["entities"] == [] and len(llm.prompts) == 1


class _FakeLaya:
    """Probabilities by mention text, and by mention text plus the LLM's suggestion.
    Unlisted candidates share what is left. Records every call."""

    def __init__(self, plain: dict[str, dict[str, float]], hinted: dict[str, dict[str, float]] | None = None,
                 verdicts: dict[str, float] | None = None) -> None:
        self.plain, self.hinted, self.verdicts = plain, hinted or {}, verdicts or {}
        self.calls: list[tuple[str, str | None]] = []

    async def verify(self, context, items, records, prompt=None):
        """P(yes) by mention text; records a ("verify", label) call."""
        out = []
        for item, record in zip(items, records):
            self.calls.append((item["text"], f"verify {record.get('label') or record['id']}"))
            out.append(self.verdicts[item["text"]])
        return out

    async def score(self, context, items, suggestions=None, prompt=None):
        out = []
        for item, suggestion in zip(items, suggestions or [None] * len(items)):
            self.calls.append((item["text"], suggestion))
            given = (self.hinted if suggestion else self.plain).get(item["text"], {})
            ids = [c["id"] for c in item["candidates"]]
            rest = [i for i in ids if i not in given]
            left = max(1.0 - sum(given.values()), 0.0)
            out.append({**{i: left / len(rest) for i in rest}, **{i: p for i, p in given.items() if i in ids}})
        return out


def _use_laya(monkeypatch: pytest.MonkeyPatch, tmp_path, fake: _FakeLaya, consult_below: float = 0.6) -> None:
    import dataclasses

    import moose.linkers as linkers
    import moose.ner as ner_module

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"consult_below": consult_below}))
    schema = dataclasses.replace(get_schema_config("dpv_privacy"), linker_calibration=path)
    monkeypatch.setattr(ner_module, "get_schema_config", lambda name: schema)
    async def get_fake(model):
        return fake

    monkeypatch.setattr(linkers, "laya_available", lambda: True)
    monkeypatch.setattr(linkers, "get_scorer", get_fake)
    monkeypatch.setattr(ner_module, "get_scorer", get_fake)


TWO = {"mentions": MENTIONS["mentions"][:2]}


@pytest.mark.asyncio
async def test_system1_decides_alone_without_a_second_llm_call(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _use_laya(monkeypatch, tmp_path, _FakeLaya({"Luca Bianchi": {"dpv-pd:Name": 0.7}, "USR 661 204": {"dpv-pd:Identifier": 0.3}}))
    llm = _ScriptedLLM(TWO)
    result = await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm, linker="system1")

    entities = result["results"][0]["entities"]
    assert [(e["type_id"], e["confidence"], e["decided_by"], e["calibrated"]) for e in entities] == [
        ("dpv-pd:Name", 0.7, "system1", True),
        ("dpv-pd:Identifier", 0.3, "system1", True),
    ]
    assert entities[0]["ranking"][0] == {"type_id": "dpv-pd:Name", "probability": 0.7}
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_system1_consults_the_llm_only_when_unsure_and_decides_again(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = _FakeLaya(
        plain={"Luca Bianchi": {"dpv-pd:Name": 0.9}, "USR 661 204": {"dpv-pd:Reference": 0.4}},
        hinted={"USR 661 204": {"dpv-pd:Identifier": 0.8}},
    )
    _use_laya(monkeypatch, tmp_path, fake)
    llm = _ScriptedLLM(TWO, {"choices": [{"mention": 0, "type_id": "dpv-pd:Identifier", "confidence": 1.0}]})
    result = await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm)  # schema default: auto

    name, identifier = result["results"][0]["entities"]
    assert (name["decided_by"], name["confidence"]) == ("system1", 0.9)
    assert (identifier["type_id"], identifier["decided_by"], identifier["confidence"], identifier["llm_suggestion"]) == (
        "dpv-pd:Identifier", "system1+llm", 0.8, "Identifier",
    )
    # Only the unsure mention went to the LLM, and Laya saw the LLM's suggestion for it.
    assert "USR 661 204" in llm.prompts[1] and '"Luca Bianchi" (described' not in llm.prompts[1]
    assert ("USR 661 204", "Identifier") in fake.calls


@pytest.mark.asyncio
async def test_llm_linker_takes_its_probability_from_laya_not_the_llm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = _FakeLaya(plain={"Luca Bianchi": {"dpv-pd:Name": 0.4}}, verdicts={"Luca Bianchi": 0.7})
    _use_laya(monkeypatch, tmp_path, fake)
    llm = _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name", "confidence": 0.99}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm, linker="llm"))["results"][0]["entities"]
    # The LLM's 0.99 is ignored; Laya answers yes/no on the LLM's choice, and ranks the
    # candidates on its own (never told what the LLM chose). Never checked for llm: not calibrated.
    assert (entity["decided_by"], entity["confidence"], entity["calibrated"]) == ("llm", 0.7, False)
    assert {"type_id": "dpv-pd:Name", "probability": 0.4} in entity["ranking"]
    assert fake.calls == [("Luca Bianchi", None), ("Luca Bianchi", "verify Name")] and '"confidence"' not in llm.prompts[1]


@pytest.mark.asyncio
async def test_without_laya_the_llm_decides_with_uncalibrated_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    import moose.linkers as linkers

    monkeypatch.setattr(linkers, "laya_available", lambda: False)
    llm = _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name", "confidence": 0.99}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm))["results"][0]["entities"]
    assert (entity["confidence"], entity["calibrated"], entity["decided_by"]) == (None, False, "llm") and "ranking" not in entity


@pytest.mark.asyncio
async def test_linker_applies_only_to_two_step_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    import moose.ner as ner_module

    one_step = dataclasses.replace(get_schema_config("coarse"), candidate_mode="all")
    monkeypatch.setattr(ner_module, "get_schema_config", lambda name: one_step)
    with pytest.raises(ValueError, match="one step"):
        await run_text_ner([{"task_id": "t", "text": TEXT}], "coarse", _ScriptedLLM(), linker="system1")


@pytest.mark.asyncio
async def test_auto_schemas_use_two_steps_only_with_the_system1_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from moose.ner import two_step_active

    coarse = get_schema_config("coarse")
    assert coarse.candidate_mode == "auto"
    assert not two_step_active(coarse)  # tests run without Laya
    assert two_step_active(coarse, linker="llm")  # naming a linker asks for two steps
    import moose.linkers as linkers

    monkeypatch.setattr(linkers, "laya_available", lambda: True)
    assert two_step_active(coarse)
    # Every coarse type is a candidate (8 types), each with a definition.
    index_records = coarse.load_type_records()
    assert len(index_records) == 8 and all(r["definition"] for r in index_records)


ROWS = [
    {"Customer": "Luca Bianchi", "E-mail": "luca@example.com", "Plan": "Gold"},
    {"Customer": "Maria Rossi", "E-mail": "maria@example.com", "Plan": "Silver"},
]


@pytest.mark.asyncio
async def test_table_columns_are_described_then_typed_by_system1(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from moose.ner import run_table_annotate

    fake = _FakeLaya(
        plain={"Customer: Luca Bianchi | Maria Rossi": {"dpv-pd:Name": 0.9},
               "E-mail: luca@example.com | maria@example.com": {"dpv-pd:EmailAddress": 0.45}},
        hinted={"E-mail: luca@example.com | maria@example.com": {"dpv-pd:EmailAddress": 0.8}},
    )
    _use_laya(monkeypatch, tmp_path, fake)
    llm = _ScriptedLLM(
        {"columns": [
            {"column": "Customer", "relevant": True, "category": "customer names"},
            {"column": "e-mail", "relevant": True, "category": "email addresses"},  # case differs: still matched
            {"column": "Plan", "relevant": False, "category": ""},
        ]},
        {"choices": [{"column": 0, "type_id": "dpv-pd:EmailAddress", "confidence": 0.9}]},
    )
    result = await run_table_annotate([{"task_id": "t", "table_id": "tbl", "sampled_rows": ROWS}], "dpv_privacy", llm)

    [table] = result["results"]
    assert [(c["column"], c["type_id"], c["decided_by"], c["confidence"]) for c in table["columns"]] == [
        ("Customer", "dpv-pd:Name", "system1", 0.9),
        ("E-mail", "dpv-pd:EmailAddress", "system1+llm", 0.8),
    ]
    assert table["unannotated_columns"] == ["Plan"]
    # Laya's probabilities are validated on text only, so table ones are not marked calibrated.
    assert not any(c["calibrated"] for c in table["columns"])
    assert "dpv-pd:" not in llm.prompts[0] and '"Plan": Gold | Silver' in llm.prompts[0]
    assert "[0] E-mail: luca@example.com" in llm.prompts[1] and "Customer: Luca" not in llm.prompts[1]


@pytest.mark.asyncio
async def test_one_step_table_schema_rejects_a_linker() -> None:
    from moose.ner import run_table_annotate

    with pytest.raises(ValueError, match="one step"):
        await run_table_annotate([{"task_id": "t", "table_id": "x", "sampled_rows": ROWS}], "sti", _ScriptedLLM(), linker="system1")


def test_mentions_of_a_kind_of_data_count_except_where_a_schema_says_otherwise() -> None:
    from moose.prompts import build_mention_prompt

    general = build_mention_prompt(get_schema_config("dpv_pd"), "We collect email addresses.")
    benchmark = build_mention_prompt(get_schema_config("dpv_privacy"), "We collect email addresses.")
    named = build_mention_prompt(get_schema_config("coarse"), "Several companies opened offices.")
    assert 'mentions of a kind of data (e.g. "email addresses")' in general
    assert "only name a category without giving a value" in benchmark and "mentions of a kind of data" not in benchmark
    assert "named entities only" in named and "mentions of a kind" not in named


def test_two_letter_words_like_ip_find_their_type() -> None:
    index = candidate_index(get_schema_config("dpv_pd"))
    assert index.top_k("IP addresses", 8)[0]["id"] == "dpv-pd:IPAddress"


@pytest.mark.asyncio
async def test_unsure_system1_and_a_rejecting_llm_drop_the_item(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = _FakeLaya(plain={"Luca Bianchi": {"dpv-pd:Name": 0.9}, "USR 661 204": {"dpv-pd:Reference": 0.3}})
    _use_laya(monkeypatch, tmp_path, fake)
    llm = _ScriptedLLM(TWO, {"choices": [{"mention": 0, "type_id": "none"}]})
    entities = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm))["results"][0]["entities"]
    assert [e["text"] for e in entities] == ["Luca Bianchi"]



def _failing_laya(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Laya installed, but its model cannot be loaded; returns the models it tried."""
    import moose.linkers as linkers

    tried: list[str] = []

    class _Broken:
        def __init__(self, model):
            tried.append(model)
            raise OSError("no route to huggingface.co")

    monkeypatch.setattr(linkers, "laya_available", lambda: True)
    monkeypatch.setattr(linkers, "LayaScorer", _Broken)
    monkeypatch.setattr(linkers, "_SCORERS", {})
    monkeypatch.setattr(linkers, "_FAILURES", {})
    return tried


@pytest.mark.asyncio
async def test_auto_schema_falls_back_to_one_step_when_laya_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import moose.linkers as linkers

    tried = _failing_laya(monkeypatch)
    from moose.schema import COARSE_TYPES

    # coarse scores densely: the one-step answer scores every type.
    scores = {t: (1.0 if t == "PERSON" else 0.0) for t in COARSE_TYPES}
    one_step = {"entities": [{"start": 0, "end": 5, "text": "Maria", "scores": scores}]}
    llm = _ScriptedLLM(one_step)
    result = await run_text_ner([{"task_id": "t", "text": "Maria lives in Milan."}], "coarse", llm)

    assert [e["type_id"] for e in result["results"][0]["entities"]] == ["PERSON"]
    assert len(llm.prompts) == 1 and "TASK" in llm.prompts[0]
    # The failure is remembered: no new attempt for a while, and auto schemas stay one-step.
    assert not linkers.system1_ready(linkers.Calibration.load(get_schema_config("coarse").linker_calibration))
    await run_text_ner([{"task_id": "t", "text": "Maria lives in Milan."}], "coarse", _ScriptedLLM(one_step))
    assert len(tried) == 1


@pytest.mark.asyncio
async def test_two_step_schema_falls_back_to_the_llm_and_named_linkers_fail_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    from moose.linkers import LinkerUnavailable

    _failing_laya(monkeypatch)
    llm = _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm))["results"][0]["entities"]
    assert (entity["type_id"], entity["decided_by"], entity["calibrated"]) == ("dpv-pd:Name", "llm", False)

    with pytest.raises(LinkerUnavailable, match="could not be loaded"):
        await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}), linker="system1")


@pytest.mark.asyncio
async def test_llm_linker_still_works_when_laya_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_laya(monkeypatch)
    llm = _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name", "confidence": 0.9}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm, linker="llm"))["results"][0]["entities"]
    assert (entity["type_id"], entity["confidence"], entity["calibrated"]) == ("dpv-pd:Name", None, False)


def test_dpv_annotates_in_two_steps_with_the_llm_deciding() -> None:
    from moose.ner import two_step_active

    dpv = get_schema_config("dpv")
    # Laya chooses badly among DPV's 8,000+ concepts zero-shot (DPVCallBench dev: 0.55 against
    # 0.77 for the LLM), but two steps beat one step even with the LLM deciding.
    assert (dpv.candidate_mode, dpv.linker) == ("two_step", "llm") and two_step_active(dpv)


def test_a_calibration_can_name_its_own_model_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from moose.linkers import Calibration

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"model": "org/fine-tuned", "model_env": "MOOSE_LAYA_DPV_MODEL"}))
    monkeypatch.setenv("MOOSE_LAYA_MODEL", "/base")
    assert Calibration.load(path).model == "org/fine-tuned"  # the general override does not apply
    monkeypatch.setenv("MOOSE_LAYA_DPV_MODEL", "/local/fine-tuned")
    assert Calibration.load(path).model == "/local/fine-tuned"
    # Schemas without their own variable follow MOOSE_LAYA_MODEL.
    assert Calibration.load(get_schema_config("dpv_privacy").linker_calibration).model == "/base"


@pytest.mark.asyncio
async def test_the_llm_choice_is_in_the_ranking_even_below_the_top_3(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The LLM picks Laya's fourth type: it joins the ranking, with Laya's own probability.
    found = [c["id"] for c in candidate_index(get_schema_config("dpv_privacy")).top_k("person name Luca Bianchi", 8)]
    others = [c for c in found if c != "dpv-pd:Name"][:3]
    hinted = {others[0]: 0.4, others[1]: 0.3, others[2]: 0.2, "dpv-pd:Name": 0.1}
    fake = _FakeLaya(plain={"Luca Bianchi": hinted}, verdicts={"Luca Bianchi": 0.3})
    _use_laya(monkeypatch, tmp_path, fake)
    llm = _ScriptedLLM({"mentions": MENTIONS["mentions"][:1]}, {"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": TEXT}], "dpv_privacy", llm, linker="llm"))["results"][0]["entities"]
    assert (entity["type_id"], entity["confidence"]) == ("dpv-pd:Name", 0.3)
    assert {"type_id": "dpv-pd:Name", "probability": 0.1} in entity["ranking"] and len(entity["ranking"]) == 4


@pytest.mark.asyncio
async def test_one_step_text_gets_its_probability_from_laya(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import dataclasses

    import moose.ner as ner_module
    from moose.schema import COARSE_TYPES

    fake = _FakeLaya(plain={"Maria": {"PERSON": 0.6}}, verdicts={"Maria": 0.8})
    _use_laya(monkeypatch, tmp_path, fake)
    coarse = dataclasses.replace(get_schema_config("coarse"), candidate_mode="all",
                                 linker_calibration=get_schema_config("dpv_privacy").linker_calibration)
    monkeypatch.setattr(ner_module, "get_schema_config", lambda name: coarse)
    llm = _ScriptedLLM({"entities": [{"start": 0, "end": 5, "text": "Maria", "type_id": "PERSON"}]})
    [entity] = (await run_text_ner([{"task_id": "t", "text": "Maria lives in Milan."}], "coarse", llm))["results"][0]["entities"]
    assert (entity["type_id"], entity["confidence"], entity["decided_by"]) == ("PERSON", 0.8, "llm")
    assert '"scores"' not in llm.prompts[0] and fake.calls == [("Maria", None), ("Maria", "verify person")]


def test_only_checked_linkers_count_as_calibrated(tmp_path) -> None:
    from moose.linkers import Calibration

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"validated_for": ["text"], "linkers": {
        "system1": {"validated_for": ["text", "table"]}, "llm": {"validated_for": []},
    }}))
    cal = Calibration.load(path)
    assert cal.validated_tasks("system1") == ("text", "table") and cal.validated_tasks("llm") == ()
    assert cal.validated_tasks("system1+llm") == ("text",)  # Laya's own linker: the file's validated_for
    assert cal.validated_tasks("one_step") == ()  # never checked: not calibrated


def test_the_yes_no_question_names_one_type_and_not_the_llm() -> None:
    from moose.linkers import TABLE_PROMPT, laya_verify_inputs

    record = {"id": "dpv-pd:Name", "label": "Name", "definition": "A person's name."}
    state, questions = laya_verify_inputs(TEXT, {"text": "Luca Bianchi", "category": "person name"}, record)
    assert "LLM" not in state and "person name" not in state and '"Luca Bianchi"' in state
    assert questions["verdict"]["instructions"] == "Is the mention Name?"
    assert questions["verdict"]["criteria"] == {"yes": "Yes, it is Name: A person's name.", "no": "No, it is not Name"}
    _, table = laya_verify_inputs("Customer, E-mail", {"text": "E-mail: a@b.c"}, record, TABLE_PROMPT)
    assert table["verdict"]["instructions"] == "Do the column's values hold Name?"


def test_long_texts_are_cut_into_chunks_at_paragraph_breaks() -> None:
    from moose.ner import _text_chunks

    text = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(20))
    chunks = _text_chunks(text, 1000)
    assert len(chunks) > 1 and "".join(piece for _, piece in chunks) == text
    assert all(len(piece) <= 1000 and text[offset : offset + len(piece)] == piece for offset, piece in chunks)
    assert all(piece.endswith("\n\n") for _, piece in chunks[:-1])
    assert _text_chunks("short") == [(0, "short")]


@pytest.mark.asyncio
async def test_mentions_in_a_long_text_are_found_per_chunk_with_offsets_into_the_whole_text() -> None:
    from moose.ner import _text_chunks

    filler = "Nothing to see here. " * 60
    text = f"{filler}\n\nThe applicant, Mr Luca Bianchi, was born in 1954.\n\n{filler}"
    chunks = _text_chunks(text)
    assert len(chunks) > 1

    class _PerChunkLLM:
        async def generate(self, prompt: str) -> str:
            if 'Return {"mentions": []}' not in prompt:  # the decision step
                return json.dumps({"choices": [{"mention": 0, "type_id": "dpv-pd:Name"}]})
            # Offsets relative to the chunk; only the chunk holding the name has a mention.
            found = [{"start": 0, "end": 0, "text": "Luca Bianchi", "category": "person name"}] if "Luca Bianchi" in prompt else []
            return json.dumps({"mentions": found})

    result = await run_text_ner([{"task_id": "t", "text": text}], "dpv_privacy", _PerChunkLLM())
    [entity] = result["results"][0]["entities"]
    assert text[entity["start"] : entity["end"]] == "Luca Bianchi" and entity["type_id"] == "dpv-pd:Name"


def test_laya_sees_the_text_around_a_mention_in_a_long_text() -> None:
    from moose.linkers import LAYA_CONTEXT_CHARS, laya_inputs

    candidates = [{"id": "dpv-pd:Name", "label": "Name"}]
    long_text = "Procedure. " * 200 + "The applicant, Mr Luca Bianchi, was born in 1954. " + "Facts. " * 200
    start = long_text.index("Luca Bianchi")
    item = {"text": "Luca Bianchi", "start": start, "end": start + 12, "candidates": candidates}
    state, _ = laya_inputs(long_text, item)
    context = state.split("\n")[0]
    assert "Mr Luca Bianchi, was born" in context and len(context) < LAYA_CONTEXT_CHARS + 40
    # Short texts, and items without a position (table columns), keep the whole context.
    assert laya_inputs(TEXT, {**item, "start": 0, "end": 4})[0].startswith(f'Call transcript: "{TEXT}"')
    assert long_text in laya_inputs(long_text, {"text": "x", "candidates": candidates})[0]


def test_mentions_without_offsets_are_placed_in_order_and_each_repeat_gets_its_own_span() -> None:
    from moose.ner import _parse_mentions

    text = "Mr Rossi met Ms Bianchi. Later, Mr Rossi called Ms Bianchi and then Mr Rossi left."
    raw = json.dumps({"mentions": [
        {"text": "Rossi", "category": "person name"}, {"text": "Bianchi", "category": "person name"},
        {"text": "Rossi", "category": "person name"}, {"text": "Bianchi", "category": "person name"},
        {"text": "Rossi", "category": "person name"}, {"text": "Verdi", "category": "person name"},
    ]})
    mentions = _parse_mentions(raw, text)
    rossi = [i for i in range(len(text)) if text.startswith("Rossi", i)]
    assert [m["start"] for m in mentions if m["text"] == "Rossi"] == rossi
    assert len(mentions) == 5 and all(text[m["start"] : m["end"]] == m["text"] for m in mentions)
    # A repeat listed out of order still gets a free occurrence; one too many is dropped.
    out_of_order = json.dumps({"mentions": [{"text": "Bianchi"}, {"text": "Bianchi"}, {"text": "Rossi"}, {"text": "Bianchi"}]})
    placed = _parse_mentions(out_of_order, text)
    bianchi = [i for i in range(len(text)) if text.startswith("Bianchi", i)]
    assert [m["start"] for m in placed] == [*bianchi, rossi[2]]


def test_the_mention_prompt_asks_for_text_in_order_not_offsets() -> None:
    from moose.prompts import build_mention_prompt

    prompt = build_mention_prompt(get_schema_config("dpv_privacy"), "We collect email addresses.")
    assert '"start"' not in prompt and "do not give character offsets" in prompt
    assert "in the order they appear" in prompt


@pytest.mark.asyncio
async def test_a_chunk_the_llm_answers_nothing_for_is_split_in_two_and_tried_again() -> None:
    from moose.llm.base import EmptyResponse
    from moose.ner import MENTION_CHUNK_CHARS, _find_mentions
    from moose.config import Settings

    first = "Mr Luca Bianchi lives in Milan. " + "Nothing to see here. " * 20
    second = "His sister Anna Bianchi works in Turin. " + "Nothing more here. " * 20
    text = f"{first}\n\n{second}"
    assert len(text) <= MENTION_CHUNK_CHARS

    class _ChokesOnTheWholeText:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, prompt: str) -> str:
            self.calls += 1
            if "Milan" in prompt and "Turin" in prompt:
                raise EmptyResponse("reasoned until the output limit")
            names = [n for n in ("Luca Bianchi", "Milan", "Anna Bianchi", "Turin") if n in prompt]
            return json.dumps({"mentions": [{"text": n, "category": "x"} for n in names]})

    llm = _ChokesOnTheWholeText()
    found = await _find_mentions(text, lambda piece: f"TEXT {piece}", llm, Settings(MOOSE_MAX_RETRIES=0))
    assert [m["text"] for m in found] == ["Luca Bianchi", "Milan", "Anna Bianchi", "Turin"]
    assert all(text[m["start"] : m["end"]] == m["text"] for m in found) and llm.calls > 2
