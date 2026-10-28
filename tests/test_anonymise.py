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


def test_decisions_must_name_listed_entities_and_known_types() -> None:
    assert _parse_decisions('{"decisions": [{"entity": 1, "identifier_type": "no_mask"}]}', 2) == {1: "NO_MASK"}
    with pytest.raises(ValueError, match="listed entity"):
        _parse_decisions('{"decisions": [{"entity": 2, "identifier_type": "QUASI"}]}', 2)
    with pytest.raises(ValueError, match="must be one of"):
        _parse_decisions('{"decisions": [{"entity": 0, "identifier_type": "MAYBE"}]}', 2)
