from __future__ import annotations

import json

import pytest

from moose.anonymise import CATEGORIES, build_identifier_mention_prompt, find_identifiers

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
