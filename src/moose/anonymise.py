"""Anonymisation: find everything in a text that could identify a person (docs/design/anonymisation.md).

Step 1, here: find every mention of the eight kinds of information TAB, the Text
Anonymization Benchmark, annotates, whether or not it will need masking. Deciding what to
mask comes next, per entity and over the whole document.
"""

from __future__ import annotations

import json
from typing import Any

from moose.config import Settings, get_settings
from moose.ner import _find_mentions

# TAB's categories, described in our words. Every mention gets one of them.
CATEGORIES: dict[str, str] = {
    "PERSON": "names of people, with their titles (Mr, Dr), and initials, nicknames and usernames",
    "CODE": "numbers and codes that identify something: case and application numbers, ID, passport, "
            "phone and account numbers, licence plates",
    "LOC": "places: countries, regions, cities, districts, addresses, named buildings and infrastructure",
    "ORG": "named organisations: companies, schools, universities, public bodies, courts, prisons, "
           "hospitals, parties, churches, associations",
    "DEM": "demographic attributes of a person: nationality, ethnicity, native language, job title, rank, "
           "education, age, physical description, diagnosis",
    "DATETIME": 'specific dates, times and durations (e.g. "3 October 2018", "9.48 a.m.", "eighteen years")',
    "QUANTITY": "meaningful quantities: amounts of money, percentages, counts, measurements",
    "MISC": "any other specific detail that describes a person (an injury, a vehicle, a distinctive event)",
}
FALLBACK_CATEGORY = "MISC"


def build_identifier_mention_prompt(text: str) -> str:
    lines = [
        "You are an anonymisation engine.",
        "",
        "TASK",
        "Find every mention in the input text of the kinds of information below, whether or not it",
        "would need masking (that is decided later), and give each one its category:",
        *[f"- {name}: {description}" for name, description in CATEGORIES.items()],
        "",
        "SPAN RULES",
        "- Mark the shortest span that denotes the thing; for a person's name, keep titles and",
        '  honorifics (e.g. "Mr John Smith").',
        '- Keep units and qualifiers that are part of the value (e.g. "five years", "over 65", "GBP 94,000").',
        '- Leave out leading articles and possessives unless they are part of a name (e.g. "Ministry of',
        '  the Interior", not "the Ministry of the Interior"; but "The Hague").',
        '- Do not mark words that only refer to someone without naming or describing them (e.g. "the applicant").',
        "- Spans must not overlap.",
        "- List the spans in the order they appear, and a value again each time it appears.",
        "",
        "OUTPUT (API CONTRACT)",
        "Return ONLY valid JSON. No markdown. No extra text.",
        "Copy each span's text exactly as it is written in the input; do not give character offsets.",
        "{",
        '  "mentions": [',
        f'    {{"text": "<exact substring of the input>", "category": "<one of {", ".join(CATEGORIES)}>"}}',
        "  ]",
        "}",
        'Return {"mentions": []} if there is nothing to annotate.',
        "",
        "INPUT TEXT",
        json.dumps(text, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _category(raw: str) -> str:
    name = raw.strip().upper()
    return name if name in CATEGORIES else FALLBACK_CATEGORY


async def find_identifiers(text: str, llm_client, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Every mention of the kinds in CATEGORIES: {"start", "end", "text", "category"}, in order."""
    settings = settings or get_settings()
    mentions = await _find_mentions(text, build_identifier_mention_prompt, llm_client, settings)
    return [{**m, "category": _category(m["category"])} for m in mentions]
