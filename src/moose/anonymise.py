"""Anonymisation: find everything in a text that could identify a person (docs/design/anonymisation.md).

1. Find every mention of the eight kinds of information TAB, the Text Anonymization
   Benchmark, annotates, whether or not it will need masking (find_identifiers).
2. Decide, once per document, which entities to mask: DIRECT, QUASI or NO_MASK for each
   distinct mention text, applied to all its mentions; then mask them (anonymise).
"""

from __future__ import annotations

import json
from typing import Any

from moose.config import Settings, get_settings
from moose.ner import _find_mentions, _run_with_retries
from moose.validate import extract_json

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


IDENTIFIER_TYPES = ("DIRECT", "QUASI", "NO_MASK")
MASKED = ("DIRECT", "QUASI")
DEFAULT_PROTECT = "the private individuals in it"
# Distinct entities decided per call; a court case has a few dozen.
DECISION_BATCH = 80
# Characters of text around an entity's first mention, shown with it.
_CONTEXT = 60


def _entities(text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct mention texts, in order of first appearance, with their mentions."""
    by_text: dict[str, dict[str, Any]] = {}
    for m in mentions:
        entity = by_text.setdefault(m["text"], {"text": m["text"], "category": m["category"], "mentions": []})
        entity["mentions"].append(m)
    for entity in by_text.values():
        first = entity["mentions"][0]
        lo, hi = max(0, first["start"] - _CONTEXT), min(len(text), first["end"] + _CONTEXT)
        entity["context"] = " ".join(text[lo:hi].split())
    return list(by_text.values())


def build_masking_prompt(text: str, entities: list[dict[str, Any]], protect: str) -> str:
    listed = [
        json.dumps({"entity": i, "text": e["text"], "category": e["category"], "mentions": len(e["mentions"]),
                    "context": e["context"]}, ensure_ascii=False)
        for i, e in enumerate(entities)
    ]
    lines = [
        "You are an anonymisation engine.",
        "",
        "TASK",
        f"The text below is to be published so that {protect} cannot be identified. For each",
        "entity listed, decide whether it must be masked to protect them:",
        "- DIRECT: identifies the person on its own: their name, nickname or initials, an identity,",
        "  passport, phone or account number, the number of their own case or file.",
        "- QUASI: does not identify the person alone, but could together with other details in the",
        "  text and what others could know or find (news, social media, registers):",
        "  - dates, times and durations of events in their life; amounts, counts and measurements;",
        "  - places within a country (regions, cities, districts, streets, buildings), and",
        "    organisations tied to a place or to the person: local courts, councils, police and",
        "    prisons, their employer, school, party, union or association;",
        "  - their nationality, ethnicity, language, family, job, rank, age, health and injuries;",
        "  - the names of anyone else (relatives, witnesses, lawyers, officials, judges), and the",
        "    numbers of other cases, files and applications: they tie the text to the person's case.",
        "- NO_MASK: does not help identify them: the country or state itself, its national",
        "  government, ministries and highest courts, international bodies and courts, references to",
        '  laws, articles and conventions, and vague expressions that fit many people ("several',
        '  years", "a relative").',
        "Decide over the whole text: the more specific details remain, the easier re-identification",
        "is. When unsure whether a detail could help identify the person, choose QUASI.",
        "",
        "ENTITIES (one JSON object per line)",
        *listed,
        "",
        "OUTPUT (API CONTRACT)",
        "Return ONLY valid JSON. No markdown. No extra text.",
        '{"decisions": [{"entity": <int>, "identifier_type": "DIRECT" | "QUASI" | "NO_MASK"}]}',
        "Give a decision for every entity.",
        "",
        "TEXT",
        json.dumps(text, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _parse_decisions(raw_text: str, count: int) -> dict[int, str]:
    data = extract_json(raw_text)
    items = data.get("decisions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError('Expected {"decisions": [...]}.')
    decisions: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("entity"), int) or not 0 <= item["entity"] < count:
            raise ValueError('Each decision needs the "entity" number of a listed entity.')
        kind = str(item.get("identifier_type", "")).strip().upper()
        if kind not in IDENTIFIER_TYPES:
            raise ValueError(f'"identifier_type" must be one of {", ".join(IDENTIFIER_TYPES)}, not {kind!r}.')
        decisions[item["entity"]] = kind
    return decisions


def mask(text: str, entities: list[dict[str, Any]]) -> str:
    """The text with every masked entity's mentions replaced by one placeholder per entity,
    numbered per category in order of first appearance: [PERSON_1], [DATETIME_2]."""
    counts: dict[str, int] = {}
    spans = []
    for entity in entities:
        if entity["identifier_type"] not in MASKED:
            continue
        counts[entity["category"]] = counts.get(entity["category"], 0) + 1
        placeholder = f"[{entity['category']}_{counts[entity['category']]}]"
        spans += [(m["start"], m["end"], placeholder) for m in entity["mentions"]]
    out, cursor = [], 0
    for start, end, placeholder in sorted(spans):
        out += [text[cursor:start], placeholder]
        cursor = end
    return "".join(out + [text[cursor:]])


async def anonymise(text: str, llm_client, protect: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Find what could identify `protect` (by default the text's private individuals), decide what
    to mask, and mask it. Returns {"entities": [...], "masked_text": str}: one entry per
    mention, with its category, its identifier_type and the number of its entity."""
    settings = settings or get_settings()
    return await decide_and_mask(text, await find_identifiers(text, llm_client, settings), llm_client, protect, settings)


async def decide_and_mask(
    text: str, mentions: list[dict[str, Any]], llm_client, protect: str | None = None, settings: Settings | None = None
) -> dict[str, Any]:
    """Step 2 of anonymise, given the mentions find_identifiers found."""
    settings = settings or get_settings()
    protect = protect or DEFAULT_PROTECT
    entities = _entities(text, mentions)
    for offset in range(0, len(entities), DECISION_BATCH):
        batch = entities[offset : offset + DECISION_BATCH]
        decisions = await _run_with_retries(
            llm_client, build_masking_prompt(text, batch, protect),
            lambda raw, n=len(batch): _parse_decisions(raw, n), settings.MOOSE_MAX_RETRIES,
        )
        for i, entity in enumerate(batch):
            entity["identifier_type"] = decisions.get(i, "QUASI")  # not decided: mask, to be safe
    mentions = [
        {**m, "identifier_type": e["identifier_type"], "entity": n, "confidence": None, "calibrated": False}
        for n, e in enumerate(entities) for m in e["mentions"]
    ]
    return {"entities": sorted(mentions, key=lambda m: m["start"]), "masked_text": mask(text, entities)}
