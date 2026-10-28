from __future__ import annotations

import json
from typing import Any

from moose.schema import SchemaConfig


def _format_intro(text: str) -> str:
    return f"{text.rstrip()}\n"


def _escape_md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    # Markdown table delimiter is '|'
    text = text.replace("|", "\\|")
    return text


def table_to_markdown(sampled_rows: list[dict[str, Any]], columns: list[str], max_rows: int = 5) -> str:
    """
    Render a markdown table with given columns, using up to max_rows sampled rows.
    """
    cols = list(columns)
    header = "| " + " | ".join(_escape_md_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in sampled_rows[:max_rows]:
        lines.append("| " + " | ".join(_escape_md_cell(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _ordered_union_columns(sampled_rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in sampled_rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if not isinstance(key, str):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def build_cpa_prompt(schema: SchemaConfig, task: dict[str, Any], relation_ids: list[str], max_rows: int = 5) -> str:
    subject = task["subject_column"]
    targets: list[str] = task["target_columns"]
    table_md = table_to_markdown(task["sampled_rows"], [subject] + targets, max_rows=max_rows)

    none_label = "moose:NONE" if "moose:NONE" in relation_ids else ("CPA:NONE" if "CPA:NONE" in relation_ids else None)
    other_label = "moose:OTHER" if "moose:OTHER" in relation_ids else ("CPA:OTHER" if "CPA:OTHER" in relation_ids else None)

    intro = _format_intro(schema.cpa_intro)
    allowed = ", ".join(relation_ids)

    rules_lines = [
        "- Output relationships for EXACTLY the provided target columns (no extra, no missing).",
        "- Give exactly one relation_id per target column, from the allowed relationship labels.",
        "- Do not give scores, probabilities or confidence values.",
    ]
    if none_label:
        rules_lines.append(f"- Choose {none_label} if there is no relationship.")
    if other_label:
        rules_lines.append(f"- Choose {other_label} if there is a relationship but it is not represented in the label set.")
    if len(targets) > 1:
        rules_lines.append(
            "- Different target columns usually hold different properties: compare similar columns "
            "(for example a start and an end date) before choosing."
        )
    # Deep CPA: each target column's shortlist from the recall step (build_cpa_shortlist_prompt).
    candidates: dict[str, list[str]] = task.get("candidates") or {}
    shortlist_lines = []
    if candidates:
        rules_lines.append("- Prefer the labels in each target column's shortlist; any allowed label may still be chosen.")
        shortlist_lines = ["Shortlist per target column (most likely first):"] + [
            f"- {target}: {', '.join(candidates.get(target) or []) or '(none)'}" for target in targets
        ] + [""]

    return "\n".join(
        [
            intro.rstrip(),
            "Task: Predict the semantic relationship (predicate) between the SUBJECT column and each TARGET column.",
            "Return ONLY valid JSON.",
            f"Allowed relationship labels: {allowed}",
            "",
            "Output format (JSON array):",
            "[",
            "  {",
            '    "table_id": "...",',
            '    "subject_column": "...",',
            '    "relationships": [',
            "      {",
            '        "target_column": "...",',
            '        "relation_id": "<label>"',
            "      }",
            "    ]",
            "  }",
            "]",
            "",
            "Rules:",
            *rules_lines,
            "No extra text around the JSON.",
            "",
            "Input:",
            f'Table ID: {task["table_id"]}',
            f"Subject column: {subject}",
            f"Target columns: {json.dumps(targets, ensure_ascii=True)}",
            "",
            *shortlist_lines,
            "Sample table (markdown):",
            table_md,
        ]
    )


def build_cpa_shortlist_prompt(
    schema: SchemaConfig, task: dict[str, Any], relation_ids: list[str], size: int, max_rows: int = 5
) -> str:
    """Deep CPA, recall step: the possible relations for every target column of a table at once,
    so each column gets its own shortlist and the final choice sees the whole table."""
    subject = task["subject_column"]
    targets: list[str] = task["target_columns"]
    subject_class = task.get("subject_class")
    return "\n".join(
        [
            _format_intro(schema.cpa_intro).rstrip(),
            "Task: for EACH target column, list the relationship labels that could relate the SUBJECT column "
            f"to it, most likely first, at most {size} per column. Be recall-oriented: the final choice is made later.",
            *([f"The subject column holds {subject_class} entities."] if subject_class else []),
            "Return ONLY a JSON object mapping each target column name to its list of labels:",
            '{"<target column>": ["<label>", "<label>"]}',
            "",
            "Rules:",
            "- Include every target column, named exactly as given; use [] when no label could apply.",
            "- Only use labels from the allowed relationship labels.",
            "No extra text around the JSON.",
            "",
            f"Allowed relationship labels: {', '.join(relation_ids)}",
            "",
            f"Subject column: {subject}",
            f"Target columns: {json.dumps(targets, ensure_ascii=True)}",
            "",
            "Sample table (markdown):",
            table_to_markdown(task["sampled_rows"], [subject] + targets, max_rows=max_rows),
        ]
    )

TYPE_SELECT_INTRO = "You are a type inventory selector for semantic typing."


# Speaker labels in transcripts ("Agent:", "Customer:") say who talks; they are not mentions.
SPEAKER_LABEL_RULE = 'Do not annotate speaker labels that only mark who is talking (e.g. "Agent:", "Customer:").'
# Span boundaries, shared by the one-step and the two-step prompts.
VALUE_SPAN_RULES = (
    'The span is the value itself, without cue words or field names before it (e.g. "USR 661 204", not "Customer ID USR 661 204").',
    'Do not include trailing unit words that are not part of the value (e.g. "seventy two", not "seventy two years old").',
    "Split a value into separate spans when its parts are different kinds of data (e.g. a street address and its postal code).",
    'Keep qualifiers that are part of a value or range (e.g. "over 65", "under 18").',
    'Keep a value spoken in parts as one span (e.g. "month nine, year twenty eight" for an expiry date).',
    'Leave out leading articles and possessives unless they are part of a name (e.g. "courier", not "the courier"; but "The Hague").',
)
# Up to this many types, the one-step prompt also gives each type's definition; beyond it
# the list of ids alone is already long (dpv has 9,000+).
MAX_DEFINED_TYPES = 60


def _type_definition_lines(schema: SchemaConfig, type_ids: list[str]) -> list[str]:
    if len(type_ids) > MAX_DEFINED_TYPES:
        return []
    try:
        records = {r["id"]: r for r in schema.load_type_records()}
    except (OSError, ValueError):
        return []
    lines = []
    for type_id in type_ids:
        record = records.get(type_id) or {}
        definition = (record.get("definition") or "").strip()
        if definition:
            label = record.get("label") or type_id
            lines.append(f"- {type_id}" + (f" ({label})" if label != type_id else "") + f": {definition}")
    return ["", "TYPE DEFINITIONS", *lines] if lines else []


def build_text_ner_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    is_single = len(tasks) == 1
    payload = tasks[0]["text"] if is_single else [t["text"] for t in tasks]
    intro = _format_intro(schema.text_intro).strip()
    allowed_json = json.dumps(type_ids, ensure_ascii=True)

    task_block = (
        "Identify spans that explicitly denote concepts mappable to the allowed type_ids."
        if is_single
        else "For each input text, identify spans that explicitly denote concepts mappable to the allowed type_ids."
    )
    has_pd = any(type_id.startswith("dpv-pd:") for type_id in type_ids)
    has_ai = any(type_id.startswith("dpv-ai:") for type_id in type_ids)
    if has_pd and has_ai:
        task_block = (
            "\n".join(
                [
                    "Identify spans that explicitly denote:",
                    "1) personal data categories (DPV-PD), and/or",
                    "2) AI-related concepts/activities (DPV-AI),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
            if is_single
            else "\n".join(
                [
                    "For each input text, identify spans that explicitly denote:",
                    "1) personal data categories (DPV-PD), and/or",
                    "2) AI-related concepts/activities (DPV-AI),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
        )
    elif has_pd:
        task_block = (
            "\n".join(
                [
                    "Identify spans that explicitly denote personal data categories (DPV-PD),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
            if is_single
            else "\n".join(
                [
                    "For each input text, identify spans that explicitly denote personal data categories (DPV-PD),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
        )
    elif has_ai:
        task_block = (
            "\n".join(
                [
                    "Identify spans that explicitly denote AI-related concepts/activities (DPV-AI),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
            if is_single
            else "\n".join(
                [
                    "For each input text, identify spans that explicitly denote AI-related concepts/activities (DPV-AI),",
                    "using ONLY the provided allowed type_ids.",
                ]
            )
        )

    lines = [
        intro,
        "",
        "TASK",
        task_block,
        "",
        "OUTPUT (API CONTRACT)",
        "Return ONLY valid JSON. No markdown. No extra text.",
    ]

    if is_single:
        lines.extend(
            [
                "Top-level output MUST be a JSON object for this single input text.",
                "",
                "RESPONSE SHAPE",
                "{",
                '  "entities": [',
                "    {",
                '      "start": <int>,',
                '      "end": <int>,',
                '      "text": "<exact substring from the input text>",',
                '      "type_id": "<one allowed type_id>"',
                "    },",
                "    ...",
                "  ]",
                "}",
            ]
        )
    else:
        lines.extend(
            [
                "Top-level output MUST be a JSON array with length exactly equal to the number of input texts.",
                "Output item at index i corresponds to input text at index i (order-preserving, 1:1 mapping).",
                "",
                "RESPONSE SHAPE (per input text)",
                "Each output item MUST be:",
                "{",
                '  "entities": [',
                "    {",
                '      "start": <int>,',
                '      "end": <int>,',
                '      "text": "<exact substring from the input text>",',
                '      "type_id": "<one allowed type_id>"',
                "    },",
                "    ...",
                "  ]",
                "}",
            ]
        )

    lines.extend(
        [
            "",
            "ALLOWED LABEL SPACE",
            '"type_id" MUST be one of this allowed set:',
            allowed_json,
            *_type_definition_lines(schema, type_ids),
            "",
            "SPAN RULES (NER BEST PRACTICES)",
            "- Offsets are computed on the raw input text exactly as provided (including punctuation and spaces).",
            '- Offsets are 0-based, and "end" is end-exclusive.',
            '- "text" MUST exactly equal input_text[start:end] (exact match).',
            "- Prefer the most specific and complete mention span.",
            "- Do NOT output duplicate entities with identical (start,end).",
            "- Avoid overlaps. Only allow overlapping spans if both are necessary and semantically distinct.",
            "- Do not annotate generic words unless they explicitly indicate relevant concepts from allowed type_ids"
            + (" (the schema's rules at the end of this list take precedence)." if schema.mention_rules else "."),
            *[f"- {rule}" for rule in VALUE_SPAN_RULES],
            f"- {SPEAKER_LABEL_RULE}",
            *[f"- {rule}" for rule in (schema.mention_rules or ())],
            '- If no entities exist, return: "entities": [].',
            "",
            "TYPE RULES",
            "- Give exactly one type_id per entity: the most specific allowed type that fits.",
            "- Do not give scores, probabilities or confidence values.",
            "- Do NOT infer unstated concepts.",
            "",
            "INPUT",
            "Input text JSON:" if is_single else "Input texts JSON:",
            json.dumps(payload, ensure_ascii=True),
            "",
            "OUTPUT",
            (
                "Return ONLY the JSON object described above."
                if is_single
                else "Return ONLY the JSON array described above."
            ),
        ]
    )

    return "\n".join(lines)


def build_tabular_cell_ner_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    """
    Each task is ONE TABLE CELL. Offsets are relative to that cell's text.
    Expected task keys:
      - text: cell string
      - row_index, column identify the cell
      - table_id is optional context
    """
    payload = [
        {
            "table_id": t.get("table_id"),
            "row_index": t.get("row_index"),
            "column": t.get("column"),
            "text": t["text"],
        }
        for t in tasks
    ]

    types = ", ".join(type_ids)
    intro = _format_intro(schema.text_intro)

    return "".join(
        [
            intro,
            "You are a high-precision NER engine operating over TABLE CELLS.\n",
            "Each task corresponds to one cell's text. Offsets are relative to that cell's text.\n",
            f"Schema: {schema.name}\n",
            f"Allowed type_ids: {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\n",
            "  {\n",
            "    \"row_index\": 0,\n",
            "    \"column\": \"...\",\n",
            "    \"entities\": [\n",
            "      {\n",
            "        \"start\": 0,\n",
            "        \"end\": 0,\n",
            "        \"text\": \"exact substring from the cell text\",\n",
            "        \"type_id\": \"PERSON\"\n",
            "      }\n",
            "    ]\n",
            "  }\n",
            "]\n",
            "Rules:\n",
            "- Return EXACTLY one item per input cell and keep input order.\n",
            "- Each output item must keep the same row_index and column as its input cell.\n",
            "- Offsets are 0-based, end-exclusive.\n",
            "- entity.text MUST equal cell_text[start:end] exactly.\n",
            "- Give exactly one type_id per entity, from the allowed type_ids.\n",
            "- Do not give scores, probabilities or confidence values.\n",
            "No extra text around the JSON.\n\n",
            "Input cell tasks JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


def build_type_selection_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str], mode: str) -> str:
    if mode == "text":
        payload = [t["text"] for t in tasks]
        mode_hint = "text"
    elif mode == "table":
        payload = [
            {
                "table_id": t["table_id"],
                "sampled_rows": t["sampled_rows"],
            }
            for t in tasks
        ]
        mode_hint = "tabular"
    elif mode == "cpa":
        payload = [
            {
                "table_id": t["table_id"],
                "subject_column": t["subject_column"],
                "target_column": t["target_column"],
                "sampled_rows": t["sampled_rows"],
            }
            for t in tasks
        ]
        mode_hint = "cpa"
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    types = ", ".join(type_ids)
    intro = _format_intro(TYPE_SELECT_INTRO)
    return "".join(
        [
            intro,
            f"Schema: {schema.name}\n",
            f"Input mode: {mode_hint}\n",
            f"Allowed type_ids (subset): {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\"type_id\", \"type_id\"]\n",
            "Rules:\n",
            "- Only include type_ids from the allowed list.\n",
            "- Be recall-oriented: include all types that could apply.\n",
            "- Return unique type_ids only.\n",
            "- If none apply, return an empty list [].\n",
            "No extra text around the JSON.\n\n",
            "Input data JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


def build_table_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    payload = [
        {
            "table_id": t["table_id"],
            "observed_columns": _ordered_union_columns(t["sampled_rows"]),
            "sampled_rows": t["sampled_rows"],
        }
        for t in tasks
    ]
    types = ", ".join(type_ids)
    intro = _format_intro(schema.table_intro)
    return "".join(
        [
            intro,
            f"Schema: {schema.name}\n",
            f"Allowed type_ids: {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\n",
            "  {\n",
            "    \"table_id\": \"...\",\n",
            "    \"columns\": [\n",
            "      {\"column\": \"name\", \"type_id\": \"PERSON\"}\n",
            "    ]\n",
            "  }\n",
            "]\n",
            "Rules:\n",
            "- Return EXACTLY one item per input table and keep input order.\n",
            "- Return one entry per observed column name from observed_columns.\n",
            "- Column names in output MUST match observed_columns exactly (including case, underscores, and punctuation).\n",
            "- Do not rename, normalize, split, or merge column names.\n",
            "- Give exactly one type_id per column from the allowed type_ids.\n",
            "- Do not give scores, probabilities or confidence values.\n",
            "No extra text around the JSON.\n\n",
            "Input tables JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


# Extra span rules come from the schema's registry entry ("mention_rules"); none by default.
DEFAULT_MENTION_RULES: tuple[str, ...] = ()


def build_mention_prompt(schema: SchemaConfig, text: str) -> str:
    """Two-step mode, step 1: find mentions and describe each one, without a type list."""
    guidance = schema.mention_guidance or schema.description or f"concepts of the {schema.label} vocabulary"
    lines = [
        _format_intro(schema.text_intro).strip(),
        "",
        "TASK",
        f"Find every span in the input text that denotes: {guidance}",
        "For each span, give a short generic category (2-6 words) saying what kind of thing it is,",
        'for example "customer identifier", "medical condition" or "company receiving the data".',
        "",
        "SPAN RULES",
        *[f"- {rule}" for rule in VALUE_SPAN_RULES],
        "- Spans must not overlap.",
        f"- {SPEAKER_LABEL_RULE}",
        *[f"- {rule}" for rule in (schema.mention_rules or DEFAULT_MENTION_RULES)],
        "- List the spans in the order they appear, and a value again each time it appears.",
        "",
        "OUTPUT (API CONTRACT)",
        "Return ONLY valid JSON. No markdown. No extra text.",
        "Copy each span's text exactly as it is written in the input; do not give character offsets.",
        "{",
        '  "mentions": [',
        '    {"text": "<exact substring of the input>", "category": "<short generic category>"}',
        "  ]",
        "}",
        'Return {"mentions": []} if there is nothing to annotate.',
        "",
        "INPUT TEXT",
        json.dumps(text, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _describe_candidate(record: dict[str, Any], max_definition: int = 220) -> str:
    definition = (record.get("definition") or "").strip()
    if len(definition) > max_definition:
        definition = definition[: max_definition - 1].rstrip() + "…"
    label = record.get("label") or record["id"]
    return f"{record['id']} ({label})" + (f": {definition}" if definition else "")


def build_candidate_choice_prompt(schema: SchemaConfig, text: str, mentions: list[dict[str, Any]]) -> str:
    """Two-step mode, step 2: choose each mention's type among its own candidates."""
    lines = [
        _format_intro(schema.text_intro).strip(),
        "",
        "TASK",
        "Each mention below was found in the input text. Choose the single type that best fits each mention,",
        "using ONLY that mention's candidates. Prefer the most specific type that is correct.",
        'Choose "none" when no candidate fits.',
        "",
        "INPUT TEXT",
        json.dumps(text, ensure_ascii=False),
        "",
        "MENTIONS",
    ]
    for index, mention in enumerate(mentions):
        lines.append(f"[{index}] {json.dumps(mention['text'], ensure_ascii=False)} (described as: {mention['category']})")
        for record in mention["candidates"]:
            lines.append(f"    - {_describe_candidate(record)}")
    lines.extend(
        [
            "",
            "OUTPUT (API CONTRACT)",
            "Return ONLY valid JSON. No markdown. No extra text. One choice per mention:",
            "{",
            '  "choices": [',
            '    {"mention": <index>, "type_id": "<one of its candidate ids, or none>"}',
            "  ]",
            "}",
        ]
    )
    return "\n".join(lines)


def build_column_description_prompt(schema: SchemaConfig, columns: dict[str, list[str]]) -> str:
    """Two-step mode for tables, step 1: describe each column, without a type list."""
    guidance = schema.mention_guidance or schema.description or f"concepts of the {schema.label} vocabulary"
    lines = [
        _format_intro(schema.table_intro).strip(),
        "",
        "TASK",
        f"For each column of the table below, decide whether its values denote: {guidance}",
        "For each such column, give a short generic category (2-6 words) saying what its values are,",
        'for example "customer email addresses", "dates of birth" or "companies receiving the data".',
        "",
        "COLUMNS (name: sample values)",
    ]
    for name, values in columns.items():
        lines.append(f"- {json.dumps(name, ensure_ascii=False)}: {' | '.join(values) if values else '(empty)'}")
    lines.extend(
        [
            "",
            "OUTPUT (API CONTRACT)",
            "Return ONLY valid JSON. No markdown. No extra text. One entry per column, using its exact name:",
            "{",
            '  "columns": [',
            '    {"column": "<column name>", "relevant": <true|false>, "category": "<short generic category, or empty>"}',
            "  ]",
            "}",
        ]
    )
    return "\n".join(lines)


def build_column_choice_prompt(schema: SchemaConfig, columns: list[dict[str, Any]]) -> str:
    """Two-step mode for tables, step 2: choose each column's type among its own candidates."""
    lines = [
        _format_intro(schema.table_intro).strip(),
        "",
        "TASK",
        "Choose the single type that best fits the values of each column below, using ONLY that column's",
        'candidates. Prefer the most specific type that is correct. Choose "none" when no candidate fits.',
        "",
        "COLUMNS",
    ]
    for index, column in enumerate(columns):
        lines.append(f"[{index}] {column['text']} (described as: {column['category']})")
        for record in column["candidates"]:
            lines.append(f"    - {_describe_candidate(record)}")
    lines.extend(
        [
            "",
            "OUTPUT (API CONTRACT)",
            "Return ONLY valid JSON. No markdown. No extra text. One choice per column:",
            "{",
            '  "choices": [',
            '    {"column": <index>, "type_id": "<one of its candidate ids, or none>"}',
            "  ]",
            "}",
        ]
    )
    return "\n".join(lines)
