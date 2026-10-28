from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from moose.config import Settings, get_settings
from moose.linkers import CPA_PROMPT, Calibration, llm_choice_verdicts
from moose.ner import _choices_to_scores, run_table_annotate
from moose.prob import choose_argmax
from moose.prompts import build_cpa_prompt, build_cpa_shortlist_prompt, table_to_markdown
from moose.schema import DATA_DIR, get_schema_config
from moose.validate import extract_json, validate_cpa_response


# --------------------------
# Utilities
# --------------------------
def _ordered_union_columns(sampled_rows: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for row in sampled_rows:
        for k in row.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)
    return cols


def _sentinel_labels(relation_ids: list[str]) -> list[str]:
    out: list[str] = []
    for s in ("moose:NONE", "moose:OTHER", "CPA:NONE", "CPA:OTHER"):
        if s in relation_ids:
            out.append(s)
    return out


def _normalize_schema_curie(value: str) -> str:
    """
    Accept schema CURIE or IRI:
      - schema:Book
      - https://schema.org/Book
      - http://schema.org/Book
    Return schema:Book (best-effort).
    """
    v = (value or "").strip()
    if not v:
        return v
    if v.startswith("schema:"):
        return v
    if v.startswith("https://schema.org/"):
        return "schema:" + v[len("https://schema.org/") :]
    if v.startswith("http://schema.org/"):
        return "schema:" + v[len("http://schema.org/") :]
    return v


def _coerce_cpa_output_with_task_ids(raw_text: str, tasks: list[dict[str, Any]]) -> str:
    data = extract_json(raw_text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("CPA response must be a JSON array or object.")
    if all(isinstance(item, dict) and isinstance(item.get("task_id"), str) for item in data):
        return json.dumps(data, ensure_ascii=True)
    if len(data) != len(tasks):
        raise ValueError("CPA response length mismatch.")

    coerced: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Each CPA response item must be an object.")
        relationships = item.get("relationships")
        if not isinstance(relationships, list):
            raise ValueError("Each CPA response item must include a relationships array.")

        expected = tasks[index]
        table_id = item.get("table_id")
        if not isinstance(table_id, str):
            table_id = expected["table_id"]
        subject_column = item.get("subject_column")
        if not isinstance(subject_column, str):
            subject_column = expected["subject_column"]

        coerced.append(
            {
                "task_id": expected["task_id"],
                "table_id": table_id,
                "subject_column": subject_column,
                "relationships": relationships,
            }
        )

    return json.dumps(coerced, ensure_ascii=True)


def _batch_ids(build_prompt, ids: list[str], max_chars: int) -> list[list[str]]:
    """Split ids into batches whose prompt (build_prompt(batch)) stays within max_chars."""
    batches: list[list[str]] = []
    current: list[str] = []
    for type_id in ids:
        candidate = current + [type_id]
        if current and len(build_prompt(candidate)) > max_chars:
            batches.append(current)
            current = [type_id]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


async def _run_with_retries(llm_client, prompt: str, validator, max_retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt == 0:
            response = await llm_client.generate(prompt)
        else:
            correction = (
                "\n\nThe previous output was invalid: "
                f"{last_error}. Return ONLY valid JSON following the schema."
            )
            response = await llm_client.generate(prompt + correction)
        try:
            return validator(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"LLM output invalid after {max_retries} retries: {last_error}")


# --------------------------
# schema.org domain filtering
# --------------------------
@lru_cache
def _load_schemaorg_subset_indices(subset: str = "curated_v1") -> tuple[dict[str, Any], dict[str, list[str]]]:
    base = DATA_DIR / "schemaorg" / "subsets" / subset
    prop_index_path = base / "property_index.json"
    class_anc_path = base / "class_ancestors.json"

    if not prop_index_path.exists():
        raise FileNotFoundError(f"schema.org property_index not found: {prop_index_path}")
    if not class_anc_path.exists():
        raise FileNotFoundError(f"schema.org class_ancestors not found: {class_anc_path}")

    prop_index = json.loads(prop_index_path.read_text(encoding="utf-8"))
    class_anc = json.loads(class_anc_path.read_text(encoding="utf-8"))

    if not isinstance(prop_index, dict):
        raise ValueError("schema.org property_index.json must be a JSON object")
    if not isinstance(class_anc, dict):
        raise ValueError("schema.org class_ancestors.json must be a JSON object")

    class_anc_norm: dict[str, list[str]] = {}
    for k, v in class_anc.items():
        if isinstance(k, str):
            if isinstance(v, list):
                class_anc_norm[k] = [x for x in v if isinstance(x, str)]
            else:
                class_anc_norm[k] = []
    return prop_index, class_anc_norm


FLATTENED_VALUE_CLASSES = frozenset({
    "schema:PostalAddress", "schema:GeoCoordinates", "schema:Offer", "schema:AggregateRating", "schema:Rating",
    "schema:OpeningHoursSpecification",
})


def _filter_schemaorg_predicates_by_domain(
    relation_ids: list[str],
    subject_class: str,
    subset: str = "curated_v1",
) -> list[str]:
    """
    Keep predicate p if domainIncludes(p) intersects {subject_class + ancestors(subject_class)}.
    Always keep sentinel labels.
    """
    subject_class = _normalize_schema_curie(subject_class)
    if not subject_class:
        return relation_ids

    prop_index, class_anc = _load_schemaorg_subset_indices(subset=subset)
    allowed_domains = {subject_class}
    allowed_domains.update(class_anc.get(subject_class, []))
    # Tables often hold the parts of a structured value flat, next to the subject (a business's
    # postalCode rather than its address): keep the parts' properties when the subject can have one.
    for meta in prop_index.values():
        if isinstance(meta, dict) and any(d in allowed_domains for d in meta.get("domains") or []):
            allowed_domains.update(r for r in meta.get("ranges") or [] if r in FLATTENED_VALUE_CLASSES)

    sentinels = set(_sentinel_labels(relation_ids))
    filtered: list[str] = []

    for rid in relation_ids:
        if rid in sentinels:
            filtered.append(rid)
            continue

        meta = prop_index.get(rid)
        if not isinstance(meta, dict):
            filtered.append(rid)
            continue

        domains = meta.get("domains") or []
        if not isinstance(domains, list):
            domains = []
        domains = [d for d in domains if isinstance(d, str)]

        # schema.org sometimes omits domains; keep if unknown
        if not domains:
            filtered.append(rid)
            continue

        if any(d in allowed_domains for d in domains):
            filtered.append(rid)

    # Guard against over-filtering; if too small, fall back
    if len(filtered) < 3:
        return relation_ids
    return filtered


# --------------------------
# CTA inference (default)
# --------------------------
async def _infer_subject_class_via_cta(
    sampled_rows: list[dict[str, Any]],
    subject_column: str,
    llm_client,
    settings: Settings,
    cta_schema: str = "schemaorg_cta_v1",
) -> str | None:
    """
    Infer subject class by running schema.org CTA typing and extracting type for subject_column.
    Returns None if unknown or moose:OTHER.
    """
    task = {"task_id": "cta-subject-1", "table_id": "cta-subject", "sampled_rows": sampled_rows}
    out = await run_table_annotate([task], cta_schema, llm_client, settings=settings)
    results = out.get("results", [])
    if not results:
        return None
    cols = results[0].get("columns", [])
    for c in cols:
        if c.get("column") == subject_column:
            tid = c.get("type_id")
            if not isinstance(tid, str):
                return None
            tid = _normalize_schema_curie(tid)
            if tid in {"moose:OTHER", "CPA:OTHER"}:
                return None
            # Only accept schema:* for subject class
            if tid.startswith("schema:"):
                return tid
            return None
    return None


# --------------------------
# Deep CPA: shortlists, then one decision per table
# --------------------------
SHORTLIST_SIZE = 8


def _to_allowed(value: str, allowed: set[str]) -> str | None:
    """A label as the LLM wrote it (schema:author, author, https://schema.org/author) to its id."""
    rid = _normalize_schema_curie(value)
    if rid in allowed:
        return rid
    return f"schema:{rid}" if f"schema:{rid}" in allowed else None


def _parse_shortlists(raw_text: str, targets: list[str], allowed: set[str]) -> dict[str, list[str]]:
    data = extract_json(raw_text)
    if not isinstance(data, dict):
        raise ValueError("The shortlist response must be a JSON object keyed by target column.")
    out: dict[str, list[str]] = {}
    for target in targets:
        values = data.get(target)
        if not isinstance(values, list):
            raise ValueError(f"The shortlist response has no list for target column {target!r}.")
        ids = (_to_allowed(v, allowed) for v in values if isinstance(v, str))
        out[target] = list(dict.fromkeys(rid for rid in ids if rid))
    return out


async def _shortlist_relation_ids(
    schema_config,
    *,
    task: dict[str, Any],
    relation_ids: list[str],
    llm_client,
    settings: Settings,
    max_rows: int,
) -> dict[str, list[str]]:
    """Recall step: for every target column at once, the relations that could link the subject
    column to it, most likely first. Large label sets are asked in batches."""
    targets: list[str] = task["target_columns"]
    sentinels = set(_sentinel_labels(relation_ids))
    candidates = [rid for rid in relation_ids if rid not in sentinels]
    shortlists: dict[str, list[str]] = {t: [] for t in targets}

    def build(ids: list[str]) -> str:
        return build_cpa_shortlist_prompt(schema_config, task, ids, SHORTLIST_SIZE, max_rows=max_rows)

    for batch in _batch_ids(build, candidates, settings.MOOSE_MAX_CHARS_PER_PROMPT):
        allowed = set(batch)
        found = await _run_with_retries(
            llm_client, build(batch), lambda raw, allowed=allowed: _parse_shortlists(raw, targets, allowed), settings.MOOSE_MAX_RETRIES
        )
        for target, ids in found.items():
            shortlists[target].extend(rid for rid in ids if rid not in shortlists[target])
    return shortlists


async def _decide_relations(
    schema_config,
    task_base: dict[str, Any],
    targets: list[str],
    relation_ids: list[str],
    llm_client,
    settings: Settings,
    max_rows: int,
    candidates: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """One LLM call choosing a relation for each target column, which sees the others too;
    then the System 1 model's probability for each choice."""
    task_id, table_id, subject_column = task_base["task_id"], task_base["table_id"], task_base["subject_column"]
    chunk_task = {**task_base, "target_columns": targets}
    if candidates:
        chunk_task["candidates"] = candidates
    prompt = build_cpa_prompt(schema_config, chunk_task, relation_ids, max_rows=max_rows)
    allowed_set = set(relation_ids)

    def validator(raw_text: str):
        raw_text = _choices_to_scores(raw_text, relation_ids, schema_config.require_all_scores, key="relation_id")
        normalized_raw = _coerce_cpa_output_with_task_ids(
            raw_text, [{"task_id": task_id, "table_id": table_id, "subject_column": subject_column}]
        )
        return validate_cpa_response(
            [{"task_id": task_id, "table_id": table_id, "subject_column": subject_column, "target_columns": targets}],
            normalized_raw,
            allowed_set,
            require_all_scores=schema_config.require_all_scores,
            type_aliases=schema_config.type_aliases,
            type_alias_prefixes=schema_config.type_alias_prefixes,
        )

    parsed = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
    out: dict[str, dict[str, Any]] = {}
    for rel in parsed[0].relationships:
        scores = {rid: float(rel.scores.get(rid, 0.0)) for rid in relation_ids}
        relation_id, _llm_score, _distribution = choose_argmax(scores)
        out[rel.target_column] = {"target_column": rel.target_column, "relation_id": relation_id}
    await _system1_probabilities(
        schema_config, task_base["sampled_rows"], subject_column, [out[c] for c in targets if c in out], max_rows
    )
    return out


# --------------------------
# Fast CPA chunking
# --------------------------
def _chunk_targets_by_prompt_size(
    schema_config,
    task_base: dict[str, Any],
    relation_ids: list[str],
    targets: list[str],
    max_chars: int,
    max_rows: int,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []

    for col in targets:
        candidate = current + [col]
        candidate_task = {**task_base, "target_columns": candidate}
        prompt = build_cpa_prompt(schema_config, candidate_task, relation_ids, max_rows=max_rows)

        if current and len(prompt) > max_chars:
            chunks.append(current)
            current = [col]
        else:
            current = candidate

    if current:
        candidate_task = {**task_base, "target_columns": current}
        prompt = build_cpa_prompt(schema_config, candidate_task, relation_ids, max_rows=max_rows)
        if len(prompt) > max_chars and len(current) == 1:
            raise ValueError(
                "CPA prompt too large even for a single target column. "
                "Reduce sampled_rows size or relationship label set."
            )
        chunks.append(current)

    return chunks


# --------------------------
# Main CPA function
# --------------------------
async def _system1_probabilities(
    schema_config, sampled_rows: list[dict[str, Any]], subject_column: str, relationships: list[dict[str, Any]], max_rows: int
) -> None:
    """Every probability is the System 1 model's: its yes/no answer on each relation the LLM
    chose. Adds confidence (None without Laya), calibrated and decided_by in place."""
    calibration = Calibration.load(schema_config.linker_calibration)
    records = {r["id"]: r for r in schema_config.load_type_records()}
    targets = [r["target_column"] for r in relationships]
    context = table_to_markdown(sampled_rows, [subject_column, *targets], max_rows=max_rows)
    items = [{"text": f"{subject_column} -> {target}"} for target in targets]
    chosen = [records.get(r["relation_id"]) or {"id": r["relation_id"]} for r in relationships]
    verdicts = await llm_choice_verdicts(calibration, context, items, chosen, CPA_PROMPT)
    calibrated = "table" in calibration.validated_tasks("one_step") if calibration else False
    for rel, yes in zip(relationships, verdicts):
        rel.update(confidence=round(yes, 4) if yes is not None else None, calibrated=calibrated, decided_by="llm")


async def run_table_cpa(
    tasks: list[dict[str, Any]],
    schema: str,
    llm_client,
    settings: Settings | None = None,
    max_rows_in_prompt: int = 5,
) -> dict[str, Any]:
    settings = settings or get_settings()
    schema_config = get_schema_config(schema)
    if not getattr(schema_config, "supports_cpa", False):
        raise ValueError(f"Schema '{schema}' does not support CPA.")

    relation_ids = schema_config.load_type_ids()

    results_by_id: dict[str, dict[str, Any]] = {}

    for task in tasks:
        task_id = task["task_id"]
        table_id = task["table_id"]
        sampled_rows = task["sampled_rows"]

        subject_column = task.get("subject_column")
        if not isinstance(subject_column, str) or not subject_column.strip():
            raise ValueError(f"CPA requires subject_column for task_id={task_id}")
        subject_column = subject_column.strip()

        subject_class = task.get("subject_class")
        if subject_class is not None and not isinstance(subject_class, str):
            raise ValueError(f"subject_class must be a string if provided (task_id={task_id})")
        subject_class = _normalize_schema_curie(subject_class) if isinstance(subject_class, str) else None

        target_columns = task.get("target_columns")
        debug = bool(task.get("debug", False))
        debug_preview_limit = int(task.get("debug_preview_limit", 20))
        debug_preview_limit = max(0, min(debug_preview_limit, 200))

        all_columns = _ordered_union_columns(sampled_rows)
        if subject_column not in all_columns:
            raise ValueError(
                f"subject_column '{subject_column}' not found in sampled_rows columns for task_id={task_id}. "
                f"Available: {all_columns}"
            )

        if target_columns is None:
            target_columns = [c for c in all_columns if c != subject_column]
        if not isinstance(target_columns, list) or not all(isinstance(c, str) and c.strip() for c in target_columns):
            raise ValueError(f"Invalid target_columns for task_id={task_id}")
        target_columns = [c.strip() for c in target_columns if c.strip() and c.strip() != subject_column]

        if not target_columns:
            results_by_id[task_id] = {
                "task_id": task_id,
                "table_id": table_id,
                "subject_column": subject_column,
                "relationships": [],
            }
            continue

        # ------------------------------------------------------------
        # DEEP CPA: a shortlist per target column, then one decision for the table
        # ------------------------------------------------------------
        if schema_config.prefilter_types:
            debug_obj: dict[str, Any] = {}
            if debug:
                debug_obj = {
                    "schema": schema_config.name,
                    "prefilter_types": True,
                    "selection_mode": "per_table",
                    "targets": {},
                }

            # Default CTA→CPA orchestration: infer subject_class if needed (schema.org CPA only)
            subject_class_source = "provided" if subject_class else "inferred"
            inferred = None
            if schema_config.name.startswith("schemaorg_") and not subject_class:
                inferred = await _infer_subject_class_via_cta(
                    sampled_rows=sampled_rows,
                    subject_column=subject_column,
                    llm_client=llm_client,
                    settings=settings,
                    cta_schema="schemaorg_cta_v1",
                )
                subject_class = inferred
                subject_class_source = "inferred" if subject_class else "none"

            if debug:
                debug_obj["subject_class"] = subject_class
                debug_obj["subject_class_source"] = subject_class_source
                debug_obj["relation_ids_total"] = len(relation_ids)

            # Apply deterministic domain filtering if we have a schema.org subject_class
            domain_filtered_relation_ids = relation_ids
            if schema_config.name.startswith("schemaorg_") and subject_class:
                try:
                    domain_filtered_relation_ids = _filter_schemaorg_predicates_by_domain(
                        relation_ids, subject_class, subset="curated_v1"
                    )
                except Exception:  # noqa: BLE001
                    domain_filtered_relation_ids = relation_ids

            if debug:
                debug_obj["relation_ids_after_domain_filter"] = len(domain_filtered_relation_ids)

            task_base = {
                "task_id": task_id,
                "table_id": table_id,
                "sampled_rows": sampled_rows,
                "subject_column": subject_column,
            }
            shortlists = await _shortlist_relation_ids(
                schema_config,
                task={**task_base, "target_columns": target_columns, "subject_class": subject_class},
                relation_ids=domain_filtered_relation_ids,
                llm_client=llm_client,
                settings=settings,
                max_rows=max_rows_in_prompt,
            )
            sentinels = _sentinel_labels(relation_ids)

            def allowed_for(targets: list[str]) -> list[str]:
                listed = {rid for t in targets for rid in shortlists[t]}
                return [rid for rid in domain_filtered_relation_ids if rid in listed and rid not in sentinels] + sentinels

            relationships_out: dict[str, dict[str, Any]] = {}
            chunks = _chunk_targets_by_prompt_size(
                schema_config=schema_config,
                task_base=task_base,
                relation_ids=allowed_for(target_columns),
                targets=target_columns,
                max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
                max_rows=max_rows_in_prompt,
            )
            for chunk_targets in chunks:
                relationships_out.update(
                    await _decide_relations(
                        schema_config, task_base, chunk_targets, allowed_for(chunk_targets), llm_client, settings,
                        max_rows_in_prompt, candidates={t: shortlists[t] for t in chunk_targets},
                    )
                )

            if debug:
                debug_obj["targets"] = {
                    t: {"shortlist_count": len(shortlists[t]), "shortlist_preview": shortlists[t][:debug_preview_limit]}
                    for t in target_columns
                }

            ordered_relationships = [relationships_out[c] for c in target_columns if c in relationships_out]
            out_item: dict[str, Any] = {
                "task_id": task_id,
                "table_id": table_id,
                "subject_column": subject_column,
                "relationships": ordered_relationships,
            }
            if debug:
                out_item["debug"] = debug_obj

            results_by_id[task_id] = out_item
            continue

        # ------------------------------------------------------------
        # FAST CPA: CPA-lite, chunk targets into few prompts
        # ------------------------------------------------------------
        task_base = {
            "task_id": task_id,
            "table_id": table_id,
            "sampled_rows": sampled_rows,
            "subject_column": subject_column,
        }
        chunks = _chunk_targets_by_prompt_size(
            schema_config=schema_config,
            task_base=task_base,
            relation_ids=relation_ids,
            targets=target_columns,
            max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
            max_rows=max_rows_in_prompt,
        )

        relationships_out = {}
        for chunk_targets in chunks:
            relationships_out.update(
                await _decide_relations(schema_config, task_base, chunk_targets, relation_ids, llm_client, settings, max_rows_in_prompt)
            )

        ordered_relationships = [relationships_out[c] for c in target_columns if c in relationships_out]
        results_by_id[task_id] = {
            "task_id": task_id,
            "table_id": table_id,
            "subject_column": subject_column,
            "relationships": ordered_relationships,
        }

    ordered_results = [results_by_id[t["task_id"]] for t in tasks]
    return {"results": ordered_results}
