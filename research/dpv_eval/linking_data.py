"""Mentions from two-step runs, with their gold concept and the LLM's choice.

Only mentions whose span matches a gold span exactly are kept, so scores computed on them
estimate P(chosen type is right | span is right).
"""

from __future__ import annotations

from typing import Any

import bench
from moose.schema import get_schema_config

SCHEMA = "dpv_privacy"
RUN = "moose_dpv_privacy_expand_openai_gpt-oss-120b_{split}"


def correct(type_id: str, gold: str) -> bool:
    return bench.collapse_to_benchmark(type_id) == gold


def mentions(split: str, run: str = RUN, schema: str = SCHEMA) -> list[dict[str, Any]]:
    """`run` is a run folder name under runs/, with {split} for the split."""
    run_dir = bench.RUNS / run.format(split=split)
    gold = {r["id"]: r for r in bench.load_split(split)}
    # Candidates no longer in the schema's pool (it was cleaned since the run) are dropped.
    records = {r["id"]: r for r in get_schema_config(schema).load_type_records()}
    out = []
    for row in bench.read_jsonl(run_dir / "raw.jsonl"):
        record = gold[row["id"]]
        spans = {(a["start"], a["end"]): a for a in record["annotations"]}
        for ent in row["entities"]:
            ann = spans.get((ent["start"], ent["end"]))
            candidates = [records[c] for c in ent.get("candidates") or [] if c in records]
            if ann and candidates:
                out.append({
                    "id": row["id"], "group": record["template_group_id"], "text": record["text"],
                    "mention": {"text": ent["text"], "category": ent.get("category"), "candidates": candidates},
                    "gold": ann["schema_id"], "llm_choice": ent["type_id"], "llm_stated": float(ent.get("confidence") or 0),
                })
    return out
