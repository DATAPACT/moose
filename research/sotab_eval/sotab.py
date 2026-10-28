"""Moose's schema.org table annotation on SOTAB v2 (WDC Schema.org Table Annotation Benchmark).

Runs the shipped schemas as they are: `schemaorg_cta_v1` (a class per column) and
`schemaorg_cpa_v1` (the property relating the main column to each other column). The LLM
chooses in one step; the confidence is Laya's yes/no probability that the choice is right.
So this measures the LLM's accuracy and how well Laya's probabilities match it.

SOTAB's labels and Moose's label sets only partly overlap (see `map_cta` and `map_cpa`):
columns whose SOTAB label has no Moose counterpart are left out and counted as coverage.
SOTAB tables have no headers: columns are named "0", "1", ...

Data (CC BY 4.0, https://webdatacommons.org/structureddata/sotab/v2/), unzipped under data/:
    curl -LO https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/v2/CTA_validation_schemaorg.zip
    (and CTA_test, CPA_validation, CPA_test)

    ../.venv-laya/bin/python sotab.py --task cta --split validation --tables 50
    ../.venv-laya/bin/python sotab.py --task cpa --split validation --tables 50

Validation is the development split; test stays untouched until the approach is frozen.
Moose outputs are cached per table (raw.jsonl), so a rerun doesn't pay again, and a run
stops starting new tables once it has spent --max-cost (USD, $0.50 by default).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "research" / "dpv_eval"))

from compare_linkers import metrics  # noqa: E402
from moose import OpenRouterClient  # noqa: E402
from moose.config import Settings  # noqa: E402
from moose.cpa import run_table_cpa  # noqa: E402
from moose.llm.usage import track_llm_usage  # noqa: E402
from moose.ner import run_table_annotate  # noqa: E402
from moose.schema import DATA_DIR  # noqa: E402

DATA = HERE / "data"
RUNS = HERE / "runs"
CURATED = DATA_DIR / "schemaorg" / "subsets" / "curated_v1"
FILES = {
    ("cta", "validation"): ("CTA_validation_schemaorg", "Validation", "sotab_v2_cta_validation_set.csv"),
    ("cta", "test"): ("CTA_test_schemaorg", "Test", "sotab_v2_cta_test_set.csv"),
    ("cpa", "validation"): ("CPA_validation_schemaorg", "Validation", "sotab_v2_cpa_validation_set.csv"),
    ("cpa", "test"): ("CPA_test_schemaorg", "Test", "sotab_v2_cpa_test_set.csv"),
}
SCHEMAS = {"cta": "schemaorg_cta_v1", "cpa": "schemaorg_cpa_v1"}
# SOTAB's "AT" labels accept either of these types.
CTA_ALTERNATIVES = {"MusicArtistAT": ("MusicGroup", "Person")}


def _ids(name: str) -> set[str]:
    return {r["id"] for r in json.loads((CURATED / name).read_text(encoding="utf-8")) if r["id"].startswith("schema:")}


CTA_IDS, CPA_IDS = _ids("cta_labels.json"), _ids("cpa_labels.json")
ANCESTORS: dict[str, list[str]] = json.loads((CURATED / "class_ancestors.json").read_text(encoding="utf-8"))


def map_cta(label: str) -> tuple[str, ...]:
    """Moose class ids that are right for a SOTAB CTA label; () when Moose has none.
    "Book/name" (the names of books) is the class Book: Moose types a column by the
    entities it holds. Descriptions, literal properties (telephone, price, ...) and
    classes outside curated_v1 have no counterpart."""
    names = CTA_ALTERNATIVES.get(label) or ((label[: -len("/name")],) if label.endswith("/name") else (label,))
    return tuple(f"schema:{n}" for n in names if f"schema:{n}" in CTA_IDS)


def map_cpa(label: str) -> tuple[str, ...]:
    return (f"schema:{label}",) if f"schema:{label}" in CPA_IDS else ()


def gold(task: str, split: str) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Labelled columns per table, and how many columns each unmapped label has."""
    folder, _, labels = FILES[(task, split)]
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped: Counter = Counter()
    with open(DATA / folder / labels, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            accepted = (map_cta if task == "cta" else map_cpa)(row["label"])
            if not accepted:
                unmapped[row["label"]] += 1
                continue
            tables[row["table_name"]].append({"column": row["column_index"], "label": row["label"], "accepted": accepted,
                                              "main": row.get("main_column_index")})
    return tables, unmapped


def load_rows(task: str, split: str, table: str, max_rows: int, max_chars: int) -> list[dict[str, Any]]:
    folder, sub, _ = FILES[(task, split)]
    rows = []
    with gzip.open(DATA / folder / sub / table, "rt", encoding="utf-8") as f:
        for line in f:
            if len(rows) == max_rows:
                break
            record = json.loads(line)
            rows.append({k: (v[:max_chars] if isinstance(v, str) else v) for k, v in record.items()})
    return rows


def correct_cta(pred: str | None, accepted: tuple[str, ...], lenient: bool) -> bool:
    """Exact match; lenient also accepts a more specific class (Restaurant for LocalBusiness)."""
    if pred in accepted:
        return True
    return lenient and any(a in ANCESTORS.get(pred or "", []) for a in accepted)


async def annotate(task: str, split: str, table: str, columns: list[dict[str, Any]], llm, settings: Settings,
                   args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(task, split, table, args.rows, args.max_chars)
    with track_llm_usage(llm.model) as usage:
        if task == "cta":
            out = await run_table_annotate([{"task_id": "0", "table_id": table, "sampled_rows": rows}], SCHEMAS[task], llm, settings)
            preds = {c["column"]: c for c in out["results"][0]["columns"]}
        else:
            main = columns[0]["main"]
            targets = sorted({c["column"] for c in columns}, key=int)
            cpa_task = {"task_id": "0", "table_id": table, "sampled_rows": rows, "subject_column": main, "target_columns": targets}
            out = await run_table_cpa([cpa_task], SCHEMAS[task], llm, settings)
            preds = {r["target_column"]: {**r, "type_id": r.get("relation_id")} for r in out["results"][0]["relationships"]}
    return {"table": table, "preds": preds, "cost": usage.cost_usd, "calls": usage.calls, "prompt_tokens": usage.prompt_tokens}


def score(task: str, tables: dict[str, list[dict[str, Any]]], raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for table, columns in tables.items():
        preds = raw[table]["preds"]
        for col in columns:
            pred = preds.get(col["column"]) or {}
            pid = pred.get("type_id")
            rows.append({"table": table, "column": col["column"], "label": col["label"], "pred": pid,
                         "confidence": pred.get("confidence"),
                         "ok": pid in col["accepted"],
                         "ok_lenient": correct_cta(pid, col["accepted"], True) if task == "cta" else pid in col["accepted"]})
    ok = np.array([r["ok"] for r in rows], dtype=float)
    report: dict[str, Any] = {"columns": len(rows), "accuracy": round(float(ok.mean()), 3)}
    if task == "cta":
        report["accuracy_more_specific_ok"] = round(float(np.mean([r["ok_lenient"] for r in rows])), 3)
    scored = [r for r in rows if r["confidence"] is not None]
    report["with_laya_confidence"] = len(scored)
    if scored:
        p = np.array([r["confidence"] for r in scored], dtype=float)
        report["laya_yes_no"] = metrics(p, np.array([r["ok"] for r in scored], dtype=float))
    per_label: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        per_label[r["label"]].append(r["ok"])
    report["per_label"] = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 2)} for k, v in sorted(per_label.items(), key=lambda kv: -len(kv[1]))}
    report["errors"] = [{k: r[k] for k in ("table", "column", "label", "pred", "confidence")} for r in rows if not r["ok"]][:60]
    return report


async def main_async(args: argparse.Namespace) -> None:
    tables, unmapped = gold(args.task, args.split)
    if args.tables_from:
        # The same tables as an earlier run (its raw.jsonl), to compare like with like.
        lines = Path(args.tables_from).read_text(encoding="utf-8").splitlines()
        chosen = [t for t in (json.loads(line)["table"] for line in lines) if t in tables]
    else:
        chosen = sorted(tables)
        random.Random(args.seed).shuffle(chosen)
        chosen = chosen[: args.tables]
    model_tag = args.model.replace("/", "_")
    out_dir = RUNS / "_".join(filter(None, (args.task, args.split, model_tag, args.tag)))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "raw.jsonl"
    raw: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            raw[entry["table"]] = entry

    llm = OpenRouterClient(api_key=_api_key(), model=args.model)
    settings = Settings(MOOSE_MAX_RETRIES=2)
    gate = asyncio.Semaphore(args.concurrency)

    spent = 0.0

    async def one(table: str) -> None:
        nonlocal spent
        async with gate:
            if spent >= args.max_cost:
                return  # over budget: the table stays unannotated (and unscored)
            try:
                entry = await annotate(args.task, args.split, table, tables[table], llm, settings, args)
            except Exception as exc:  # noqa: BLE001 - one bad table must not stop the run
                print(f"  {table} failed: {exc}", flush=True)
                return
            raw[table] = entry
            spent += entry.get("cost") or 0
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  {table} done ({len(raw)}/{len(chosen)})", flush=True)

    await asyncio.gather(*(one(t) for t in chosen if t not in raw))
    if spent >= args.max_cost:
        print(f"Stopped at the --max-cost budget: ${spent:.2f} spent this run.", flush=True)

    done = {t: tables[t] for t in chosen if t in raw}
    all_columns = sum(len(v) for v in tables.values()) + sum(unmapped.values())
    report = {
        "task": args.task, "split": args.split, "schema": SCHEMAS[args.task], "model": args.model,
        "tables": len(done), "failed_tables": len(chosen) - len(done),
        "coverage": {"labelled_columns_in_split": all_columns,
                     "with_a_moose_label": all_columns - sum(unmapped.values()),
                     "share": round(1 - sum(unmapped.values()) / all_columns, 3),
                     "unmapped_labels": dict(unmapped.most_common())},
        "cost_usd": round(sum(raw[t].get("cost") or 0 for t in done), 4),
        **score(args.task, done, raw),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_label", "errors", "coverage")}, indent=1))
    print("coverage:", report["coverage"]["share"], "| report:", out_dir / "report.json")


def _api_key() -> str:
    key = os.getenv("MOOSE_OPENROUTER_API_KEY")
    if not key:
        for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("MOOSE_OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("MOOSE_OPENROUTER_API_KEY is not set (environment or .env).")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("cta", "cpa"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--tables", type=int, default=50, help="random tables with at least one mapped label")
    parser.add_argument("--max-cost", type=float, default=0.5,
                        help="USD this run may spend on the LLM; no new table starts once it is reached "
                        "(about $0.002 per CTA table, $0.008 per CPA table with gpt-oss-120b)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=5, help="rows shown to Moose per table")
    parser.add_argument("--max-chars", type=int, default=150, help="longest cell value shown")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--tables-from", help="an earlier run's raw.jsonl: annotate the same tables")
    parser.add_argument("--tag", help="suffix for the run folder")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
