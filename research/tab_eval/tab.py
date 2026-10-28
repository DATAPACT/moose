"""Moose's personal-data detection on TAB, the Text Anonymization Benchmark (ECHR court cases).

TAB marks, for each document, the mentions that identify the person to protect, directly
(DIRECT, a name) or together with other facts (QUASI, a date, a place, an employer), and those
that need no masking (NO_MASK). Moose annotates each document with `dpv_privacy` as it is; the
spans it gives a personal-data type (`dpv-pd:*`) are the spans it would mask, scored with TAB's
own evaluation.py (entity recall on DIRECT and QUASI identifiers, token precision).
Roles such as `dpv:Applicant` are not masked; the `all` variant masks every annotated span.

TAB has 8 coarse types (PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC) that do not
map one to one onto DPV, so for types the report only lists which DPV concepts Moose gave the
mentions of each TAB type.

Data (MIT, https://github.com/NorskRegnesentral/text-anonymization-benchmark) under data/:
echr_dev.json, echr_test.json and evaluation.py. The scorer needs spacy and en_core_web_md.

    PYTHONPATH=../../src ../.venv-laya/bin/python tab.py --split dev --docs 15

Dev is the development split; test is for checking frozen changes. Moose outputs are cached
per document (raw.jsonl), and a run stops starting new documents once it has spent --max-cost.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from moose import OpenRouterClient
from moose.anonymise import find_identifiers
from moose.config import Settings
from moose.llm.usage import track_llm_usage
from moose.ner import run_text_ner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
RUNS = HERE / "runs"
SCHEMA = "dpv_privacy"


def _scorer():
    """TAB's own evaluation.py, as downloaded (not part of this repository)."""
    spec = importlib.util.spec_from_file_location("tab_evaluation", DATA / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def masked_spans(entities: list[dict[str, Any]], variant: str) -> list[list[int]]:
    keep = (lambda e: True) if variant == "all" else (lambda e: str(e.get("type_id", "")).startswith("dpv-pd:"))
    return [[e["start"], e["end"]] for e in entities if keep(e)]


def type_table(docs: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """For gold DIRECT/QUASI mentions (first annotator) that a Moose entity overlaps: the DPV
    concepts Moose gave them, per TAB type. Also how many mentions of each type Moose found."""
    concepts: dict[str, Counter] = defaultdict(Counter)
    found: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for doc_id, entry in raw.items():
        mentions = next(iter(docs[doc_id]["annotations"].values()))["entity_mentions"]
        for m in mentions:
            if m["identifier_type"] == "NO_MASK":
                continue
            hits = [e for e in entry["entities"] if e["start"] < m["end_offset"] and m["start_offset"] < e["end"]]
            found[m["entity_type"]][1] += 1
            if hits:
                found[m["entity_type"]][0] += 1
                concepts[m["entity_type"]][hits[0].get("type_id")] += 1
    return {t: {"found": f"{found[t][0]}/{found[t][1]}", "moose_types": dict(concepts[t].most_common(6))}
            for t in sorted(found, key=lambda t: -found[t][1])}


async def main_async(args: argparse.Namespace) -> None:
    docs = {d["doc_id"]: d for d in json.loads((DATA / f"echr_{args.split}.json").read_text(encoding="utf-8"))}
    if args.docs_from:
        chosen = [json.loads(line)["doc_id"] for line in Path(args.docs_from).read_text().splitlines()]
    else:
        chosen = sorted(docs)
        random.Random(args.seed).shuffle(chosen)
        chosen = chosen[: args.docs]
    mode = "anonymise" if args.mode == "anonymise" else None
    out_dir = RUNS / "_".join(filter(None, (mode, args.split, args.model.replace("/", "_"), args.tag)))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "raw.jsonl"
    raw: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            raw[entry["doc_id"]] = entry

    llm = OpenRouterClient(api_key=_api_key(), model=args.model)
    settings = Settings(MOOSE_MAX_RETRIES=2)

    async def annotate(text: str) -> list[dict[str, Any]]:
        if args.mode == "anonymise":  # step 1 only: every mention found, its category as its type
            return [{**m, "type_id": m["category"]} for m in await find_identifiers(text, llm, settings)]
        out = await run_text_ner([{"task_id": "t", "text": text}], SCHEMA, llm, settings)
        return out["results"][0]["entities"]
    gate = asyncio.Semaphore(args.concurrency)
    spent = 0.0

    async def one(doc_id: str) -> None:
        nonlocal spent
        async with gate:
            if spent >= args.max_cost:
                return
            try:
                with track_llm_usage(llm.model) as usage:
                    entities = await annotate(docs[doc_id]["text"])
            except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
                print(f"  {doc_id} failed: {exc}", flush=True)
                return
            entry = {"doc_id": doc_id, "entities": entities, "cost": usage.cost_usd, "calls": usage.calls}
            raw[doc_id] = entry
            spent += usage.cost_usd or 0
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  {doc_id} done ({len(raw)}/{len(chosen)})", flush=True)

    await asyncio.gather(*(one(d) for d in chosen if d not in raw))
    if spent >= args.max_cost:
        print(f"Stopped at the --max-cost budget: ${spent:.2f} spent this run.", flush=True)

    done = {d: raw[d] for d in chosen if d in raw}
    scorer = _scorer()
    gold = scorer.GoldCorpus(str(DATA / f"echr_{args.split}.json"))
    report: dict[str, Any] = {"split": args.split, "mode": args.mode, "model": args.model, "documents": len(done),
                              "failed_documents": len(chosen) - len(done),
                              "cost_usd": round(sum(e.get("cost") or 0 for e in done.values()), 4)}
    variants = ("all",) if args.mode == "anonymise" else ("personal_data", "all")
    for variant in variants:
        masked = [scorer.MaskedDocument(d, [tuple(s) for s in masked_spans(e["entities"], variant)]) for d, e in done.items()]
        report[variant] = scorer.evaluate(gold, masked)
    report["types"] = type_table(docs, done)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = ("recall_direct_entities", "recall_quasi_entities", "mention_recall", "token_precision", "mention_precision", "token_f1")
    print(json.dumps({k: report[k] for k in ("documents", "failed_documents", "cost_usd")}))
    for variant in variants:
        print(variant, {k: report[variant][k] for k in keys})
    print("report:", out_dir / "report.json")


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
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--mode", choices=("dpv_privacy", "anonymise"), default="dpv_privacy",
                        help="anonymise: moose.anonymise's mention step (every mention found is masked)")
    parser.add_argument("--docs", type=int, default=15, help="random documents of the split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--docs-from", help="an earlier run's raw.jsonl: annotate the same documents")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=0.5, help="USD this run may spend on the LLM")
    parser.add_argument("--tag", help="suffix for the run folder")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
