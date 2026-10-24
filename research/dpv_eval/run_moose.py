"""Run DPVCallBench records through a local Moose API and score them.

Two settings:
  --schema dpv                  open DPV: Moose's full DPV schema (9k+ concepts)
  --schema dpvcallbench_labels  closed set: the benchmark's 21 labels as a user schema
                                (run with --register first; needs login off locally)

Raw Moose outputs are cached per record in runs/<run>/raw.jsonl, so an interrupted or
repeated run does not pay for records it already has.

    python research/dpv_eval/run_moose.py --schema dpv --split dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

import bench

LABEL_SCHEMA = "dpvcallbench_labels"
# docker-compose.yml mounts ./data/moose-user here; ingest accepts files under /app.
HOST_SCHEMA_DIR = bench.REPO / "data" / "moose-user"
CONTAINER_SCHEMA_DIR = "/app/src/moose/data/user"


def _env() -> dict[str, str]:
    env = {}
    for line in (bench.REPO / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def _base_url(env: dict[str, str]) -> str:
    return os.getenv("MOOSE_URL") or f"http://127.0.0.1:{env.get('MOOSE_API_PORT', '8000')}"


async def _wait(client: httpx.AsyncClient, job_id: str, timeout: float = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = (await client.get(f"/jobs/{job_id}")).raise_for_status().json()
        if job["status"] in ("completed", "failed"):
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout:.0f}s.")


async def register_label_schema() -> None:
    """Ingest the benchmark's labels and definitions as a Moose text schema."""
    env = _env()
    if env.get("MOOSE_AUTH_ENABLED", "true").lower() != "false":
        raise SystemExit("Registering needs MOOSE_AUTH_ENABLED=false (local admin) in .env.")
    types = [{"id": item["label"], "label": item["label"].replace("_", " "), "description": item["definition"]} for item in bench.labels()]
    HOST_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    (HOST_SCHEMA_DIR / f"{LABEL_SCHEMA}.json").write_text(json.dumps({"types": types}, indent=2), encoding="utf-8")
    async with httpx.AsyncClient(base_url=_base_url(env), timeout=60) as client:
        resp = await client.post(
            "/schemas/ingest",
            json={
                "name": LABEL_SCHEMA,
                "label": "DPVCallBench labels",
                "description": "The 21 DPVCallBench v0.1 labels, for closed-set evaluation.",
                "source_type": "file",
                "source_path": f"{CONTAINER_SCHEMA_DIR}/{LABEL_SCHEMA}.json",
                "supports_table": False,
            },
        )
        resp.raise_for_status()
        job = await _wait(client, resp.json()["job_id"])
    if job["status"] != "completed":
        raise SystemExit(f"Ingest failed: {job.get('error')}")
    print(f"Registered schema {LABEL_SCHEMA} with {len(types)} types.")


def _to_benchmark(record: dict[str, Any], entities: list[dict[str, Any]], schema: str) -> list[dict[str, Any]]:
    text = record["text"]
    label_to_schema = bench.label_to_schema()
    schema_to_label = bench.schema_to_label()
    out = []
    for ent in entities:
        start, end, span = ent.get("start"), ent.get("end"), ent.get("text") or ""
        if not (isinstance(start, int) and isinstance(end, int) and text[start:end] == span):
            start = text.find(span) if span else -1
            end = start + len(span)
        if start < 0:
            continue
        type_id = ent.get("type_id") or ""
        if schema == LABEL_SCHEMA:
            if type_id not in label_to_schema:
                continue
            label, schema_id = type_id, label_to_schema[type_id]
        else:
            schema_id = type_id
            label = schema_to_label.get(bench.collapse_to_benchmark(type_id), "")
        out.append({"start": start, "end": end, "text": text[start:end], "label": label, "schema_id": schema_id,
                    "confidence": ent.get("confidence")})
    return out


async def run(args: argparse.Namespace) -> None:
    env = _env()
    records = bench.load_split(args.split, args.limit)
    linker = f"_{args.linker.replace('+', '-')}" if args.linker else ""
    run_dir = bench.RUNS / (args.run or f"moose_{args.schema}{linker}_{args.model.replace('/', '_')}_{args.split}")
    raw_path = run_dir / "raw.jsonl"
    done = {row["id"]: row for row in bench.read_jsonl(raw_path)}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} records, {len(done)} cached, {len(todo)} to run.")

    headers = {"X-API-Key": env["MOOSE_API_KEY"]} if env.get("MOOSE_API_KEY") else {}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    run_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(base_url=_base_url(env), headers=headers, timeout=60) as client:

        async def one(record: dict[str, Any]) -> None:
            async with sem:
                started = time.monotonic()
                resp = await client.post(
                    f"/schemas/{args.schema}/ner",
                    json={"text": record["text"], "llm": {"model": args.model}, **({"linker": args.linker} if args.linker else {})},
                )
                resp.raise_for_status()
                job = await _wait(client, resp.json()["job_id"])
            row = {"id": record["id"], "status": job["status"], "seconds": round(time.monotonic() - started, 2),
                   "llm": job.get("llm"), "entities": (job.get("result") or {}).get("entities", []),
                   "error": job.get("error")}
            async with lock:
                with raw_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[record["id"]] = row
                print(f"  {record['id']} {job['status']} {row['seconds']}s", flush=True)

        await asyncio.gather(*(one(r) for r in todo))

    summarize(run_dir, records, args.schema)


def _stem(word: str) -> str:
    for suffix in ("es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _category_mention(entity: dict[str, Any]) -> bool:
    """Heuristic: a span naming a kind of data ("email addresses") rather than giving a value.
    The benchmark leaves those unannotated; the generic DPV schemas annotate them, as privacy
    policies need. No digits, "@" or capitalised words, and a word shared with the concept's name."""
    text = entity.get("text") or ""
    if any(ch.isdigit() for ch in text) or "@" in text or any(w[:1].isupper() for w in text.split()):
        return False
    concept = re.findall(r"[A-Z][a-z]+|[a-z]+", (entity.get("type_id") or "").split(":")[-1])
    return bool({_stem(w.lower()) for w in re.findall(r"[A-Za-z]+", text)} & {_stem(w.lower()) for w in concept})


def mention_level(records: list[dict[str, Any]], raw: dict[str, Any], scored: list[dict[str, Any]]) -> dict[str, Any]:
    """On predicted spans that match a gold span exactly: is the concept right, and how well does
    the confidence match that (ECE)? Also who decided, and how many spans look like category mentions."""
    gold = {(r["id"], a["start"], a["end"]): a["schema_id"] for r in records for a in r["annotations"]}
    pairs, deciders, categories = [], {}, 0
    for row in raw.values():
        for ent in row["entities"]:
            deciders[ent.get("decided_by", "llm")] = deciders.get(ent.get("decided_by", "llm"), 0) + 1
            categories += _category_mention(ent)
            key = (row["id"], ent.get("start"), ent.get("end"))
            if key in gold and ent.get("confidence") is not None:
                pairs.append((float(ent["confidence"]), bench.collapse_to_benchmark(ent["type_id"]) == gold[key], bool(ent.get("calibrated"))))
    out: dict[str, Any] = {"decided_by": deciders, "category_mentions": categories}
    if pairs:
        bins: dict[int, list[tuple[float, bool]]] = {}
        for c, ok, _ in pairs:
            bins.setdefault(min(int(c * 10), 9), []).append((c, ok))
        ece = sum(len(b) / len(pairs) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values())
        out["matched_spans"] = {
            "n": len(pairs),
            "concept_accuracy": round(sum(ok for _, ok, _ in pairs) / len(pairs), 3),
            "mean_confidence": round(sum(c for c, _, _ in pairs) / len(pairs), 3),
            "ece": round(ece, 3),
            "marked_calibrated": sum(cal for _, _, cal in pairs),
        }
    return out


def summarize(run_dir: Path, records: list[dict[str, Any]], schema: str) -> dict[str, Any]:
    raw = {row["id"]: row for row in bench.read_jsonl(run_dir / "raw.jsonl")}
    gold_path = run_dir / "gold.jsonl"
    bench.write_jsonl(gold_path, records)
    predictions = [{"id": r["id"], "annotations": _to_benchmark(r, raw[r["id"]]["entities"], schema)} for r in records if r["id"] in raw]
    bench.write_jsonl(run_dir / "predictions.jsonl", predictions)
    report: dict[str, Any] = {"schema": schema, "records": len(records),
                              "failed_jobs": sum(raw[r["id"]]["status"] != "completed" for r in records if r["id"] in raw),
                              "raw": bench.score(gold_path, run_dir / "predictions.jsonl")}
    if schema != LABEL_SCHEMA:
        # Adapter variant: finer DPV concepts mapped up to the benchmark's concepts.
        collapsed = [{"id": p["id"], "annotations": [{**a, "schema_id": bench.collapse_to_benchmark(a["schema_id"])} for a in p["annotations"]]}
                     for p in predictions]
        bench.write_jsonl(run_dir / "predictions_collapsed.jsonl", collapsed)
        report["collapsed_to_benchmark"] = bench.score(gold_path, run_dir / "predictions_collapsed.jsonl")
        scored = collapsed
        # Without spans that only name a kind of data, which the benchmark leaves unannotated.
        values_only = []
        for r, p in zip([r for r in records if r["id"] in raw], collapsed):
            kept = {(e["start"], e["end"]) for e in raw[r["id"]]["entities"] if not _category_mention(e)}
            values_only.append({"id": p["id"], "annotations": [a for a in p["annotations"] if (a["start"], a["end"]) in kept]})
        bench.write_jsonl(run_dir / "predictions_values_only.jsonl", values_only)
        report["collapsed_values_only"] = bench.score(gold_path, run_dir / "predictions_values_only.jsonl")
    else:
        scored = predictions
    usage = [raw[r["id"]].get("llm") or {} for r in records if r["id"] in raw]
    report["cost_usd"] = round(sum(u.get("cost_usd") or 0 for u in usage), 4)
    report["prompt_tokens_per_record"] = round(sum(u.get("prompt_tokens") or 0 for u in usage) / max(len(usage), 1))
    report["seconds_per_record"] = round(sum(raw[r["id"]]["seconds"] for r in records if r["id"] in raw) / max(len(usage), 1), 2)
    report["per_label"] = bench.per_label(records, scored)
    report.update(mention_level(records, raw, scored))
    report["errors"] = bench.errors(records, scored)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    headline = report.get("collapsed_to_benchmark", report["raw"])
    print(json.dumps({k: (v["f1"] if isinstance(v, dict) and "f1" in v else v) for k, v in headline.items()}, indent=1))
    print(f"cost ${report['cost_usd']}, {report['prompt_tokens_per_record']} prompt tokens/record -> {run_dir}")
    print("matched spans:", report.get("matched_spans"), "| decided_by:", report.get("decided_by"), "| category mentions:", report.get("category_mentions"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", default="dpv")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--run", help="Run directory name under runs/.")
    parser.add_argument("--linker", choices=["system1+llm", "system1", "llm"], help="Two-step schemas: who decides.")
    parser.add_argument("--register", action="store_true", help=f"Ingest the {LABEL_SCHEMA} schema and exit.")
    args = parser.parse_args()
    asyncio.run(register_label_schema() if args.register else run(args))


if __name__ == "__main__":
    main()
