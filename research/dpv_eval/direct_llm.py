"""Direct-LLM baseline: one closed-set prompt per record, as in moose_api_quickstart.ipynb.

The LLM gets the 21 DPVCallBench labels with their DPV ids and definitions, and the
benchmark's own rules (values only, no speaker labels), and returns spans, labels and ids
in one call. It knows the answer space in advance, so it is the closed-set ceiling for
this benchmark rather than a rival for open-vocabulary DPV annotation.

The provider key is read from the repository .env (MOOSE_OPENROUTER_API_KEY or
OPENROUTER_API_KEY) and never printed. Responses are cached per record in
runs/<run>/raw.jsonl.

    ../../.venv/bin/python direct_llm.py --split dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

import bench

BASE_URL = "https://openrouter.ai/api/v1"
KEY_NAMES = ("MOOSE_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")


def _key() -> str:
    env = {}
    for line in (bench.REPO / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    key = next((env[k] for k in KEY_NAMES if env.get(k)), "")
    if not key:
        raise SystemExit(f"Set one of {', '.join(KEY_NAMES)} in .env")
    return key


def prompt(text: str) -> str:
    """The notebook's prompt, unchanged."""
    contract = "\n".join(f"- {i['label']} -> {i['schema_id']}: {i['definition']}" for i in bench.labels())
    return f"""You are a high-precision privacy entity annotation system.
Find every target entity in the transcript and link it to the DPVCallBench label and DPV ID.

Return exactly one JSON object with this shape:
{{"annotations":[{{"start":0,"end":4,"text":"exact substring","label":"name","schema_id":"dpv-pd:Name"}}]}}

Rules:
- Offsets are zero-based and end-exclusive.
- Copy each text span exactly from the transcript.
- Use only the label/schema pairs below.
- Annotate concrete values and explicitly named processor, recipient, or third-party entities.
- Do not annotate speaker labels or category-only phrases without a value.
- Return an empty annotations list when there are no targets.
- Return JSON only, without Markdown or explanation.

Allowed label/schema pairs:
{contract}

Transcript:
{json.dumps(text, ensure_ascii=False)}"""


def _parse(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(cleaned[cleaned.find("{"): cleaned.rfind("}") + 1])


def _to_benchmark(record: dict[str, Any], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """The notebook's normalisation: keep the model's offsets when they match its text,
    else the first unused exact occurrence; labels and ids are not repaired."""
    text, used, out = record["text"], set(), []
    for item in parsed.get("annotations") or []:
        if not isinstance(item, dict):
            continue
        span = str(item.get("text") or "")
        location = None
        try:
            start, end = int(item["start"]), int(item["end"])
            if (start, end) not in used and text[start:end] == span:
                location = (start, end)
        except (KeyError, TypeError, ValueError):
            pass
        if location is None and span:
            pos = text.find(span)
            while pos >= 0 and (pos, pos + len(span)) in used:
                pos = text.find(span, pos + 1)
            location = (pos, pos + len(span)) if pos >= 0 else None
        if location is None:
            continue
        used.add(location)
        out.append({"start": location[0], "end": location[1], "text": text[location[0]:location[1]],
                    "label": str(item.get("label") or ""), "schema_id": str(item.get("schema_id") or "")})
    return out


async def run(args: argparse.Namespace) -> None:
    records = bench.load_split(args.split, args.limit)
    run_dir = bench.RUNS / (args.run or f"direct_llm_{args.model.replace('/', '_')}_{args.split}")
    raw_path = run_dir / "raw.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = {row["id"]: row for row in bench.read_jsonl(raw_path)}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} records, {len(done)} cached, {len(todo)} to run.")
    headers = {"Authorization": f"Bearer {_key()}", "X-Title": "DPVCallBench direct LLM baseline"}
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=180) as client:

        async def one(record: dict[str, Any]) -> None:
            async with sem:
                started = time.monotonic()
                resp = await client.post("/chat/completions", json={
                    "model": args.model, "temperature": 0, "usage": {"include": True},
                    "messages": [{"role": "user", "content": prompt(record["text"])}],
                })
                resp.raise_for_status()
                data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            row = {"id": record["id"], "seconds": round(time.monotonic() - started, 2), "content": content,
                   "prompt_tokens": usage.get("prompt_tokens"), "cost_usd": usage.get("cost")}
            async with lock:
                with raw_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[record["id"]] = row
                print(f"  {record['id']} {row['seconds']}s", flush=True)

        await asyncio.gather(*(one(r) for r in todo))

    predictions = []
    for record in records:
        try:
            parsed = _parse(done[record["id"]]["content"])
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        predictions.append({"id": record["id"], "annotations": _to_benchmark(record, parsed)})
    bench.write_jsonl(run_dir / "gold.jsonl", records)
    bench.write_jsonl(run_dir / "predictions.jsonl", predictions)
    metrics = bench.score(run_dir / "gold.jsonl", run_dir / "predictions.jsonl")
    gold = {(r["id"], a["start"], a["end"]): a["schema_id"] for r in records for a in r["annotations"]}
    matched = [(a["schema_id"] == gold[(p["id"], a["start"], a["end"])]) for p in predictions for a in p["annotations"]
               if (p["id"], a["start"], a["end"]) in gold]
    report = {"metrics": metrics, "matched_spans": {"n": len(matched), "concept_accuracy": round(sum(matched) / max(len(matched), 1), 3)},
              "cost_usd": round(sum(r.get("cost_usd") or 0 for r in done.values()), 4),
              "prompt_tokens_per_record": round(sum(r.get("prompt_tokens") or 0 for r in done.values()) / max(len(done), 1)),
              "per_label": bench.per_label(records, predictions), "errors": bench.errors(records, predictions)}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v["f1"] if isinstance(v, dict) else v) for k, v in metrics.items()}, indent=1))
    print("matched spans:", report["matched_spans"], f"cost ${report['cost_usd']}, {report['prompt_tokens_per_record']} prompt tokens/record")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--run")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
