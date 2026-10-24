"""End-to-end score of System 1 decisions on a two-step run.

Takes a run where the LLM decided (run_moose.py --linker llm): its mentions, candidates and
the LLM's choices. Laya then decides each mention as Moose would, with moose.linkers' own
inputs and confidence: alone (`--consult-below 0`) or consulting the LLM below a threshold
(its suggestion added to Laya's input). Writes predictions and scores them with the
benchmark's evaluator, plus calibration on spans that match gold.

Mentions the LLM rejected ("none") are not in the run, so they stay out here too.

    ../.venv-laya/bin/python system1_eval.py --run moose_dpv_privacy_llm_spanrules_openai_gpt-oss-120b_dev \\
        --schema dpv_privacy --model ../models/laya-dpv-privacy-v1 --consult-below 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np

import bench
from compare_linkers import metrics
from moose.linkers import Calibration, laya_scorer
from moose.schema import get_schema_config


async def main_async(args: argparse.Namespace) -> None:
    run_dir = bench.RUNS / args.run
    gold = bench.read_jsonl(run_dir / "gold.jsonl")
    raw = {r["id"]: r for r in bench.read_jsonl(run_dir / "raw.jsonl")}
    records = {r["id"]: r for r in get_schema_config(args.schema).load_type_records()}
    calibration = Calibration(model=args.model, consult_below=args.consult_below)
    scorer = laya_scorer(args.model)
    to_label = bench.schema_to_label()

    predictions, pairs, consulted, n = [], [], 0, 0
    gold_spans = {(g["id"], a["start"], a["end"]): a["schema_id"] for g in gold for a in g["annotations"]}
    for record in gold:
        entities = [e for e in raw[record["id"]]["entities"] if e.get("candidates")]
        items = [{"text": e["text"], "category": e.get("category"),
                  "candidates": [records[c] for c in e["candidates"] if c in records]} for e in entities]
        keep = [i for i, item in enumerate(items) if item["candidates"]]
        entities, items = [entities[i] for i in keep], [items[i] for i in keep]
        plain = await scorer.score(record["text"], items) if items else []
        annotations = []
        for entity, item, probs in zip(entities, items, plain):
            n += 1
            best = max(probs, key=probs.get)
            confidence = probs[best]
            if confidence < calibration.consult_below:
                label = next((c["label"] for c in item["candidates"] if c["id"] == entity["type_id"]), None)
                if label:
                    consulted += 1
                    [probs] = await scorer.score(record["text"], [item], [label])
                    best = max(probs, key=probs.get)
                    confidence = probs[best]
            schema_id = bench.collapse_to_benchmark(best)
            annotations.append({"start": entity["start"], "end": entity["end"], "text": entity["text"],
                                "label": to_label.get(schema_id, ""), "schema_id": schema_id})
            key = (record["id"], entity["start"], entity["end"])
            if key in gold_spans:
                pairs.append((confidence, schema_id == gold_spans[key]))
        predictions.append({"id": record["id"], "annotations": annotations})

    name = args.name or f"system1_c{args.consult_below}"
    pred_path = run_dir / f"{name}_predictions.jsonl"
    bench.write_jsonl(pred_path, predictions)
    scores = bench.score(run_dir / "gold.jsonl", pred_path)
    report = {"run": args.run, "model": args.model, "consult_below": args.consult_below,
              "metrics": scores, "llm_consulted": round(consulted / max(n, 1), 3),
              "matched_spans": {"n": len(pairs), **metrics(np.array([p for p, _ in pairs]), np.array([ok for _, ok in pairs], float))}}
    (run_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["f1"] if isinstance(v, dict) and "f1" in v else v) for k, v in scores.items()}))
    print("matched spans:", report["matched_spans"], "| LLM consulted:", report["llm_consulted"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--consult-below", type=float, default=0.0)
    parser.add_argument("--name")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
