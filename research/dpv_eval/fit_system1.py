"""Fit when the System 1 model consults the LLM, and write the schema's calibration file.

Runs on any two-step schema: --schema and --run name a schema and the run (with the LLM
deciding, so every mention has the LLM's choice) whose mentions are scored, for each of
--splits. It also says whether System 1 + LLM should be the schema's default linker
(accuracy within --margin of the LLM's) and whether its probabilities can be marked
validated (ECE at most --max-ece).

For every mention of the two-step runs on train and dev, Laya scores the candidates twice,
with Moose's own inputs (moose.linkers): alone, and with the LLM's choice as a suggestion.
`system1+llm` uses the first score, and the second when the first is below `consult_below`.
This reports accuracy, calibration (ECE, Brier) and how well the probability separates
right from wrong (AUROC) for each threshold, per split and pooled, picks the threshold
with the best pooled accuracy (ties: fewer LLM calls), and writes it for the schema.

Train and dev hold different template families, so a threshold that works on both is a
better sign than one tuned on either. Report on test once the choice is frozen.

    ../.venv-laya/bin/python fit_system1.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

import bench
from linking_data import RUN, SCHEMA, correct, mentions
from moose.linkers import DEFAULT_LAYA_MODEL, laya_scorer
from moose.schema import DATA_DIR, get_schema_config
THRESHOLDS = (0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.01)


def metrics(p: np.ndarray, ok: np.ndarray, bins: int = 10) -> dict[str, float]:
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[edges == b].mean() - ok[edges == b].mean()) * (edges == b).mean() for b in range(bins) if (edges == b).any())
    out = {"accuracy": round(float(ok.mean()), 3), "mean_confidence": round(float(p.mean()), 3),
           "ece": round(float(ece), 3), "brier": round(float(((p - ok) ** 2).mean()), 3)}
    if 0 < ok.mean() < 1:
        out["auroc"] = round(float(roc_auc_score(ok, p)), 3)
    return out


async def scores(split: str, model: str, run: str, schema: str) -> list[dict[str, Any]]:
    """Laya's probabilities alone and with the LLM's suggestion, cached next to the run."""
    cache = bench.RUNS / run.format(split=split) / f"system1_scores_{model.replace('/', '_')}.jsonl"
    items = mentions(split, run, schema)
    cached = bench.read_jsonl(cache)
    if len(cached) == len(items):
        return cached
    scorer = laya_scorer(model)
    rows = []
    for item in items:
        label = next((c["label"] for c in item["mention"]["candidates"] if c["id"] == item["llm_choice"]), None)
        [plain] = await scorer.score(item["text"], [item["mention"]])
        hinted = (await scorer.score(item["text"], [item["mention"]], [label]))[0] if label else None
        rows.append({"id": item["id"], "gold": item["gold"], "llm_choice": item["llm_choice"], "plain": plain, "hinted": hinted})
    bench.write_jsonl(cache, rows)
    return rows


def system1_llm(rows: list[dict[str, Any]], threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
    p, ok, asked = [], [], 0
    for r in rows:
        s = r["plain"]
        if max(s.values()) < threshold and r["hinted"]:
            s, asked = r["hinted"], asked + 1
        best = max(s, key=s.get)
        p.append(s[best])
        ok.append(correct(best, r["gold"]))
    return np.array(p), np.array(ok, dtype=float), asked / len(rows)


async def main_async(args: argparse.Namespace) -> None:
    by_split = {split: await scores(split, args.model, args.run, args.schema) for split in args.splits}
    if len(args.splits) > 1:
        by_split["pooled"] = [r for split in args.splits for r in by_split[split]]
    report: dict[str, Any] = {"schema": args.schema, "run": args.run, "model": args.model}
    for name, rows in by_split.items():
        llm_ok = np.array([correct(r["llm_choice"], r["gold"]) for r in rows], dtype=float)
        report[name] = {"mentions": len(rows), "llm alone accuracy": round(float(llm_ok.mean()), 3)}
        for threshold in THRESHOLDS:
            p, ok, share = system1_llm(rows, threshold)
            report[name][f"consult below {threshold}"] = {**metrics(p, ok), "llm_consulted": round(share, 3)}

    fit = by_split["pooled" if len(args.splits) > 1 else args.splits[0]]
    best = max(THRESHOLDS[:-1], key=lambda t: (system1_llm(fit, t)[1].mean(), -system1_llm(fit, t)[2]))
    p, ok, _ = system1_llm(fit, best)
    chosen = metrics(p, ok)
    llm_accuracy = float(np.mean([correct(r["llm_choice"], r["gold"]) for r in fit]))
    default_system1 = chosen["accuracy"] >= llm_accuracy - args.margin
    validated = chosen["ece"] <= args.max_ece
    calibration = {
        "model": args.model,
        "consult_below": best,
        "validated_for": ["text"] if validated else [],
        "fitted_on": f"DPVCallBench v0.1 {'+'.join(args.splits)} ({len(fit)} mentions with exact gold spans), "
                     f"{args.schema} two-step candidates, LLM openai/gpt-oss-120b; research/dpv_eval/fit_system1.py. "
                     f"At this threshold: accuracy {chosen['accuracy']}, ECE {chosen['ece']}; LLM alone {round(llm_accuracy, 3)}.",
    }
    report["chosen"] = {**calibration, "recommended_default_linker": "system1+llm" if default_system1 else "llm"}
    (bench.RUNS / f"system1_report_{args.schema}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out:
        out = DATA_DIR / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    for name, rows in by_split.items():
        print(name, "LLM alone", report[name]["llm alone accuracy"])
        for threshold in THRESHOLDS:
            print(f"  consult below {threshold}:", report[name][f"consult below {threshold}"])
    print("chosen:", report["chosen"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_LAYA_MODEL, help="Laya repo id or a fine-tuned checkpoint folder.")
    parser.add_argument("--schema", default=SCHEMA)
    parser.add_argument("--run", default=RUN, help="Run folder under runs/, with {split} for the split.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--out", help="Calibration file to write, relative to src/moose/data.")
    parser.add_argument("--margin", type=float, default=0.02, help="Accuracy System 1 + LLM may give up against the LLM and still be the default.")
    parser.add_argument("--max-ece", type=float, default=0.10, help="Largest ECE for probabilities to count as validated.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
