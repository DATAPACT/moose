"""Rank two-step candidates with Laya and compare with the LLM's choice.

Takes a `dpv_privacy` run (step 1 mentions + retrieved candidates, from run_moose.py),
asks Laya for a probability per candidate, and measures, on mentions that match a gold
span exactly:
  - ranking: gold concept ranked first / in top 3, mean reciprocal rank;
  - calibration: ECE before and after a temperature fitted by 2-fold cross-validation;
  - selective accuracy: accuracy on the share of mentions Laya is most confident about;
  - the LLM's own choice on the same mentions, for comparison.

A candidate counts as correct when it maps to the gold concept (itself or a finer DPV
concept, see bench.collapse_to_benchmark).

    ../.venv-laya/bin/python laya_rank.py --definition-chars 60
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import bench

DEFAULT_RUN = "moose_dpv_privacy_expand_openai_gpt-oss-120b_dev"
TYPES = bench.REPO / "src/moose/data/dpv/subsets/privacy/types.json"


def option_text(record: dict[str, Any], definition_chars: int) -> str:
    definition = (record.get("definition") or "").strip()
    if definition_chars and definition:
        short = definition if len(definition) <= definition_chars else definition[:definition_chars].rsplit(" ", 1)[0] + "…"
        return f"{record['label']}: {short}"
    return record["label"]


def mentions(run_dir: Path) -> list[dict[str, Any]]:
    """Predicted entities that match a gold span exactly, with gold concept and LLM choice."""
    gold = {r["id"]: r for r in bench.read_jsonl(run_dir / "gold.jsonl")}
    out = []
    for row in bench.read_jsonl(run_dir / "raw.jsonl"):
        record = gold[row["id"]]
        spans = {(a["start"], a["end"]): a for a in record["annotations"]}
        for ent in row["entities"]:
            ann = spans.get((ent["start"], ent["end"]))
            if ann and ent.get("candidates"):
                out.append({"id": row["id"], "text": record["text"], "mention": ent["text"], "category": ent.get("category", ""),
                            "candidates": ent["candidates"], "gold": ann["schema_id"], "llm_choice": ent["type_id"]})
    return out


def is_correct(candidate: str, gold: str) -> bool:
    return bench.collapse_to_benchmark(candidate) == gold


def rank(args: argparse.Namespace) -> Path:
    import laya
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    records = {t["id"]: t for t in json.loads(TYPES.read_text(encoding="utf-8"))["types"]}
    agent = laya.load("convaiinnovations/laya", device=device)
    items = mentions(bench.RUNS / args.run)
    out_path = bench.RUNS / args.run / f"laya_rank_def{args.definition_chars}{'_shuffled' if args.shuffle else ''}.jsonl"
    rows, started, truncated = [], time.monotonic(), 0
    rng = random.Random(0)
    for item in items:
        candidates = list(item["candidates"])
        if args.shuffle:
            rng.shuffle(candidates)  # to check for a bias towards the retrieval order
        criteria = {f"option_{i}": option_text(records[c], args.definition_chars) for i, c in enumerate(candidates)}
        state = f'Call transcript: "{item["text"]}"\nMention: "{item["mention"]}" (described as: {item["category"]})'
        question = {"type": "choice", "instructions": "Which category of personal data or privacy role is the mention?",
                    "criteria": criteria}
        result = agent.predict(state, {"category": question})
        usage = result.get("usage") or {}
        truncated += bool(usage.get("truncated") or usage.get("truncated_questions") or usage.get("state_tokens_dropped"))
        probs = result["answers"]["category"]["probabilities"]
        scores = {candidates[int(k.split("_")[1])]: float(p) for k, p in probs.items()}
        rows.append({**item, "laya": scores})
    bench.write_jsonl(out_path, rows)
    print(f"{len(rows)} mentions ranked in {time.monotonic() - started:.0f}s; {truncated} truncated inputs -> {out_path.name}")
    return out_path


def scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    logits = {k: math.log(max(p, 1e-12)) / temperature for k, p in probs.items()}
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def nll(rows: list[dict[str, Any]], temperature: float) -> float:
    total, n = 0.0, 0
    for row in rows:
        probs = scale(row["laya"], temperature)
        mass = sum(p for c, p in probs.items() if is_correct(c, row["gold"]))
        if mass > 0 or any(is_correct(c, row["gold"]) for c in probs):
            total -= math.log(max(mass, 1e-12))
            n += 1
    return total / max(n, 1)


def fit_temperature(rows: list[dict[str, Any]]) -> float:
    return min((round(0.25 * 1.1 ** i, 4) for i in range(60)), key=lambda t: nll(rows, t))


def top1(probs: dict[str, float]) -> tuple[str, float]:
    best = max(probs, key=probs.get)
    return best, probs[best]


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, ok in pairs:
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, ok))
    return sum(len(b) / len(pairs) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in buckets if b)


def report(path: Path) -> dict[str, Any]:
    rows = bench.read_jsonl(path)
    n = len(rows)
    ranks = []
    for row in rows:
        ordered = sorted(row["laya"], key=row["laya"].get, reverse=True)
        ranks.append(next((i + 1 for i, c in enumerate(ordered) if is_correct(c, row["gold"])), None))
    retrievable = [r for r in ranks if r is not None]

    # Temperature by 2-fold cross-validation over records, so no row calibrates itself.
    folds = [[r for r in rows if int(r["id"].split("_")[1]) % 2 == f] for f in (0, 1)]
    calibrated: list[tuple[float, bool]] = []
    temperatures = []
    for f in (0, 1):
        t = fit_temperature(folds[1 - f])
        temperatures.append(t)
        for row in folds[f]:
            choice, confidence = top1(scale(row["laya"], t))
            calibrated.append((confidence, is_correct(choice, row["gold"])))
    raw = [(top1(r["laya"])[1], is_correct(top1(r["laya"])[0], r["gold"])) for r in rows]

    # Ranked by raw confidence: the two folds' temperatures differ, so calibrated
    # confidences are not comparable across folds (temperature does not change the order).
    by_confidence = sorted(raw, key=lambda x: -x[0])
    selective = {}
    for share in (0.25, 0.5, 0.75, 1.0):
        kept = by_confidence[: max(1, round(share * n))]
        selective[f"top_{int(share * 100)}pct"] = round(sum(ok for _, ok in kept) / len(kept), 3)

    llm_ok = [is_correct(r["llm_choice"], r["gold"]) for r in rows]
    laya_ok = [ok for _, ok in raw]
    result = {
        "mentions": n,
        "candidates_per_mention": round(sum(len(r["laya"]) for r in rows) / n, 1),
        "gold_among_candidates": round(len(retrievable) / n, 3),
        "laya_top1": round(sum(r == 1 for r in retrievable) / n, 3),
        "laya_top3": round(sum(r <= 3 for r in retrievable) / n, 3),
        "laya_mrr": round(sum(1 / r for r in retrievable) / n, 3),
        "llm_choice_accuracy": round(sum(llm_ok) / n, 3),
        "both_right": sum(a and b for a, b in zip(laya_ok, llm_ok)),
        "only_llm_right": sum(b and not a for a, b in zip(laya_ok, llm_ok)),
        "only_laya_right": sum(a and not b for a, b in zip(laya_ok, llm_ok)),
        "ece_raw": round(ece(raw), 3),
        "ece_calibrated_cv": round(ece(calibrated), 3),
        "temperatures": temperatures,
        "laya_accuracy_on_most_confident": selective,
    }
    path.with_suffix(".report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--definition-chars", type=int, default=60, help="0 = labels only.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidate order before ranking.")
    args = parser.parse_args()
    print(json.dumps(report(rank(args)), indent=1))


if __name__ == "__main__":
    main()
