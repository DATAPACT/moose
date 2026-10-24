"""Compare Laya checkpoints and the LLM as step-2 linkers on one split's mentions.

For each checkpoint (zero-shot or fine-tuned), on the mentions of a two-step run that
match a gold span exactly:
  - accuracy of Laya's choice, its calibration (ECE, Brier) and how well its probability
    separates right from wrong (AUROC);
  - the hybrid curve: Laya decides mentions at or above a probability, the LLM the rest;
    accuracy and share of mentions sent to the LLM at each threshold.
The LLM's own choice and stated confidence are reported alongside.

Fine-tuned checkpoints apply their own fitted temperatures, so their probabilities are
used as they come.

    ../.venv-laya/bin/python compare_linkers.py --split dev --models convaiinnovations/laya ../models/laya-dpv-privacy-train-soft-ce
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import bench
from linking_data import correct, mentions
from moose.linkers import LayaScorer


def metrics(p: np.ndarray, ok: np.ndarray, bins: int = 10) -> dict[str, float]:
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[edges == b].mean() - ok[edges == b].mean()) * (edges == b).mean() for b in range(bins) if (edges == b).any())
    out = {"accuracy": round(float(ok.mean()), 3), "mean_confidence": round(float(p.mean()), 3),
           "ece": round(float(ece), 3), "brier": round(float(((p - ok) ** 2).mean()), 3)}
    if 0 < ok.mean() < 1:
        out["auroc"] = round(float(roc_auc_score(ok, p)), 3)
    return out


async def main_async(args: argparse.Namespace) -> None:
    items = mentions(args.split)
    llm_ok = np.array([correct(i["llm_choice"], i["gold"]) for i in items], dtype=float)
    report = {"split": args.split, "mentions": len(items),
              "llm": metrics(np.array([i["llm_stated"] for i in items]), llm_ok)}
    for model in args.models:
        scorer = LayaScorer(model)
        scores = [(await scorer.score(i["text"], [i["mention"]]))[0] for i in items]
        top = [max(s, key=s.get) for s in scores]
        p = np.array([s[t] for s, t in zip(scores, top)])
        ok = np.array([correct(t, i["gold"]) for t, i in zip(top, items)], dtype=float)
        curve = []
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            use = p >= threshold
            curve.append({"threshold": threshold, "laya_decides": round(float(use.mean()), 3),
                          "accuracy": round(float(np.where(use, ok, llm_ok).mean()), 3),
                          "laya_accuracy_on_its_share": round(float(ok[use].mean()), 3) if use.any() else None})
        report[Path(model).name if Path(model).exists() else model] = {**metrics(p, ok), "hybrid": curve}
        print(model, json.dumps(report[Path(model).name if Path(model).exists() else model])[:300], flush=True)
    out = bench.RUNS / f"compare_linkers_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--models", nargs="+", default=["convaiinnovations/laya"])
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
