"""Laya's probability for anonymisation's masking decisions, checked against TAB.

For each entity of a cached `tab.py --mode anonymise` run (its first mention, as TAB's
annotators decide on the first mention), Laya is asked a yes/no question: must this be
masked to protect the applicant? Asked twice: alone ("plain") and told what the LLM decided
("hinted"), as the system1+llm linker does for types.

Gold: the first mention overlaps a mention TAB masks (DIRECT or QUASI, first annotator).
Mentions TAB did not annotate are not masked. Reported:
  - AUC of P(mask) against gold: does Laya know what to mask at all?
  - the LLM's decision with Laya's probability that it is right (P(mask) if masked, else
    1 - P(mask)): accuracy and ECE, as a confidence Moose would report.

Scores are cached next to the run (laya_mask.jsonl); no LLM calls.

    PYTHONPATH=../../src ../.venv-laya/bin/python laya_mask.py runs/anonymise_test_openai_gpt-oss-120b --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moose.anonymise import MASKED, laya_mask_inputs
from moose.linkers import DEFAULT_LAYA_MODEL

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCORES = "laya_mask.jsonl"
PROTECT = "the applicant (the person who brought the case before the Court)"


def first_mentions(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for e in sorted(entities, key=lambda e: e["start"]):
        seen.setdefault(e["entity"], e)
    return list(seen.values())


def gold_masked(doc: dict[str, Any], e: dict[str, Any]) -> bool:
    mentions = next(iter(doc["annotations"].values()))["entity_mentions"]
    return any(m["identifier_type"] in MASKED and m["start_offset"] < e["end"] and e["start"] < m["end_offset"] for m in mentions)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.array([r["gold"] for r in rows], dtype=float)
    llm = np.array([r["llm_masked"] for r in rows], dtype=float)
    out: dict[str, Any] = {"entities": len(rows), "gold_masked": round(float(y.mean()), 3),
                           "llm_accuracy": round(float((llm == y).mean()), 3)}
    for variant in ("plain", "hinted"):
        p = np.array([r[variant] for r in rows])
        conf = np.where(llm == 1, p, 1 - p)
        right = (llm == y).astype(float)
        out[variant] = {"auc_p_mask": round(auc(p, y), 3), "laya_accuracy": round(float(((p >= 0.5) == y).mean()), 3),
                        "mean_confidence_llm": round(float(conf.mean()), 3), "ece_llm": round(ece(conf, right), 3)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a tab.py --mode anonymise run folder")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--model", default=DEFAULT_LAYA_MODEL)
    args = parser.parse_args()

    docs = {d["doc_id"]: d for d in json.loads((DATA / f"echr_{args.split}.json").read_text(encoding="utf-8"))}
    runs = [json.loads(line) for line in (args.run / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    cache = args.run / SCORES
    done = {(r["doc_id"], r["entity"]) for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else set()

    import laya

    agent = None
    for entry in runs:
        doc = docs[entry["doc_id"]]
        todo = [e for e in first_mentions(entry["entities"]) if (entry["doc_id"], e["entity"]) not in done]
        if not todo:
            continue
        agent = agent or laya.load(args.model)
        with cache.open("a", encoding="utf-8") as f:
            for e in todo:
                masked = e["identifier_type"] in MASKED
                row = {"doc_id": entry["doc_id"], "entity": e["entity"], "text": e["text"], "category": e["category"],
                       "llm_masked": masked, "gold": gold_masked(doc, e)}
                for variant, hint in (("plain", None), ("hinted", masked)):
                    state, questions = laya_mask_inputs(doc["text"], e, PROTECT, hint)
                    row[variant] = float(agent.predict(state, questions)["answers"]["mask"]["probabilities"]["yes"])
                f.write(json.dumps(row) + "\n")
        print(f"  {entry['doc_id']}: {len(todo)} entities", flush=True)

    rows = [r for r in map(json.loads, cache.read_text().splitlines()) if r["doc_id"] in {e["doc_id"] for e in runs}]
    print(json.dumps(report(rows), indent=2))


if __name__ == "__main__":
    main()
