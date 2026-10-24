"""Fully local DPV annotation: GLiNER finds spans, then optionally Laya links them.

Systems:
  gliner       GLiNER alone, zero-shot with the 21 benchmark labels
  gliner+laya  GLiNER spans; candidate retrieval + Laya choose the concept (or "none")

The GLiNER threshold is chosen on train; results are reported on --split (dev).

    ../.venv-laya/bin/python local_pipeline.py --split dev
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import bench
import laya_link

GLINER_MODEL = "urchade/gliner_multi_pii-v1"


def gliner_labels() -> dict[str, str]:
    """GLiNER prompt text -> benchmark label."""
    return {item["label"].replace("_", " "): item["label"] for item in bench.labels()}


def detect(model: Any, records: list[dict[str, Any]], threshold: float) -> list[list[dict[str, Any]]]:
    names = gliner_labels()
    out = []
    for record in records:
        found = model.predict_entities(record["text"], list(names), threshold=threshold, flat_ner=True)
        out.append([{"start": e["start"], "end": e["end"], "label": names[e["label"]], "score": e["score"]} for e in found])
    return out


def as_predictions(records: list[dict[str, Any]], spans: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    to_schema = bench.label_to_schema()
    return [{"id": r["id"], "annotations": [{"start": s["start"], "end": s["end"], "text": r["text"][s["start"]:s["end"]],
                                             "label": s["label"], "schema_id": to_schema[s["label"]]} for s in found]}
            for r, found in zip(records, spans)]


def score_predictions(name: str, records: list[dict[str, Any]], predictions: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    bench.write_jsonl(out_dir / "gold.jsonl", records)
    bench.write_jsonl(out_dir / f"{name}_predictions.jsonl", predictions)
    metrics = bench.score(out_dir / "gold.jsonl", out_dir / f"{name}_predictions.jsonl")
    return {"metrics": metrics, "per_label": bench.per_label(records, predictions)}


def relink(agent: Any, retriever: laya_link.Retriever, records: list[dict[str, Any]],
           spans: list[list[dict[str, Any]]], k: int, none_margin: float) -> list[dict[str, Any]]:
    """Replace GLiNER's labels by Laya's choice among retrieved labels, dropping "none"."""
    to_schema = bench.label_to_schema()
    predictions = []
    for record, found in zip(records, spans):
        anns = []
        hits = retriever.top_k([laya_link.mention_query(record["text"], s["start"], s["end"]) for s in found], k) if found else []
        for span, idx in zip(found, hits):
            options = [retriever.entries[i] for i in idx]
            probs, _ = laya_link.choose(agent, laya_link.laya_state(record["text"], span["start"], span["end"]), options, True)
            best = max((key for key in probs if key != laya_link.NONE), key=probs.get)
            if probs.get(laya_link.NONE, 0.0) > probs[best] + none_margin:
                continue
            anns.append({"start": span["start"], "end": span["end"], "text": record["text"][span["start"]:span["end"]],
                         "label": best, "schema_id": to_schema[best], "confidence": probs[best]})
        predictions.append({"id": record["id"], "annotations": anns})
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import laya
    import torch
    from gliner import GLiNER

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GLiNER.from_pretrained(GLINER_MODEL)
    out_dir = bench.RUNS / "local_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Threshold chosen on train (exact span + label F1), never on the reported split.
    train = bench.load_split("train", args.limit)
    best_threshold, best_f1, sweep = 0.5, -1.0, {}
    train_spans = detect(model, train, 0.1)
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [[s for s in found if s["score"] >= threshold] for found in train_spans]
        f1 = score_predictions("train_sweep", train, as_predictions(train, kept), out_dir)["metrics"]["exact_annotation_triple"]["f1"]
        sweep[threshold] = round(f1, 4)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    print(f"GLiNER threshold sweep on train: {sweep} -> {best_threshold}")

    records = bench.load_split(args.split, args.limit)
    started = time.monotonic()
    spans = detect(model, records, best_threshold)
    gliner_seconds = (time.monotonic() - started) / len(records)
    report: dict[str, Any] = {"split": args.split, "gliner_model": GLINER_MODEL, "threshold": best_threshold,
                              "threshold_sweep_train": sweep, "gliner_seconds_per_record": round(gliner_seconds, 3)}
    report["gliner"] = score_predictions("gliner", records, as_predictions(records, spans), out_dir)

    agent = laya.load("convaiinnovations/laya", device=device)
    retriever = laya_link.Retriever(laya_link.pool_entries("labels"), device)
    started = time.monotonic()
    linked = relink(agent, retriever, records, spans, args.k, none_margin=0.0)
    report["laya_seconds_per_record"] = round((time.monotonic() - started) / len(records), 3)
    report["gliner+laya"] = score_predictions("gliner_laya", records, linked, out_dir)

    (out_dir / f"report_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("gliner", "gliner+laya"):
        m = report[name]["metrics"]
        print(name, {k: round(v["f1"], 3) for k, v in m.items() if isinstance(v, dict)}, "pred", m["predicted_annotations"])


if __name__ == "__main__":
    main()
