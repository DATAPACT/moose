"""DPVCallBench helpers: load splits, map predictions to the benchmark, score runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "research" / "DPVCallBench"
SCHEMA_PATH = BENCH / "schema" / "dpvcallbench_schema_v0_1.json"
SPLITS = BENCH / "data" / "splits" / "v0.1"
RUNS = Path(__file__).resolve().parent / "runs"
DPV_ANCESTORS = REPO / "src" / "moose" / "data" / "dpv" / "concept_ancestors.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_split(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(SPLITS / f"{split}.jsonl")
    return records[:limit] if limit else records


def label_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def labels() -> list[dict[str, str]]:
    return label_schema()["labels"]


def label_to_schema() -> dict[str, str]:
    return {item["label"]: item["schema_id"] for item in labels()}


def schema_to_label() -> dict[str, str]:
    """Most frequent training label per DPV concept; several labels share one concept."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in load_split("train"):
        for ann in record["annotations"]:
            counts[ann["schema_id"]][ann["label"]] += 1
    mapping = {item["schema_id"]: item["label"] for item in labels()}
    mapping.update({schema_id: c.most_common(1)[0][0] for schema_id, c in counts.items()})
    return mapping


def collapse_to_benchmark(schema_id: str) -> str:
    """Map a DPV concept to the nearest benchmark concept among itself and its ancestors.

    This is an adapter-added mapping (see the evaluation protocol): report it next to,
    not instead of, the raw score."""
    targets = {item["schema_id"] for item in labels()}
    if schema_id in targets:
        return schema_id
    for ancestor in _ancestors().get(schema_id, []):
        if ancestor in targets:
            return ancestor
    return schema_id


_ANCESTORS: dict[str, list[str]] | None = None


def _ancestors() -> dict[str, list[str]]:
    global _ANCESTORS
    if _ANCESTORS is None:
        _ANCESTORS = json.loads(DPV_ANCESTORS.read_text(encoding="utf-8"))
    return _ANCESTORS


def _evaluator():
    spec = importlib.util.spec_from_file_location("dpvcallbench_evaluate", BENCH / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


def score(gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    return _evaluator().evaluate(gold_path, predictions_path, SCHEMA_PATH)


def per_label(gold: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Exact span + label + schema precision/recall/F1 per label.

    The label must match too: several labels share one DPV concept (e.g. Identifier)."""
    pred_by_id = {row["id"]: row["annotations"] for row in predictions}
    tp: Counter[str] = Counter()
    n_gold: Counter[str] = Counter()
    n_pred: Counter[str] = Counter()
    for record in gold:
        predicted = {(a["start"], a["end"], a["label"], a["schema_id"]) for a in pred_by_id.get(record["id"], [])}
        for ann in record["annotations"]:
            n_gold[ann["label"]] += 1
            if (ann["start"], ann["end"], ann["label"], ann["schema_id"]) in predicted:
                tp[ann["label"]] += 1
        for ann in pred_by_id.get(record["id"], []):
            n_pred[ann["label"]] += 1
    out = {}
    for label in sorted(set(n_gold) | set(n_pred)):
        p = tp[label] / n_pred[label] if n_pred[label] else 0.0
        r = tp[label] / n_gold[label] if n_gold[label] else 0.0
        out[label] = {
            "gold": n_gold[label],
            "predicted": n_pred[label],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        }
    return out


def errors(gold: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """False negatives, false positives and schema confusions on exact spans."""
    pred_by_id = {row["id"]: row["annotations"] for row in predictions}
    out: dict[str, list[dict[str, Any]]] = {"missed": [], "spurious": [], "wrong_concept": []}
    for record in gold:
        gold_spans = {(a["start"], a["end"]): a for a in record["annotations"]}
        pred_spans = {(a["start"], a["end"]): a for a in pred_by_id.get(record["id"], [])}
        for span, ann in gold_spans.items():
            if span not in pred_spans:
                out["missed"].append({"id": record["id"], "text": ann["text"], "gold": ann["schema_id"]})
            elif pred_spans[span]["schema_id"] != ann["schema_id"]:
                out["wrong_concept"].append(
                    {"id": record["id"], "text": ann["text"], "gold": ann["schema_id"], "predicted": pred_spans[span]["schema_id"]}
                )
        for span, ann in pred_spans.items():
            if span not in gold_spans:
                out["spurious"].append({"id": record["id"], "text": ann["text"], "predicted": ann["schema_id"]})
    return out
