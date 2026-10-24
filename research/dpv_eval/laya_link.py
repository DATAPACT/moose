"""Link mentions to DPV concepts with candidate retrieval + Laya (a "System 1" choice model).

For each mention: embed it in context, retrieve the top-k candidate concepts, and let
Laya choose among them (plus "none" when spans may be wrong). Probabilities are kept so
a temperature can be fitted for calibration.

Pools:
  labels  the 21 DPVCallBench labels (closed set)
  dpv     every DPV concept with a definition (open set, 9k+), scored after mapping
          predictions up to the benchmark's concepts

    ../.venv-laya/bin/python laya_link.py --pool labels --split dev --k 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import bench

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NONE = "none"
NONE_DESCRIPTION = "Not personal data or a privacy role; the mention is a cue word or ordinary text."


def pool_entries(pool: str) -> list[dict[str, str]]:
    """Candidates as {key, schema_id, label, text}: key is what Laya sees as the option."""
    if pool == "labels":
        return [{"key": item["label"], "schema_id": item["schema_id"], "label": item["label"],
                 "text": f"{item['label'].replace('_', ' ')}: {item['definition']}"} for item in bench.labels()]
    concepts = json.loads((bench.REPO / "src/moose/data/dpv/concepts.json").read_text(encoding="utf-8"))
    entries, seen = [], set()
    for concept in concepts:
        cid, label, definition = concept.get("id"), concept.get("label"), concept.get("definition")
        if not (cid and label and definition) or cid in seen:
            continue
        seen.add(cid)
        entries.append({"key": cid, "schema_id": cid, "label": label, "text": f"{label}: {definition}"})
    return entries


class Retriever:
    def __init__(self, entries: list[dict[str, str]], device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.entries = entries
        self.model = SentenceTransformer(EMBED_MODEL, device=device)
        self.matrix = self.model.encode([e["text"] for e in entries], batch_size=128, normalize_embeddings=True,
                                        convert_to_tensor=True, show_progress_bar=False)

    def top_k(self, queries: list[str], k: int) -> list[list[int]]:
        q = self.model.encode([QUERY_PREFIX + s for s in queries], batch_size=64, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=False)
        scores = q @ self.matrix.T
        return scores.topk(min(k, len(self.entries)), dim=1).indices.tolist()


def mention_query(text: str, start: int, end: int) -> str:
    return f'"{text[start:end]}" in: {text}'


def laya_state(text: str, start: int, end: int) -> str:
    return f'Call transcript: "{text}"\nMention: "{text[start:end]}"'


def choose(agent: Any, state: str, options: list[dict[str, str]], with_none: bool) -> tuple[dict[str, float], bool]:
    # Laya scores option keys; short keys keep the shared option budget for descriptions.
    criteria = {f"option_{i}": e["text"][:160] for i, e in enumerate(options)}
    if with_none:
        criteria[NONE] = NONE_DESCRIPTION
    question = {"type": "choice", "instructions": "Which category of personal data or privacy role is the mention?",
                "criteria": criteria}
    result = agent.predict(state, {"category": question})
    probs = result["answers"]["category"]["probabilities"]
    usage = result.get("usage") or {}
    truncated = bool(usage.get("truncated") or usage.get("truncated_questions") or usage.get("state_tokens_dropped"))
    return {options[int(key.split("_")[1])]["key"] if key != NONE else NONE: float(p) for key, p in probs.items()}, truncated


def mentions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": r["id"], "text": r["text"], **{k: a[k] for k in ("start", "end", "label", "schema_id")}}
            for r in records for a in r["annotations"]]


def run(args: argparse.Namespace) -> Path:
    import laya
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    entries = pool_entries(args.pool)
    by_key = {e["key"]: e for e in entries}
    started = time.monotonic()
    retriever = Retriever(entries, device)
    agent = laya.load("convaiinnovations/laya", device=device)
    print(f"{len(entries)} candidates; models loaded in {time.monotonic() - started:.0f}s on {device}")

    out_dir = bench.RUNS / f"laya_{args.pool}_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        items = mentions(bench.load_split(split, args.limit))
        hits = retriever.top_k([mention_query(m["text"], m["start"], m["end"]) for m in items], args.k)
        rows, started = [], time.monotonic()
        for m, idx in zip(items, hits):
            options = [entries[i] for i in idx]
            probs, truncated = choose(agent, laya_state(m["text"], m["start"], m["end"]), options, args.with_none)
            gold_key = m["label"] if args.pool == "labels" else m["schema_id"]
            rows.append({**{k: m[k] for k in ("id", "start", "end", "label", "schema_id")},
                         "candidates": [e["key"] for e in options],
                         "gold_in_candidates": any(bench.collapse_to_benchmark(by_key[e["key"]]["schema_id"]) == m["schema_id"]
                                                   if args.pool == "dpv" else e["key"] == gold_key for e in options),
                         "probabilities": probs, "truncated": truncated})
        bench.write_jsonl(out_dir / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} mentions in {time.monotonic() - started:.0f}s")
    return out_dir


def predicted_schema(row: dict[str, Any], pool: str, probs: dict[str, float]) -> str:
    key = max(probs, key=probs.get)
    if key == NONE:
        return NONE
    schema_id = bench.label_to_schema()[key] if pool == "labels" else key
    return bench.collapse_to_benchmark(schema_id)


def scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    logits = {k: math.log(max(p, 1e-12)) / temperature for k, p in probs.items()}
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def nll(rows: list[dict[str, Any]], pool: str, temperature: float) -> float:
    """Negative log-likelihood of the correct option (rows where it was retrieved)."""
    total, n = 0.0, 0
    for row in rows:
        target = row["label"] if pool == "labels" else None
        probs = scale(row["probabilities"], temperature)
        if target is None:
            mass = sum(p for k, p in probs.items() if k != NONE and bench.collapse_to_benchmark(k) == row["schema_id"])
        else:
            mass = probs.get(target, 0.0)
        if row["gold_in_candidates"]:
            total -= math.log(max(mass, 1e-12))
            n += 1
    return total / max(n, 1)


def fit_temperature(rows: list[dict[str, Any]], pool: str) -> float:
    grid = [round(0.25 * 1.1 ** i, 4) for i in range(60)]
    return min(grid, key=lambda t: nll(rows, pool, t))


def ece(rows: list[dict[str, Any]], pool: str, temperature: float, bins: int = 10) -> float:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row in rows:
        probs = scale(row["probabilities"], temperature)
        confidence = max(probs.values())
        correct = predicted_schema(row, pool, probs) == row["schema_id"]
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    n = sum(len(b) for b in buckets)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in buckets if b)


def report(out_dir: Path, pool: str) -> dict[str, Any]:
    train, dev = bench.read_jsonl(out_dir / "train.jsonl"), bench.read_jsonl(out_dir / "dev.jsonl")
    temperature = fit_temperature(train, pool) if train else 1.0
    result: dict[str, Any] = {"pool": pool, "temperature_fitted_on_train": temperature}
    for name, rows in (("train", train), ("dev", dev)):
        if not rows:
            continue
        correct = [predicted_schema(r, pool, r["probabilities"]) == r["schema_id"] for r in rows]
        result[name] = {
            "mentions": len(rows),
            "retrieval_recall_at_k": round(sum(r["gold_in_candidates"] for r in rows) / len(rows), 4),
            "accuracy": round(sum(correct) / len(rows), 4),
            "accuracy_when_retrieved": round(sum(c for c, r in zip(correct, rows) if r["gold_in_candidates"])
                                             / max(sum(r["gold_in_candidates"] for r in rows), 1), 4),
            "truncated_inputs": sum(bool(r.get("truncated")) for r in rows),
            "chose_none": sum(predicted_schema(r, pool, r["probabilities"]) == NONE for r in rows),
            "ece_raw": round(ece(rows, pool, 1.0), 4),
            "ece_calibrated": round(ece(rows, pool, temperature), 4),
        }
        per_label: dict[str, list[bool]] = {}
        for ok, r in zip(correct, rows):
            per_label.setdefault(r["label"], []).append(ok)
        result[name]["accuracy_per_label"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(per_label.items())}
    (out_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=["labels", "dpv"], default="labels")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--limit", type=int, help="Records per split (for quick checks).")
    parser.add_argument("--with-none", action="store_true", help="Add a 'none' option (for predicted spans).")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out_dir = bench.RUNS / f"laya_{args.pool}_k{args.k}"
    if not args.report_only:
        out_dir = run(args)
    print(json.dumps(report(out_dir, args.pool), indent=1))


if __name__ == "__main__":
    main()
