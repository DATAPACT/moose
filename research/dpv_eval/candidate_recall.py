"""How often is each gold DPV concept among the candidates picked for its text?

Candidates are chosen per text (Moose annotates a whole text in one call), from the
privacy pool: DPV personal-data concepts, legal roles and data-subject types.

    ../.venv-laya/bin/python candidate_recall.py --split dev
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

import bench

CONCEPTS = bench.REPO / "src/moose/data/dpv/concepts.json"
TOKEN = re.compile(r"[a-z0-9]+")


def privacy_pool() -> list[dict]:
    pool = []
    for c in json.loads(CONCEPTS.read_text(encoding="utf-8")):
        cid, modules = c.get("id") or "", set(c.get("modules") or [])
        if not c.get("label") or "serialisation" in cid or cid.endswith("-classes"):
            continue
        if cid.startswith("dpv-pd:") or (cid.startswith("dpv:") and modules & {"entities_legalrole", "entities_datasubject"}):
            pool.append(c)
    return pool


def concept_text(c: dict) -> str:
    alt = ", ".join(c.get("alt_labels") or [])
    return f"{c['label']}{f' ({alt})' if alt else ''}: {c.get('definition') or ''}"


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 2]


class Lexical:
    """BM25 over concept label, alternative labels and definition. No dependencies."""

    def __init__(self, pool: list[dict]) -> None:
        self.docs = [Counter(tokens(concept_text(c)) + tokens(c["label"]) * 2) for c in pool]
        self.avg = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def top(self, text: str, n: int) -> list[int]:
        query = set(tokens(text))
        scores = []
        for i, d in enumerate(self.docs):
            length = sum(d.values())
            s = sum(self.idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * length / self.avg)) for t in query if t in d)
            scores.append((s, i))
        return [i for s, i in sorted(scores, reverse=True)[:n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    records = bench.load_split(args.split)
    pool = privacy_pool()
    ids = [c["id"] for c in pool]
    gold_ids = {a["schema_id"] for r in records for a in r["annotations"]}
    print(f"pool: {len(pool)} concepts; benchmark concepts missing from pool: {sorted(gold_ids - set(ids))}")

    lexical = Lexical(pool)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    matrix = model.encode([concept_text(c) for c in pool], normalize_embeddings=True, convert_to_tensor=True)
    query = model.encode(["Represent this sentence for searching relevant passages: " + r["text"] for r in records],
                         normalize_embeddings=True, convert_to_tensor=True)
    dense_rank = (query @ matrix.T).argsort(dim=1, descending=True).tolist()

    out = {}
    for n in (10, 20, 40, 80):
        for name, rank in (("lexical", [lexical.top(r["text"], n) for r in records]), ("embedding", [x[:n] for x in dense_rank])):
            hit = total = 0
            for record, idx in zip(records, rank):
                cands = {ids[i] for i in idx[:n]}
                for a in record["annotations"]:
                    total += 1
                    hit += a["schema_id"] in cands
            out[f"{name}@{n}"] = round(hit / total, 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
