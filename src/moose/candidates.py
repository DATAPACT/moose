"""Retrieve candidate types for a mention from a schema's labels and definitions.

Used by two-step annotation: instead of listing every type id in the prompt, each
mention gets a few candidates. BM25 over each type's label, alternative labels and
definition; no model or extra dependency needed. When types have parents, the best
hits are widened with their direct parents and children, since a description often
matches a broader or narrower type than the one that fits (e.g. "medical condition"
matches MedicalHealth best, while Health is the fitting child).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from moose.schema import SchemaConfig

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the and for with that this from into are was were has have had any not but can its his her their our your "
    "which who whom whose such other than then also about over under via per "
    "of to in is on at by or an as be it we us he me my no so do if up".split()
)
# BM25 parameters (the usual defaults).
_K1 = 1.2
_B = 0.75
# How many of the best hits get their direct parents and children added.
_EXPAND_TOP = 3


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _split_camel(type_id: str) -> str:
    # "dpv-pd:PaymentCardNumber" -> "Payment Card Number"
    local = type_id.split(":", 1)[-1]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", local)


class CandidateIndex:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self._by_id = {r["id"]: r for r in records}
        self._children: dict[str, list[str]] = {}
        for record in records:
            for parent in record.get("parents") or []:
                self._children.setdefault(parent, []).append(record["id"])
        self._docs: list[Counter[str]] = []
        for record in records:
            names = " ".join([record.get("label") or "", _split_camel(record["id"]), *record.get("alt_labels", [])])
            # Names count twice: they describe the type more directly than its definition.
            self._docs.append(Counter(tokens(names) * 2 + tokens(record.get("definition") or "")))
        self._avg_len = sum(sum(d.values()) for d in self._docs) / max(len(self._docs), 1)
        df = Counter(t for d in self._docs for t in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top_k(self, query: str, k: int) -> list[dict[str, Any]]:
        terms = set(tokens(query))
        scored = []
        for i, doc in enumerate(self._docs):
            length = sum(doc.values())
            score = 0.0
            for t in terms & doc.keys():
                tf = doc[t]
                score += self._idf[t] * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / self._avg_len))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if len(self.records) <= k:
            # A small schema: every type is a candidate, best matches first.
            matched = {i for _, i in scored}
            return [self.records[i] for _, i in scored] + [r for i, r in enumerate(self.records) if i not in matched]
        hits = [self.records[i] for _, i in scored[:k]]
        return hits + self._neighbours(hits, k)

    def _neighbours(self, hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Direct parents and children of the best hits that are not hits already."""
        seen = {h["id"] for h in hits}
        extra: list[dict[str, Any]] = []
        for hit in hits[:_EXPAND_TOP]:
            for related in [*(hit.get("parents") or []), *self._children.get(hit["id"], [])]:
                if related not in seen and related in self._by_id and len(extra) < limit:
                    seen.add(related)
                    extra.append(self._by_id[related])
        return extra


@lru_cache(maxsize=16)
def _index(schema_name: str, data_path: str | None) -> CandidateIndex:
    from moose.schema import get_schema_config

    return CandidateIndex(get_schema_config(schema_name).load_type_records())


def candidate_index(schema: SchemaConfig) -> CandidateIndex:
    return _index(schema.name, str(schema.data_path) if schema.data_path else None)
