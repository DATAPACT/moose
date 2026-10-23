"""Build the DPV privacy subset used by the `dpv_privacy` schema.

Personal-data categories (dpv-pd), legal roles (controller, processor, recipient, third
party, ...) and data-subject types, each with its label, alternative labels and
definition, so two-step annotation can retrieve candidates and show them to the LLM,
and its parents within the subset, so candidates can be widened along the hierarchy.

    python scripts/build_dpv_privacy_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

DPV_DIR = Path(__file__).resolve().parents[1] / "src" / "moose" / "data" / "dpv"
OUT = DPV_DIR / "subsets" / "privacy" / "types.json"
ROLE_MODULES = {"entities_legalrole", "entities_datasubject"}


RDFS_CLASS = "http://www.w3.org/2000/01/rdf-schema#Class"


def keep(concept: dict) -> bool:
    cid = concept.get("id") or ""
    # Only categories: not properties (hasDataProcessor) or documentation pages (primer, guides).
    if not concept.get("label") or RDFS_CLASS not in (concept.get("types") or []):
        return False
    return cid.startswith("dpv-pd:") or (cid.startswith("dpv:") and bool(ROLE_MODULES & set(concept.get("modules") or [])))


def main() -> None:
    concepts = json.loads((DPV_DIR / "concepts.json").read_text(encoding="utf-8"))
    parents = json.loads((DPV_DIR / "concept_parents.json").read_text(encoding="utf-8"))
    kept = {c["id"] for c in concepts if keep(c)}
    types = [
        {
            "id": c["id"],
            "label": c["label"],
            "alt_labels": c.get("alt_labels") or [],
            "definition": c.get("definition"),
            "parents": [p for p in parents.get(c["id"], []) if p in kept],
        }
        for c in concepts
        if keep(c)
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"types": types}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(types)} types to {OUT}")


if __name__ == "__main__":
    main()
