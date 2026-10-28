from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


SCHEMAORG_LATEST_URL = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
SCHEMAORG_HTTP_PREFIXES = ("https://schema.org/", "http://schema.org/")


# -----------------------
# Optional DeepSeek usage
# -----------------------
def maybe_suggest_seeds_with_deepseek(use_case: str) -> list[str] | None:
    """
    OPTIONAL: Suggest seed classes from a natural-language description.
    This is not required for ontology ingestion. It’s only a convenience.
    Requires:
      - pip install openai
      - env DEEPSEEK_API_KEY
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system_prompt = (
        "You suggest seed schema.org classes for building a curated subset.\n"
        "Return ONLY JSON with a key 'seed_classes' which is a list of CURIEs like 'schema:Book'.\n"
        "Pick 5-12 broad but relevant classes.\n"
        "Do not include schema:Thing.\n"
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": use_case},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    seeds = data.get("seed_classes")
    if not isinstance(seeds, list) or not all(isinstance(x, str) for x in seeds):
        return None
    return seeds


# -----------------------
# JSON-LD parsing helpers
# -----------------------
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_id(obj: Any) -> str | None:
    """
    Extract @id from a JSON-LD node/term reference.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("@id") or obj.get("id")
    return None


def _pick_lang_string(value: Any, lang: str = "en") -> str | None:
    """
    schema.org JSON-LD labels may be:
      - string
      - dict: {"@value": "...", "@language": "en"}
      - list of such dicts/strings
    Return the best available string (prefer english).
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        # JSON-LD literal
        if value.get("@language") == lang and isinstance(value.get("@value"), str):
            s = value["@value"].strip()
            return s or None
        # fallback: @value without lang
        if isinstance(value.get("@value"), str):
            s = value["@value"].strip()
            return s or None
        return None

    if isinstance(value, list):
        # Prefer exact language match
        for item in value:
            if isinstance(item, dict) and item.get("@language") == lang and isinstance(item.get("@value"), str):
                s = item["@value"].strip()
                if s:
                    return s
        # Fallback: first usable string
        for item in value:
            s = _pick_lang_string(item, lang=lang)
            if s:
                return s
        return None

    return None


def _normalize_schema_id(value: str) -> tuple[str, str] | None:
    """
    Convert schema.org identifiers into:
      (curie, iri)

    Accepts:
      - schema:Book
      - https://schema.org/Book
      - http://schema.org/Book

    Returns None if not a schema.org term.
    """
    v = (value or "").strip()
    if not v:
        return None

    if v.startswith("schema:"):
        local = v[len("schema:") :]
        iri = f"https://schema.org/{local}"
        return v, iri

    for prefix in SCHEMAORG_HTTP_PREFIXES:
        if v.startswith(prefix):
            local = v[len(prefix) :]
            curie = f"schema:{local}"
            iri = f"https://schema.org/{local}"
            return curie, iri

    return None


def _node_types(node: dict[str, Any]) -> set[str]:
    t = node.get("@type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    return set()


def _is_class(node: dict[str, Any]) -> bool:
    t = _node_types(node)
    # schema.org uses rdfs:Class for classes, and also schema:Class in some contexts.
    return any(x in t for x in ("rdfs:Class", "owl:Class", "schema:Class"))


def _is_property(node: dict[str, Any]) -> bool:
    t = _node_types(node)
    return any(x in t for x in ("rdf:Property", "owl:ObjectProperty", "owl:DatatypeProperty"))


def _extract_schema_label(node: dict[str, Any]) -> str:
    # schema.org commonly uses rdfs:label, but also schema:name appears.
    for key in ("rdfs:label", "schema:name", "name"):
        s = _pick_lang_string(node.get(key), lang="en")
        if s:
            return s
    # fallback to local name from id
    raw_id = node.get("@id") or node.get("id") or ""
    norm = _normalize_schema_id(str(raw_id))
    if norm:
        curie, _ = norm
        return curie.split(":", 1)[1]
    return str(raw_id)


def _extract_id_list(node: dict[str, Any], key: str) -> list[str]:
    """
    Extract a list of schema:* CURIE ids from a JSON-LD property that may be:
      - dict with @id
      - list of dicts
      - string
    """
    out: list[str] = []
    for item in _as_list(node.get(key)):
        raw = _get_id(item)
        if not raw:
            continue
        norm = _normalize_schema_id(raw)
        if not norm:
            continue
        curie, _iri = norm
        out.append(curie)
    # unique preserve order
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


@dataclass
class SchemaOrgArtifacts:
    classes: dict[str, dict[str, str]]                # id -> {id, iri, label}
    properties: dict[str, dict[str, str]]             # id -> {id, iri, label}
    class_parents: dict[str, list[str]]               # class_id -> [parent_ids]
    property_index: dict[str, dict[str, Any]]         # prop_id -> {id, iri, label, domains, ranges}
    class_ancestors: dict[str, list[str]]             # class_id -> [ancestor_ids...]


def build_schemaorg_artifacts(doc: dict[str, Any]) -> SchemaOrgArtifacts:
    graph = doc.get("@graph")
    if not isinstance(graph, list):
        raise ValueError("JSON-LD does not contain @graph list.")

    classes: dict[str, dict[str, str]] = {}
    properties: dict[str, dict[str, str]] = {}
    class_parents: dict[str, list[str]] = {}
    property_index: dict[str, dict[str, Any]] = {}

    # First pass: identify classes/properties and basic metadata
    for node in graph:
        if not isinstance(node, dict):
            continue
        raw_id = node.get("@id") or node.get("id")
        if not isinstance(raw_id, str):
            continue
        norm = _normalize_schema_id(raw_id)
        if not norm:
            continue
        curie, iri = norm

        if _is_class(node):
            classes[curie] = {"id": curie, "iri": iri, "label": _extract_schema_label(node)}
            parents = _extract_id_list(node, "rdfs:subClassOf")
            class_parents[curie] = parents

        if _is_property(node):
            properties[curie] = {"id": curie, "iri": iri, "label": _extract_schema_label(node)}
            domains = _extract_id_list(node, "schema:domainIncludes")
            ranges = _extract_id_list(node, "schema:rangeIncludes")
            property_index[curie] = {
                "id": curie,
                "iri": iri,
                "label": _extract_schema_label(node),
                "domains": domains,
                "ranges": ranges,
            }

    # Ensure every class has a parents entry
    for cid in list(classes.keys()):
        class_parents.setdefault(cid, [])

    # Compute ancestors (transitive closure)
    class_ancestors = compute_ancestors(class_parents)

    return SchemaOrgArtifacts(
        classes=classes,
        properties=properties,
        class_parents=class_parents,
        property_index=property_index,
        class_ancestors=class_ancestors,
    )


def compute_ancestors(class_parents: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Compute transitive ancestors for each class id.
    """
    memo: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def dfs(c: str) -> list[str]:
        if c in memo:
            return memo[c]
        if c in visiting:
            # cycle guard
            return []
        visiting.add(c)
        ancestors: list[str] = []
        for p in class_parents.get(c, []):
            if p not in ancestors:
                ancestors.append(p)
            for ap in dfs(p):
                if ap not in ancestors:
                    ancestors.append(ap)
        visiting.remove(c)
        memo[c] = ancestors
        return ancestors

    for c in class_parents.keys():
        dfs(c)
    return memo


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def to_sorted_list(mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [mapping[k] for k in sorted(mapping.keys())]


def download_jsonld(url: str, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    write_json(out_path, data)
    return data


# -----------------------
# Subset builder
# -----------------------
def normalize_seed(seed: str) -> str:
    """
    Accept:
      - Book -> schema:Book
      - schema:Book -> schema:Book
      - https://schema.org/Book -> schema:Book
    """
    s = (seed or "").strip()
    if not s:
        raise ValueError("Empty seed")
    if s.startswith("schema:"):
        return s
    norm = _normalize_schema_id(s)
    if norm:
        return norm[0]
    # assume local name
    return f"schema:{s}"


def build_children_map(class_parents: dict[str, list[str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for child, parents in class_parents.items():
        for p in parents:
            children[p].append(child)
    # stable ordering
    for p in list(children.keys()):
        children[p] = sorted(set(children[p]))
    return children


def closure_descendants(seeds: set[str], children_map: dict[str, list[str]]) -> set[str]:
    """
    Return seeds + all descendants via rdfs:subClassOf.
    """
    out = set(seeds)
    q = deque(seeds)
    while q:
        cur = q.popleft()
        for ch in children_map.get(cur, []):
            if ch not in out:
                out.add(ch)
                q.append(ch)
    return out


def closure_ancestors_with_stop(
    classes: set[str],
    class_ancestors: dict[str, list[str]],
    stop_at: set[str],
) -> set[str]:
    """
    Add ancestors for each class, but stop at stop_at (do not include them, do not traverse beyond).
    """
    out = set(classes)
    for c in list(classes):
        for a in class_ancestors.get(c, []):
            if a in stop_at:
                # do not include stop nodes, and do not include their ancestors (they appear later in the list)
                continue
            out.add(a)
    return out


def build_curated_subset(
    artifacts: SchemaOrgArtifacts,
    seed_classes: list[str],
    stop_ancestors_at: list[str],
    include_ancestors: bool,
    property_domains: list[str] | None = None,
) -> SchemaOrgArtifacts:
    """
    Curate subset:
      - take seed classes
      - include their descendants
      - optionally include ancestors (but stop at schema:Thing by default)
      - keep properties whose domain intersects subset classes or property_domains
        (classes whose properties are kept without adding their descendants)
      - include range types that appear in selected properties (as classes if present)
    """
    seeds = {normalize_seed(s) for s in seed_classes}
    stop = {normalize_seed(s) for s in stop_ancestors_at} if stop_ancestors_at else set()

    children_map = build_children_map(artifacts.class_parents)
    class_subset = closure_descendants(seeds, children_map)

    if include_ancestors:
        class_subset = closure_ancestors_with_stop(class_subset, artifacts.class_ancestors, stop_at=stop)

    # Select properties by domain intersection with class subset
    property_domain_set = class_subset | {normalize_seed(c) for c in property_domains or []}
    prop_subset: dict[str, dict[str, str]] = {}
    prop_index_subset: dict[str, dict[str, Any]] = {}
    ranges_to_include: set[str] = set()

    for pid, meta in artifacts.property_index.items():
        domains = meta.get("domains") or []
        if any(d in property_domain_set for d in domains):
            prop_subset[pid] = artifacts.properties[pid]
            prop_index_subset[pid] = meta
            for r in meta.get("ranges") or []:
                ranges_to_include.add(r)

    # Ensure class subset includes range types if schema.org defines them as classes
    for r in ranges_to_include:
        if r in artifacts.classes:
            class_subset.add(r)

    classes_subset_dict = {cid: artifacts.classes[cid] for cid in class_subset if cid in artifacts.classes}
    class_parents_subset = {
        cid: [p for p in artifacts.class_parents.get(cid, []) if p in classes_subset_dict]
        for cid in classes_subset_dict.keys()
    }
    class_ancestors_subset = compute_ancestors(class_parents_subset)

    return SchemaOrgArtifacts(
        classes=classes_subset_dict,
        properties=prop_subset,
        class_parents=class_parents_subset,
        property_index=prop_index_subset,
        class_ancestors=class_ancestors_subset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build schema.org ontology artifacts for Moose.")
    parser.add_argument("--url", default=SCHEMAORG_LATEST_URL, help="schema.org JSON-LD URL")
    parser.add_argument(
        "--out-dir",
        default="src/moose/data/schemaorg",
        help="Output directory (default: src/moose/data/schemaorg)",
    )
    parser.add_argument("--no-download", action="store_true", help="Use cached raw jsonld if present")
    parser.add_argument(
        "--raw-file-name",
        default="raw_schemaorg-current-https.jsonld",
        help="File name to store raw JSON-LD",
    )

    # subset options
    parser.add_argument("--write-subset", action="store_true", help="Also produce a curated subset")
    parser.add_argument(
        "--subset-name",
        default="curated_v1",
        help="Subset output folder name under subsets/",
    )
    parser.add_argument(
        "--seed-classes",
        default="schema:Book,schema:Person,schema:Organization,schema:Place,schema:Event,schema:Product,schema:CreativeWork",
        help="Comma-separated seed classes (CURIEs or local names).",
    )
    parser.add_argument(
        "--include-ancestors",
        action="store_true",
        help="Include ancestors of seed/descendant classes (recommended).",
    )
    parser.add_argument(
        "--stop-ancestors-at",
        default="schema:Thing",
        help="Comma-separated classes where ancestor inclusion stops (do not include these). Default: schema:Thing",
    )

    parser.add_argument(
        "--property-domains",
        default="schema:Thing,schema:PostalAddress,schema:GeoCoordinates,schema:Offer,schema:Rating,"
        "schema:AggregateRating,schema:OpeningHoursSpecification",
        help="Comma-separated classes whose properties are also kept, e.g. schema:Thing (name, url, "
        "description) or schema:PostalAddress (streetAddress, postalCode), which tables often hold flattened.",
    )

    # optional LLM seed suggestion
    parser.add_argument(
        "--deepseek-suggest-seeds",
        default=None,
        help="Optional: use DeepSeek to suggest seed classes from this use-case description (requires DEEPSEEK_API_KEY).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_path = out_dir / args.raw_file_name

    if args.no_download:
        if not raw_path.exists():
            raise SystemExit(f"--no-download set but raw file not found: {raw_path}")
        doc = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        doc = download_jsonld(args.url, raw_path)

    artifacts = build_schemaorg_artifacts(doc)

    # Write full artifacts
    write_json(out_dir / "classes.json", to_sorted_list(artifacts.classes))
    write_json(out_dir / "properties.json", to_sorted_list(artifacts.properties))
    write_json(out_dir / "property_index.json", dict(sorted(artifacts.property_index.items())))
    write_json(out_dir / "class_parents.json", dict(sorted(artifacts.class_parents.items())))
    write_json(out_dir / "class_ancestors.json", dict(sorted(artifacts.class_ancestors.items())))

    print("Wrote full schema.org artifacts:")
    print(f"  classes:     {len(artifacts.classes)}")
    print(f"  properties:  {len(artifacts.properties)}")

    if args.write_subset:
        if args.deepseek_suggest_seeds:
            llm_seeds = maybe_suggest_seeds_with_deepseek(args.deepseek_suggest_seeds)
            if llm_seeds:
                seed_classes = llm_seeds
                print("DeepSeek suggested seed classes:", seed_classes)
            else:
                seed_classes = [s.strip() for s in args.seed_classes.split(",") if s.strip()]
                print("DeepSeek suggestion unavailable; using --seed-classes:", seed_classes)
        else:
            seed_classes = [s.strip() for s in args.seed_classes.split(",") if s.strip()]

        stop_ancestors_at = [s.strip() for s in args.stop_ancestors_at.split(",") if s.strip()]

        subset = build_curated_subset(
            artifacts=artifacts,
            seed_classes=seed_classes,
            stop_ancestors_at=stop_ancestors_at,
            include_ancestors=bool(args.include_ancestors),
            property_domains=[s.strip() for s in args.property_domains.split(",") if s.strip()],
        )

        subset_dir = out_dir / "subsets" / args.subset_name
        write_json(subset_dir / "classes.json", to_sorted_list(subset.classes))
        write_json(subset_dir / "properties.json", to_sorted_list(subset.properties))
        write_json(subset_dir / "property_index.json", dict(sorted(subset.property_index.items())))
        write_json(subset_dir / "class_parents.json", dict(sorted(subset.class_parents.items())))
        write_json(subset_dir / "class_ancestors.json", dict(sorted(subset.class_ancestors.items())))
        # The label sets of the schemaorg_cta_v1 / schemaorg_cpa_v1 schemas, with Moose's sentinels.
        write_json(subset_dir / "cta_labels.json",
                   [{"id": "moose:OTHER", "iri": None, "label": "Other / Unknown Class"}, *to_sorted_list(subset.classes)])
        write_json(subset_dir / "cpa_labels.json",
                   [{"id": "moose:NONE", "iri": None, "label": "No Relationship"},
                    {"id": "moose:OTHER", "iri": None, "label": "Other Relationship"}, *to_sorted_list(subset.properties)])

        print("Wrote curated subset:")
        print(f"  subset_dir:  {subset_dir}")
        print(f"  classes:     {len(subset.classes)}")
        print(f"  properties:  {len(subset.properties)}")


if __name__ == "__main__":
    main()