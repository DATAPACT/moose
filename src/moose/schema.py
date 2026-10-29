from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

COARSE_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "WORK",
    "PRODUCT",
    "CONCEPT",
    "MISC",
]

FINE_TYPES = [
    "PERSON",
    "FICTIONAL_CHARACTER",
    "COMPANY",
    "NONPROFIT_ORG",
    "GOVERNMENT_ORG",
    "EDUCATIONAL_ORG",
    "SPORTS_TEAM",
    "COUNTRY",
    "CITY",
    "REGION",
    "LANDMARK",
    "CELESTIAL_BODY",
    "CONFLICT",
    "SPORT_EVENT",
    "EVENT_GENERIC",
    "FILM",
    "BOOK",
    "MUSIC_WORK",
    "SOFTWARE",
    "INTERNET_MEME",
    "DEVICE",
    "MEDICATION",
    "FOOD_BEVERAGE",
    "PRODUCT_GENERIC",
    "LANGUAGE",
    "LAW",
    "SCIENTIFIC_THEORY",
    "BIOLOGICAL_TAXON",
    "ANATOMY",
]

FINE_TO_COARSE = {
    "PERSON": "PERSON",
    "FICTIONAL_CHARACTER": "PERSON",
    "COMPANY": "ORGANIZATION",
    "NONPROFIT_ORG": "ORGANIZATION",
    "GOVERNMENT_ORG": "ORGANIZATION",
    "EDUCATIONAL_ORG": "ORGANIZATION",
    "SPORTS_TEAM": "ORGANIZATION",
    "COUNTRY": "LOCATION",
    "CITY": "LOCATION",
    "REGION": "LOCATION",
    "LANDMARK": "LOCATION",
    "CELESTIAL_BODY": "LOCATION",
    "CONFLICT": "EVENT",
    "SPORT_EVENT": "EVENT",
    "EVENT_GENERIC": "EVENT",
    "FILM": "WORK",
    "BOOK": "WORK",
    "MUSIC_WORK": "WORK",
    "SOFTWARE": "WORK",
    "INTERNET_MEME": "WORK",
    "DEVICE": "PRODUCT",
    "MEDICATION": "PRODUCT",
    "FOOD_BEVERAGE": "PRODUCT",
    "PRODUCT_GENERIC": "PRODUCT",
    "LANGUAGE": "CONCEPT",
    "LAW": "CONCEPT",
    "SCIENTIFIC_THEORY": "CONCEPT",
    "BIOLOGICAL_TAXON": "CONCEPT",
    "ANATOMY": "CONCEPT",
}

# One-line definitions, so the System 1 model and the LLM can tell the types apart.
TYPE_DEFINITIONS = {
    "PERSON": "A named person, real or (in the fine schema's FICTIONAL_CHARACTER) fictional.",
    "ORGANIZATION": "A named organisation: company, institution, agency, team or group.",
    "LOCATION": "A named place: country, city, region, landmark or celestial body.",
    "EVENT": "A named event: conflict, sports event, festival, incident or other occurrence.",
    "WORK": "A named creative or intellectual work: film, book, music, software or meme.",
    "PRODUCT": "A named product: device, medication, food or drink, or other manufactured good.",
    "CONCEPT": "A named abstract concept: language, law, scientific theory, taxon or body part.",
    "MISC": "A named entity that fits none of the other types.",
    "FICTIONAL_CHARACTER": "A character from fiction: a novel, film, game or myth.",
    "COMPANY": "A business or commercial company.",
    "NONPROFIT_ORG": "A non-profit organisation, charity or NGO.",
    "GOVERNMENT_ORG": "A government body, ministry, agency or public authority.",
    "EDUCATIONAL_ORG": "A school, university or other educational institution.",
    "SPORTS_TEAM": "A sports team or club.",
    "COUNTRY": "A country or sovereign state.",
    "CITY": "A city, town or village.",
    "REGION": "A region, state, province, county or other area within or across countries.",
    "LANDMARK": "A building, monument, natural landmark or other specific site.",
    "CELESTIAL_BODY": "A planet, star, moon or other astronomical object.",
    "CONFLICT": "A war, battle or armed conflict.",
    "SPORT_EVENT": "A sports competition, tournament, match or race.",
    "EVENT_GENERIC": "Another named event: festival, conference, disaster or ceremony.",
    "FILM": "A film, movie or television series.",
    "BOOK": "A book, novel or other written publication.",
    "MUSIC_WORK": "A song, album or other musical work.",
    "SOFTWARE": "A software product, application, programming language or online service.",
    "INTERNET_MEME": "An internet meme or viral online phenomenon.",
    "DEVICE": "A device, machine, vehicle or piece of hardware.",
    "MEDICATION": "A medicine, drug or pharmaceutical product.",
    "FOOD_BEVERAGE": "A food, dish or drink.",
    "PRODUCT_GENERIC": "Another named product or brand.",
    "LANGUAGE": "A human language or dialect.",
    "LAW": "A law, regulation, treaty or legal act.",
    "SCIENTIFIC_THEORY": "A scientific theory, principle, model or method.",
    "BIOLOGICAL_TAXON": "A species, genus or other biological group of organisms.",
    "ANATOMY": "A part of the body, organ or anatomical structure.",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
VOCAB_REGISTRY_PATH = DATA_DIR / "vocabularies.json"


def user_data_dir() -> Path:
    """Where user (bring-your-own) schemas live: MOOSE_USER_DATA_DIR if set; ~/.moose for an
    installed package, whose own folder may not be writable; else the package's data folder
    (a source checkout and the Docker image, which mounts it)."""
    configured = os.getenv("MOOSE_USER_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if {"site-packages", "dist-packages"} & set(DATA_DIR.parts):
        return Path.home() / ".moose"
    return DATA_DIR

DEFAULT_TEXT_INTRO = "You are a high-precision NER engine."
DEFAULT_TABLE_INTRO = "You are a semantic typing engine for tabular data."
DEFAULT_CPA_INTRO = "You are a column relationship prediction engine."

# "all": every type id is in one prompt. "two_step": the LLM first finds mentions and
# describes them, then each mention's type is chosen among a few retrieved candidates,
# by the System 1 model where available (see moose.linkers). "auto": two_step when the
# System 1 model is installed, "all" otherwise.
CANDIDATE_MODES = ("all", "two_step", "auto")
DEFAULT_CANDIDATE_K = 8
# Two-step mode, step 2: who chooses among the candidates (see moose.linkers). "auto" is
# System 1 with LLM help when Laya is installed, the LLM otherwise.
LINKERS = ("auto", "system1+llm", "system1", "llm")


@dataclass(frozen=True)
class SchemaConfig:
    name: str
    label: str
    description: str | None
    require_all_scores: bool
    text_intro: str
    table_intro: str
    cpa_intro: str
    type_ids: tuple[str, ...] | None = None
    data_path: Path | None = None
    coarse_mapping: dict[str, str] | None = None
    type_aliases: dict[str, str] | None = None
    type_alias_prefixes: dict[str, str] | None = None
    prefilter_types: bool = False
    supports_text: bool = True
    supports_table: bool = True
    supports_cpa: bool = False
    candidate_mode: str = "all"
    candidate_k: int = DEFAULT_CANDIDATE_K
    # Two-step mode: what kind of mention to look for, in plain words, and extra span rules
    # (replacing the default one: values and mentions of a kind of thing both count).
    mention_guidance: str | None = None
    mention_rules: tuple[str, ...] | None = None
    linker: str = "auto"
    # System 1 settings for this schema's candidates (Laya checkpoint, when to consult the
    # LLM, which tasks its probabilities were validated on).
    linker_calibration: Path | None = DATA_DIR / "linkers" / "laya_zero_shot.json"

    def load_type_records(self) -> list[dict[str, Any]]:
        """Types with their label, alternative labels, definition and parents, where the source has them."""
        if self.type_ids is not None:
            return [
                {
                    "id": type_id,
                    "label": type_id.replace("_", " ").lower(),
                    "alt_labels": [],
                    "definition": TYPE_DEFINITIONS.get(type_id),
                    "parents": [],
                }
                for type_id in self.type_ids
            ]
        if self.data_path is None:
            raise ValueError(f"Schema {self.name} has no type source configured.")
        return [dict(record) for record in _load_type_records(self.data_path)]

    def load_type_ids(self) -> list[str]:
        if self.type_ids is not None:
            return list(self.type_ids)
        if self.data_path is None:
            raise ValueError(f"Schema {self.name} has no type source configured.")
        return list(_load_vocab_file(self.data_path))


def _extract_type_ids(data: Any) -> list[str]:
    if isinstance(data, dict):
        for key in ("types", "type_ids", "items", "data"):
            if key in data:
                data = data[key]
                break
    if isinstance(data, list):
        type_ids: list[str] = []
        for item in data:
            if isinstance(item, str):
                type_ids.append(item)
            elif isinstance(item, dict):
                type_id = item.get("id")
                if isinstance(type_id, str):
                    type_ids.append(type_id)
        return type_ids
    return []


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


RDFS_CLASS = "http://www.w3.org/2000/01/rdf-schema#Class"


@lru_cache
def _concept_files(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Concept records and parents from the nearest concepts.json / concept_parents.json
    at or above `directory` (within the data folder), nearest first."""
    concepts: dict[str, dict[str, Any]] = {}
    parents: dict[str, list[str]] = {}
    current = directory
    while True:
        if (current / "concepts.json").exists():
            for item in json.loads((current / "concepts.json").read_text(encoding="utf-8")):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    concepts.setdefault(item["id"], item)
        if (current / "concept_parents.json").exists():
            for key, value in json.loads((current / "concept_parents.json").read_text(encoding="utf-8")).items():
                parents.setdefault(key, value)
        if current == DATA_DIR or current == current.parent or DATA_DIR not in current.parents:
            break
        current = current.parent
    return concepts, parents


# Definitions that replace the source's where it misleads. DPV defines dpv-pd:Identifier as
# "an identifier or name used for identification" and dpv-pd:Name only as "information about
# names", so Laya typed 46 of 53 names in DPVCallBench's train split something else (39
# Identifier); with this one it types all 53 Name, and identifiers as before. On the test
# split, dpv_privacy's system1 accuracy went 0.81 -> 0.94 and system1+llm 0.88 -> 0.95, with
# ECE 0.15 -> 0.10 and 0.11 -> 0.10.
DEFINITION_CLARIFICATIONS = {
    "dpv-pd:Name": "The name of a person (given name, surname, full name or nickname), not a number or code",
}


@lru_cache
def _load_type_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Types with label, alternative labels, definition and parents. Bare type ids are
    looked up in the nearest DPV-style concept files; entries known not to be classes
    (properties, documentation pages) are left out, since they cannot type a mention."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("types", "type_ids", "items", "data"):
            if key in data:
                data = data[key]
                break
    concepts, parents = _concept_files(path.parent)
    records = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):
            item = concepts.get(item) or {
                "id": item, "label": item.replace("_", " ").lower(), "definition": TYPE_DEFINITIONS.get(item)
            }
        if not (isinstance(item, dict) and isinstance(item.get("id"), str)):
            continue
        if isinstance(item.get("types"), list) and item["types"] and RDFS_CLASS not in item["types"]:
            continue
        alt = item.get("alt_labels") or item.get("aliases") or []
        records.append({
            "id": item["id"],
            "label": _first_text(item, ("label", "name", "title")) or item["id"],
            "alt_labels": [a for a in alt if isinstance(a, str)] if isinstance(alt, list) else [],
            "definition": DEFINITION_CLARIFICATIONS.get(item["id"]) or _first_text(item, ("definition", "description", "comment")),
            "parents": [p for p in item.get("parents") or parents.get(item["id"], []) if isinstance(p, str)],
        })
    return tuple(records)


@lru_cache
def _load_vocab_file(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    type_ids = _extract_type_ids(data)
    if not type_ids:
        raise ValueError(f"No type ids found in vocabulary file: {path}")
    return tuple(type_ids)


def _normalize_intro(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return fallback


def _parse_score_mode(entry: dict[str, Any]) -> bool:
    require_all = entry.get("require_all_scores")
    if isinstance(require_all, bool):
        return require_all
    score_mode = entry.get("score_mode", "dense")
    if score_mode not in {"dense", "sparse"}:
        raise ValueError(f"Invalid score_mode: {score_mode}")
    return score_mode == "dense"


def _parse_alias_mapping(value: Any, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    mapping: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings.")
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{label} values must be non-empty strings.")
        mapping[key.strip()] = val.strip()
    return mapping or None


def _parse_coarse_mapping(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("coarse_mapping must be a JSON object.")
    mapping: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("coarse_mapping keys must be non-empty strings.")
        if not isinstance(val, str) or not val.strip():
            raise ValueError("coarse_mapping values must be non-empty strings.")
        mapping[key.strip()] = val.strip()
    return mapping or None


def _parse_bool_field(entry: dict[str, Any], key: str, default: bool) -> bool:
    value = entry.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _parse_linker(entry: dict[str, Any]) -> tuple[str, Path | None]:
    linker = entry.get("linker", "auto")
    if linker not in LINKERS:
        raise ValueError(f"linker must be one of {', '.join(LINKERS)}.")
    calibration = entry.get("linker_calibration")
    if calibration is None:
        return linker, DATA_DIR / "linkers" / "laya_zero_shot.json"
    if not isinstance(calibration, str) or not calibration.strip():
        raise ValueError("linker_calibration must be a path.")
    path = Path(calibration.strip())
    return linker, path if path.is_absolute() else DATA_DIR / path


def _parse_candidate_mode(entry: dict[str, Any]) -> tuple[str, int]:
    mode = entry.get("candidate_mode", "all")
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"candidate_mode must be one of {', '.join(CANDIDATE_MODES)}.")
    k = entry.get("candidate_k", DEFAULT_CANDIDATE_K)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 50:
        raise ValueError("candidate_k must be an integer from 1 to 50.")
    return mode, k


def _load_vocab_registry_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Vocabulary registry must be a JSON array.")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Vocabulary registry entries must be JSON objects.")
    return data


@lru_cache
def _schema_registry() -> dict[str, SchemaConfig]:
    registry: dict[str, SchemaConfig] = {
        "coarse": SchemaConfig(
            name="coarse",
            label="Coarse",
            description="Coarse-grained NER schema.",
            require_all_scores=True,
            text_intro=DEFAULT_TEXT_INTRO,
            table_intro=DEFAULT_TABLE_INTRO,
            cpa_intro=DEFAULT_CPA_INTRO,
            type_ids=tuple(COARSE_TYPES),
            supports_cpa=False,
        ),
        "fine": SchemaConfig(
            name="fine",
            label="Fine",
            description="Fine-grained NER schema with coarse mapping.",
            require_all_scores=False,
            text_intro=DEFAULT_TEXT_INTRO,
            table_intro=DEFAULT_TABLE_INTRO,
            cpa_intro=DEFAULT_CPA_INTRO,
            type_ids=tuple(FINE_TYPES),
            coarse_mapping=FINE_TO_COARSE,
            type_aliases={"HUMAN": "PERSON"},
            supports_cpa=False,
        ),
    }

    seen_registry_names: set[str] = set()
    # Each registry's relative paths are relative to its own folder.
    registry_paths = [(VOCAB_REGISTRY_PATH, DATA_DIR), (user_data_dir() / "user_vocabularies.json", user_data_dir())]
    for registry_path, base_dir in registry_paths:
        for entry in _load_vocab_registry_entries(registry_path):
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Vocabulary entry missing a valid name.")
            name = name.strip()
            if name in seen_registry_names:
                raise ValueError(f"Duplicate schema name: {name}")
            seen_registry_names.add(name)
            source = entry.get("type_source") or entry.get("data_path") or entry.get("path")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Vocabulary {name} missing type_source path.")
            data_path = Path(source.strip())
            if not data_path.is_absolute():
                data_path = base_dir / data_path
            label = entry.get("label")
            description = entry.get("description")
            candidate_mode, candidate_k = _parse_candidate_mode(entry)
            linker, linker_calibration = _parse_linker(entry)
            mention_guidance = entry.get("mention_guidance")
            mention_rules = entry.get("mention_rules")
            registry[name] = SchemaConfig(
                name=name,
                label=label.strip() if isinstance(label, str) and label.strip() else name,
                description=description.strip() if isinstance(description, str) else None,
                require_all_scores=_parse_score_mode(entry),
                text_intro=_normalize_intro(entry.get("text_intro"), DEFAULT_TEXT_INTRO),
                table_intro=_normalize_intro(entry.get("table_intro"), DEFAULT_TABLE_INTRO),
                cpa_intro=_normalize_intro(entry.get("cpa_intro"), DEFAULT_CPA_INTRO),
                data_path=data_path,
                type_aliases=_parse_alias_mapping(entry.get("type_aliases"), "type_aliases"),
                type_alias_prefixes=_parse_alias_mapping(
                    entry.get("type_alias_prefixes"), "type_alias_prefixes"
                ),
                coarse_mapping=_parse_coarse_mapping(entry.get("coarse_mapping")),
                prefilter_types=_parse_bool_field(entry, "prefilter_types", False),
                supports_text=_parse_bool_field(entry, "supports_text", True),
                supports_table=_parse_bool_field(entry, "supports_table", True),
                supports_cpa=_parse_bool_field(entry, "supports_cpa", False),
                candidate_mode=candidate_mode,
                candidate_k=candidate_k,
                linker=linker,
                linker_calibration=linker_calibration,
                mention_guidance=mention_guidance.strip() if isinstance(mention_guidance, str) and mention_guidance.strip() else None,
                mention_rules=tuple(r for r in mention_rules if isinstance(r, str) and r.strip()) if isinstance(mention_rules, list) else None,
            )
    return registry


def list_schema_names() -> list[str]:
    return sorted(_schema_registry())


def reload_schema_registry() -> list[str]:
    _load_vocab_file.cache_clear()
    _load_type_records.cache_clear()
    _concept_files.cache_clear()
    _schema_registry.cache_clear()
    return list_schema_names()


def get_schema_config(schema: str) -> SchemaConfig:
    registry = _schema_registry()
    if schema not in registry:
        # In multi-worker deployments, one worker may ingest a schema while another
        # still has a stale in-memory cache. Refresh once before failing.
        reload_schema_registry()
        registry = _schema_registry()
    if schema not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown schema '{schema}'. Available schemas: {known}")
    return registry[schema]


def get_types(schema: str) -> list[str]:
    config = get_schema_config(schema)
    return config.load_type_ids()


def get_coarse_type(type_id: str) -> str | None:
    if type_id in COARSE_TYPES:
        return type_id
    return FINE_TO_COARSE.get(type_id)
