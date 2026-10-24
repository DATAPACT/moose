from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from moose.schema import reload_schema_registry, user_data_dir
from moose.validate import extract_json


MAX_SOURCE_BYTES = 5 * 1024 * 1024
SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ID_FIELD_CANDIDATES = ("id", "@id", "type_id", "identifier", "code", "uri", "iri")
LABEL_FIELD_CANDIDATES = ("label", "name", "title", "prefLabel", "rdfs:label")
DESCRIPTION_FIELD_CANDIDATES = ("description", "definition", "comment", "rdfs:comment")
ALIASES_FIELD_CANDIDATES = ("aliases", "alt_labels", "altLabels", "synonyms")

TYPE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{1,127}$")
MIN_TYPES_PER_SCHEMA = 3
MAX_TYPES_PER_SCHEMA = 10000
MAX_LABEL_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 2000
MAX_ALIASES_PER_TYPE = 20
MAX_GITHUB_REF_LENGTH = 100
ALLOWED_SOURCE_EXTENSIONS = {".json", ".jsonld"}
DEFAULT_RESERVED_SCHEMA_NAMES = {
    "coarse",
    "fine",
    "dpv",
    "dpv_pd",
    "sti",
    "cpa",
    "schemaorg_cpa_v1",
    "schemaorg_cta_v1",
}
_BLOCKED_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _mapping_sha256(mapping: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(mapping, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _default_allowed_file_roots() -> list[Path]:
    roots = [Path.cwd(), Path("/tmp"), Path("/app"), Path(tempfile.gettempdir())]
    repo_root = Path(__file__).resolve().parents[2]
    roots.append(repo_root)
    roots.append(repo_root / "examples")
    roots.append(repo_root / "examples" / "schema_ingest_samples")
    roots.append(repo_root / "examples" / "schema_ingest_samples" / "schemas")
    return roots


def _allowed_file_roots() -> list[Path]:
    env_roots = _env_csv("MOOSE_SCHEMA_INGEST_ALLOWED_FILE_ROOTS")
    if env_roots:
        return [Path(p).expanduser().resolve() for p in env_roots]
    return [p.resolve() for p in _default_allowed_file_roots()]


def _assert_file_source_allowed(path: Path) -> None:
    allowed_roots = _allowed_file_roots()
    if any(_is_subpath(path, root) for root in allowed_roots):
        return
    roots = ", ".join(str(root) for root in allowed_roots)
    raise ValueError(f"File source path is outside allowed roots: {path}. Allowed roots: {roots}")


def _validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("URL source must use https.")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL source must include a hostname.")
    if host in _BLOCKED_HOSTNAMES:
        raise ValueError(f"URL source host is not allowed: {host}")
    ip = None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast):
        raise ValueError(f"URL source IP is not allowed: {host}")
    allowed_domains = _env_csv("MOOSE_SCHEMA_INGEST_ALLOWED_URL_DOMAINS")
    if allowed_domains and not any(host == d.lower() or host.endswith(f".{d.lower()}") for d in allowed_domains):
        raise ValueError(f"URL source domain is not allowlisted: {host}")


def _validate_github_source(repo: str, ref: str, rel_path: str) -> None:
    repo_str = repo.strip().removeprefix("https://github.com/").strip("/")
    if repo_str.count("/") != 1:
        raise ValueError("github_repo must be in owner/repo format.")
    if not ref.strip() or len(ref.strip()) > MAX_GITHUB_REF_LENGTH:
        raise ValueError("github_ref must be non-empty and reasonably short.")
    path_str = rel_path.strip()
    if not path_str:
        raise ValueError("github_path is required.")
    ext = Path(path_str).suffix.lower()
    if ext not in ALLOWED_SOURCE_EXTENSIONS:
        raise ValueError(f"github_path extension must be one of: {sorted(ALLOWED_SOURCE_EXTENSIONS)}")
    allowed_repos = _env_csv("MOOSE_SCHEMA_INGEST_ALLOWED_GITHUB_REPOS")
    if allowed_repos and repo_str not in allowed_repos:
        raise ValueError(f"github_repo is not allowlisted: {repo_str}")


def _validate_schema_name(schema_name: str) -> None:
    if not SCHEMA_NAME_RE.match(schema_name):
        raise ValueError("Schema name must match: ^[A-Za-z0-9][A-Za-z0-9_-]*$")
    if schema_name != schema_name.lower():
        raise ValueError("Schema name must be lowercase.")
    reserved = set(DEFAULT_RESERVED_SCHEMA_NAMES)
    reserved.update(_env_csv("MOOSE_SCHEMA_INGEST_RESERVED_NAMES"))
    if schema_name in reserved:
        raise ValueError(f"Schema name is reserved and cannot be used: {schema_name}")


def _resolve_source_file_path(source_path: str) -> Path:
    raw = source_path.strip()
    repo_root = Path(__file__).resolve().parents[2]
    raw_path = Path(raw).expanduser()

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path))
        candidates.append((repo_root / raw_path))

    # If the request carries a host-specific absolute path (e.g. /Users/.../examples/...),
    # remap to the runtime repo root in container/local mode.
    marker = "/examples/"
    raw_norm = raw.replace("\\", "/")
    if marker in raw_norm:
        suffix = raw_norm.split(marker, 1)[1]
        candidates.append(repo_root / "examples" / suffix)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return resolved

    tried = ", ".join(sorted(seen))
    raise FileNotFoundError(f"Source file not found: {raw}. Tried: {tried}")


def _normalize_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not _is_non_empty_string(item):
            continue
        alias = item.strip()
        if alias in seen:
            continue
        out.append(alias)
        seen.add(alias)
    return out


def _validate_type_record(
    *,
    type_id: str,
    label: str,
    description: str | None,
    aliases: list[str],
    index: int,
) -> None:
    if not TYPE_ID_RE.match(type_id):
        raise ValueError(f"Invalid type id format at index {index}: {type_id}")
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(f"Label too long at index {index}; max {MAX_LABEL_LENGTH} chars.")
    if description and len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"Description too long at index {index}; max {MAX_DESCRIPTION_LENGTH} chars.")
    if len(aliases) > MAX_ALIASES_PER_TYPE:
        raise ValueError(f"Too many aliases at index {index}; max {MAX_ALIASES_PER_TYPE}.")


def _coverage_ratio(records: list[dict[str, Any]], field: str) -> float:
    if not records:
        return 0.0
    valid = 0
    for record in records:
        if _is_non_empty_string(record.get(field)):
            valid += 1
    return valid / len(records)


def _best_field(records: list[dict[str, Any]], candidates: tuple[str, ...]) -> tuple[str | None, float]:
    best_name: str | None = None
    best_score = 0.0
    for field in candidates:
        score = _coverage_ratio(records, field)
        if score > best_score:
            best_name = field
            best_score = score
    return best_name, best_score


def _iter_list_paths(data: Any, max_depth: int = 4) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []

    def walk(node: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(node, list) and node:
            if all(isinstance(x, dict) for x in node) or all(isinstance(x, str) for x in node):
                out.append(path)
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    walk(value, path + (key,), depth + 1)

    walk(data, tuple(), 0)
    return out


def _path_to_string(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _string_to_path(path: str) -> tuple[str, ...]:
    if not path:
        return tuple()
    return tuple(part for part in path.split(".") if part)


def _get_at_path(data: Any, path: tuple[str, ...]) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            raise ValueError(f"Path '{_path_to_string(path)}' is not a valid object path.")
        if key not in cur:
            raise ValueError(f"Path '{_path_to_string(path)}' was not found in source.")
        cur = cur[key]
    return cur


def _normalize_records(
    records: list[Any],
    *,
    id_field: str,
    label_field: str | None,
    description_field: str | None,
    aliases_field: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    type_ids: list[str] = []
    seen: set[str] = set()

    for index, item in enumerate(records):
        if isinstance(item, str):
            type_id = item.strip()
            label = type_id
            description = None
            aliases: list[str] = []
        elif isinstance(item, dict):
            raw_id = item.get(id_field)
            if not _is_non_empty_string(raw_id):
                raise ValueError(f"Item at index {index} is missing a valid id field '{id_field}'.")
            type_id = raw_id.strip()
            raw_label = item.get(label_field) if label_field else None
            label = raw_label.strip() if _is_non_empty_string(raw_label) else type_id
            raw_desc = item.get(description_field) if description_field else None
            description = raw_desc.strip() if _is_non_empty_string(raw_desc) else None
            aliases = _normalize_aliases(item.get(aliases_field)) if aliases_field else []
        else:
            raise ValueError(f"Unsupported record type at index {index}: {type(item).__name__}")

        if not type_id:
            raise ValueError(f"Empty type id at index {index}.")
        if type_id in seen:
            raise ValueError(f"Duplicate type id: {type_id}")
        _validate_type_record(
            type_id=type_id,
            label=label,
            description=description,
            aliases=aliases,
            index=index,
        )
        seen.add(type_id)

        row: dict[str, Any] = {"id": type_id, "label": label}
        if description:
            row["description"] = description
        if aliases:
            row["aliases"] = aliases
        normalized.append(row)
        type_ids.append(type_id)

    return normalized, type_ids


def _deterministic_mapping(data: Any) -> tuple[dict[str, Any], float]:
    candidate_paths = _iter_list_paths(data)
    if not candidate_paths:
        raise ValueError("No candidate list paths found in source schema.")

    best_path: tuple[str, ...] | None = None
    best_score = 0.0
    best_kind = "dict"
    best_id_field: str | None = None

    for path in candidate_paths:
        node = _get_at_path(data, path)
        if isinstance(node, list) and node and all(isinstance(x, str) for x in node):
            if best_path is None:
                best_path = path
                best_score = 1.0
                best_kind = "string"
                best_id_field = "__self__"
            continue
        if not (isinstance(node, list) and node and all(isinstance(x, dict) for x in node)):
            continue
        id_field, score = _best_field(node, ID_FIELD_CANDIDATES)
        if id_field is None:
            continue
        if score > best_score:
            best_path = path
            best_score = score
            best_kind = "dict"
            best_id_field = id_field

    if best_path is None:
        raise ValueError("Could not find a usable records list in source schema.")
    if best_kind == "dict" and (best_id_field is None or best_score < 0.6):
        raise ValueError("Could not determine a confident id field for source schema.")

    mapping: dict[str, Any] = {
        "records_path": _path_to_string(best_path),
        "id_field": best_id_field if best_kind == "dict" else "__self__",
        "label_field": None,
        "description_field": None,
        "aliases_field": None,
        "strategy": "deterministic",
    }

    node = _get_at_path(data, best_path)
    if best_kind == "dict":
        records = [x for x in node if isinstance(x, dict)]
        label_field, _ = _best_field(records, LABEL_FIELD_CANDIDATES)
        desc_field, _ = _best_field(records, DESCRIPTION_FIELD_CANDIDATES)
        alias_field, alias_score = _best_field(records, ALIASES_FIELD_CANDIDATES)
        mapping["label_field"] = label_field
        mapping["description_field"] = desc_field
        if alias_score >= 0.2:
            mapping["aliases_field"] = alias_field
    return mapping, best_score


async def _llm_mapping(data: Any, candidate_paths: list[tuple[str, ...]], llm_client: Any) -> dict[str, Any]:
    if llm_client is None:
        raise ValueError("LLM fallback requested but no LLM client is configured.")

    samples: dict[str, Any] = {}
    for path in candidate_paths[:10]:
        path_key = _path_to_string(path)
        node = _get_at_path(data, path)
        if isinstance(node, list):
            samples[path_key] = node[:5]

    prompt = "\n".join(
        [
            "You are selecting field mappings for schema ingestion.",
            "Return ONLY valid JSON with keys:",
            '{"records_path":"", "id_field":"", "label_field":"", "description_field":"", "aliases_field":""}',
            "Rules:",
            "- records_path must be one of the provided candidate_paths",
            "- id_field must map to a stable unique identifier",
            "- Use empty string for unknown optional fields",
            "candidate_paths JSON:",
            json.dumps([_path_to_string(p) for p in candidate_paths], ensure_ascii=True),
            "samples JSON:",
            json.dumps(samples, ensure_ascii=True),
        ]
    )
    raw = await llm_client.generate(prompt)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM mapping response is not a JSON object.")

    records_path = parsed.get("records_path")
    id_field = parsed.get("id_field")
    label_field = parsed.get("label_field")
    description_field = parsed.get("description_field")
    aliases_field = parsed.get("aliases_field")
    if not isinstance(records_path, str) or not isinstance(id_field, str):
        raise ValueError("LLM mapping response missing records_path/id_field.")

    allowed_paths = {_path_to_string(p) for p in candidate_paths}
    if records_path not in allowed_paths:
        raise ValueError(f"LLM selected unknown records_path: {records_path}")

    return {
        "records_path": records_path,
        "id_field": id_field.strip(),
        "label_field": label_field.strip() if isinstance(label_field, str) and label_field.strip() else None,
        "description_field": (
            description_field.strip() if isinstance(description_field, str) and description_field.strip() else None
        ),
        "aliases_field": aliases_field.strip() if isinstance(aliases_field, str) and aliases_field.strip() else None,
        "strategy": "llm_fallback",
    }


def _upsert_user_registry_entry(data_dir: Path, entry: dict[str, Any]) -> None:
    registry_path = data_dir / "user_vocabularies.json"
    if registry_path.exists():
        existing = _read_json(registry_path)
        if not isinstance(existing, list):
            raise ValueError(f"user registry must be a JSON array: {registry_path}")
        registry = [item for item in existing if isinstance(item, dict)]
    else:
        registry = []

    replaced = False
    for idx, item in enumerate(registry):
        if item.get("name") == entry["name"]:
            registry[idx] = {**item, **entry}
            replaced = True
            break
    if not replaced:
        registry.append(entry)

    registry = sorted(registry, key=lambda x: str(x.get("name", "")))
    _write_json(registry_path, registry)


def _load_user_registry(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "user_vocabularies.json"
    if not registry_path.exists():
        return []
    raw = _read_json(registry_path)
    if not isinstance(raw, list):
        raise ValueError(f"user registry must be a JSON array: {registry_path}")
    return [item for item in raw if isinstance(item, dict)]


def _write_user_registry(data_dir: Path, registry: list[dict[str, Any]]) -> None:
    registry_path = data_dir / "user_vocabularies.json"
    ordered = sorted(registry, key=lambda x: str(x.get("name", "")))
    _write_json(registry_path, ordered)


def update_user_schema_metadata(name: str, updates: dict[str, Any]) -> dict[str, Any]:
    if not _is_non_empty_string(name):
        raise ValueError("Schema name is required.")
    schema_name = name.strip()

    allowed_keys = {
        "label",
        "description",
        "score_mode",
        "prefilter_types",
        "supports_text",
        "supports_table",
        "supports_cpa",
        "text_intro",
        "table_intro",
        "cpa_intro",
    }
    invalid = [k for k in updates.keys() if k not in allowed_keys]
    if invalid:
        raise ValueError(f"Unsupported update fields: {sorted(invalid)}")

    data_dir = user_data_dir()
    registry = _load_user_registry(data_dir)

    idx = -1
    for i, entry in enumerate(registry):
        if entry.get("name") == schema_name:
            idx = i
            break
    if idx < 0:
        raise ValueError(f"User schema not found: {schema_name}")

    entry = dict(registry[idx])
    for key, value in updates.items():
        if value is None:
            continue
        if key == "score_mode":
            if value not in {"dense", "sparse"}:
                raise ValueError("score_mode must be 'dense' or 'sparse'.")
            entry[key] = value
        elif key in {"prefilter_types", "supports_text", "supports_table", "supports_cpa"}:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean.")
            entry[key] = value
        else:
            if not _is_non_empty_string(value):
                raise ValueError(f"{key} must be a non-empty string when provided.")
            entry[key] = value.strip()

    registry[idx] = entry
    _write_user_registry(data_dir, registry)
    reload_schema_registry()
    return {
        "schema": schema_name,
        "label": entry.get("label"),
        "description": entry.get("description"),
        "updated_fields": sorted([k for k, v in updates.items() if v is not None]),
    }


def delete_user_schema(name: str, remove_files: bool = True) -> dict[str, Any]:
    if not _is_non_empty_string(name):
        raise ValueError("Schema name is required.")
    schema_name = name.strip()
    data_dir = user_data_dir()
    registry = _load_user_registry(data_dir)

    kept: list[dict[str, Any]] = []
    removed_entry: dict[str, Any] | None = None
    for entry in registry:
        if entry.get("name") == schema_name and removed_entry is None:
            removed_entry = entry
        else:
            kept.append(entry)
    if removed_entry is None:
        raise ValueError(f"User schema not found: {schema_name}")

    _write_user_registry(data_dir, kept)

    removed_path: str | None = None
    if remove_files:
        schema_dir = data_dir / "user" / schema_name
        if schema_dir.exists():
            shutil.rmtree(schema_dir)
            removed_path = str(schema_dir)

    reload_schema_registry()
    return {
        "schema": schema_name,
        "removed": True,
        "removed_files": bool(remove_files),
        "removed_path": removed_path,
    }


async def _fetch_source(payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    source_type = str(payload.get("source_type") or "").lower()
    if source_type == "file":
        source_path = payload.get("source_path")
        if not _is_non_empty_string(source_path):
            raise ValueError("source_path is required for source_type='file'.")
        path = _resolve_source_file_path(source_path)
        _assert_file_source_allowed(path)
        if path.suffix.lower() not in ALLOWED_SOURCE_EXTENSIONS:
            raise ValueError(f"File source extension must be one of: {sorted(ALLOWED_SOURCE_EXTENSIONS)}")
        data = path.read_bytes()
        return data, {"type": "file", "path": str(path)}

    if source_type == "url":
        source_url = payload.get("source_url")
        if not _is_non_empty_string(source_url):
            raise ValueError("source_url is required for source_type='url'.")
        url = source_url.strip()
        _validate_source_url(url)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
        return data, {"type": "url", "url": url}

    if source_type == "github":
        repo = payload.get("github_repo")
        ref = payload.get("github_ref") or "main"
        rel_path = payload.get("github_path")
        if not _is_non_empty_string(repo) or not _is_non_empty_string(rel_path):
            raise ValueError("github_repo and github_path are required for source_type='github'.")
        _validate_github_source(repo, str(ref), rel_path)
        repo_str = repo.strip().removeprefix("https://github.com/").strip("/")
        raw_url = f"https://raw.githubusercontent.com/{repo_str}/{str(ref).strip()}/{rel_path.strip().lstrip('/')}"
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(raw_url)
            resp.raise_for_status()
            data = resp.content
        return data, {"type": "github", "repo": repo_str, "ref": str(ref).strip(), "path": rel_path.strip()}

    raise ValueError("source_type must be one of: file, url, github")


def _build_registry_entry(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    registry_entry: dict[str, Any] = {
        "name": schema_name,
        "label": payload.get("label") or schema_name,
        "description": payload.get("description") or f"User-ingested schema '{schema_name}'.",
        "type_source": f"user/{schema_name}/type_ids.json",
        "score_mode": payload.get("score_mode") or "sparse",
        "supports_text": bool(payload.get("supports_text", True)),
        "supports_table": bool(payload.get("supports_table", True)),
        "supports_cpa": bool(payload.get("supports_cpa", False)),
        "prefilter_types": bool(payload.get("prefilter_types", False)),
    }
    for key in ("text_intro", "table_intro", "cpa_intro"):
        value = payload.get(key)
        if _is_non_empty_string(value):
            registry_entry[key] = value.strip()
    return registry_entry


def _build_guardrails_report(
    *,
    schema_name: str,
    source_type: str,
    type_ids: list[str],
    mapping_strategy: str,
    mapping_confidence: float | None,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    findings.extend({"severity": "warning", **w} for w in warnings)
    if mapping_strategy == "llm_fallback":
        findings.append(
            {
                "severity": "warning",
                "code": "llm_fallback_used",
                "reason": "Deterministic extraction was insufficient and LLM fallback was used.",
            }
        )
    passed = True
    if len(type_ids) < MIN_TYPES_PER_SCHEMA:
        findings.append(
            {
                "severity": "error",
                "code": "too_few_types",
                "reason": f"Schema must include at least {MIN_TYPES_PER_SCHEMA} types.",
            }
        )
        passed = False
    if len(type_ids) > MAX_TYPES_PER_SCHEMA:
        findings.append(
            {
                "severity": "error",
                "code": "too_many_types",
                "reason": f"Schema exceeds max type count ({MAX_TYPES_PER_SCHEMA}).",
            }
        )
        passed = False
    return {
        "passed": passed,
        "schema": schema_name,
        "source_type": source_type,
        "type_count": len(type_ids),
        "mapping_strategy": mapping_strategy,
        "mapping_confidence": mapping_confidence,
        "findings": findings,
    }


async def _prepare_schema_ingest(
    payload: dict[str, Any],
    llm_client: Any | None = None,
    *,
    enforce_guardrails: bool = True,
) -> dict[str, Any]:
    name = payload.get("name")
    if not _is_non_empty_string(name):
        raise ValueError("Schema name is required.")
    schema_name = name.strip()
    _validate_schema_name(schema_name)

    data_dir = user_data_dir()
    raw_data, source_meta = await _fetch_source(payload)
    if len(raw_data) > MAX_SOURCE_BYTES:
        raise ValueError(f"Source payload too large ({len(raw_data)} bytes). Max allowed is {MAX_SOURCE_BYTES} bytes.")

    try:
        parsed = json.loads(raw_data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Source schema is not valid UTF-8 JSON.") from exc

    used_llm = False
    warnings: list[dict[str, Any]] = []
    mapping: dict[str, Any]
    confidence = 0.0
    try:
        mapping, confidence = _deterministic_mapping(parsed)
    except Exception as exc:
        if not bool(payload.get("use_llm_fallback", False)):
            raise
        candidate_paths = _iter_list_paths(parsed)
        mapping = await _llm_mapping(parsed, candidate_paths, llm_client=llm_client)
        used_llm = True
        warnings.append(
            {
                "code": "deterministic_mapping_failed",
                "reason": str(exc),
            }
        )

    if confidence < 0.6 and bool(payload.get("use_llm_fallback", False)):
        candidate_paths = _iter_list_paths(parsed)
        mapping = await _llm_mapping(parsed, candidate_paths, llm_client=llm_client)
        used_llm = True
        warnings.append(
            {
                "code": "deterministic_low_confidence",
                "confidence": confidence,
                "threshold": 0.6,
            }
        )

    records_path = _string_to_path(mapping["records_path"])
    records = _get_at_path(parsed, records_path)
    if not isinstance(records, list) or not records:
        raise ValueError("Selected records_path does not point to a non-empty list.")

    id_field = mapping["id_field"]
    label_field = mapping.get("label_field")
    description_field = mapping.get("description_field")
    aliases_field = mapping.get("aliases_field")

    if id_field == "__self__":
        normalized, type_ids = _normalize_records(
            records,
            id_field="id",
            label_field=None,
            description_field=None,
            aliases_field=None,
        )
    else:
        normalized, type_ids = _normalize_records(
            records,
            id_field=id_field,
            label_field=label_field,
            description_field=description_field,
            aliases_field=aliases_field,
        )

    if not type_ids:
        raise ValueError("No type ids extracted from source.")

    mapping_hash = _mapping_sha256(mapping)
    expected_source_sha256 = payload.get("expected_source_sha256")
    if _is_non_empty_string(expected_source_sha256):
        expected = expected_source_sha256.strip().lower()
        actual = hashlib.sha256(raw_data).hexdigest()
        if expected != actual:
            raise ValueError(
                "Source hash mismatch between preview and activation. "
                f"expected_source_sha256={expected}, actual={actual}"
            )
    expected_mapping_sha256 = payload.get("expected_mapping_sha256")
    if _is_non_empty_string(expected_mapping_sha256):
        expected = expected_mapping_sha256.strip().lower()
        if expected != mapping_hash:
            raise ValueError(
                "Mapping hash mismatch between preview and activation. "
                f"expected_mapping_sha256={expected}, actual={mapping_hash}"
            )

    mapping_strategy = "llm_fallback" if used_llm else "deterministic"
    mapping_confidence = confidence if confidence > 0 else None
    registry_entry = _build_registry_entry(schema_name, payload)
    guardrails = _build_guardrails_report(
        schema_name=schema_name,
        source_type=str(payload.get("source_type") or ""),
        type_ids=type_ids,
        mapping_strategy=mapping_strategy,
        mapping_confidence=mapping_confidence,
        warnings=warnings,
    )
    if enforce_guardrails and not guardrails.get("passed", False):
        findings = guardrails.get("findings") or []
        error_reasons = [f.get("reason") for f in findings if isinstance(f, dict) and f.get("severity") == "error"]
        raise ValueError(f"Schema guardrails failed: {error_reasons}")

    return {
        "schema": schema_name,
        "raw_data": raw_data,
        "source": source_meta,
        "source_sha256": hashlib.sha256(raw_data).hexdigest(),
        "mapping_sha256": mapping_hash,
        "mapping": mapping,
        "mapping_strategy": mapping_strategy,
        "mapping_confidence": mapping_confidence,
        "warnings": warnings,
        "guardrails": guardrails,
        "normalized_types": normalized,
        "type_ids": type_ids,
        "registry_entry": registry_entry,
    }


async def preview_schema_payload(payload: dict[str, Any], llm_client: Any | None = None) -> dict[str, Any]:
    prepared = await _prepare_schema_ingest(payload, llm_client=llm_client, enforce_guardrails=False)
    type_ids = prepared["type_ids"]
    return {
        "schema": prepared["schema"],
        "label": prepared["registry_entry"]["label"],
        "description": prepared["registry_entry"]["description"],
        "type_count": len(type_ids),
        "sample_type_ids": type_ids[:20],
        "source": prepared["source"],
        "source_sha256": prepared["source_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "mapping_strategy": prepared["mapping_strategy"],
        "mapping_confidence": prepared["mapping_confidence"],
        "mapping": prepared["mapping"],
        "warnings": prepared["warnings"],
        "guardrails": prepared["guardrails"],
        "can_activate": bool(prepared["guardrails"].get("passed", False)),
        "activated": False,
    }


async def ingest_schema_payload(payload: dict[str, Any], llm_client: Any | None = None) -> dict[str, Any]:
    prepared = await _prepare_schema_ingest(payload, llm_client=llm_client)
    schema_name = prepared["schema"]
    data_dir = user_data_dir()
    schema_dir = data_dir / "user" / schema_name
    types_path = schema_dir / "types.json"
    type_ids_path = schema_dir / "type_ids.json"
    manifest_path = schema_dir / "manifest.json"
    registry_entry = prepared["registry_entry"]
    type_ids = prepared["type_ids"]

    manifest = {
        "name": schema_name,
        "label": registry_entry["label"],
        "description": registry_entry["description"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": prepared["source"],
        "source_sha256": prepared["source_sha256"],
        "mapping": prepared["mapping"],
        "counts": {"types": len(type_ids)},
        "paths": {
            "types": str(types_path.relative_to(data_dir).as_posix()),
            "type_ids": str(type_ids_path.relative_to(data_dir).as_posix()),
        },
    }

    _write_json(types_path, prepared["normalized_types"])
    _write_json(type_ids_path, type_ids)
    _write_json(manifest_path, manifest)
    _upsert_user_registry_entry(data_dir, registry_entry)
    schemas = reload_schema_registry()

    return {
        "schema": schema_name,
        "label": registry_entry["label"],
        "type_count": len(type_ids),
        "source": prepared["source"],
        "source_sha256": prepared["source_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "mapping_strategy": prepared["mapping_strategy"],
        "mapping_confidence": prepared["mapping_confidence"],
        "mapping": prepared["mapping"],
        "warnings": prepared["warnings"],
        "guardrails": prepared["guardrails"],
        "activated": schema_name in schemas,
    }
