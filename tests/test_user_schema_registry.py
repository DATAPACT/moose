from __future__ import annotations

import json
from pathlib import Path

import pytest

import moose.schema as schema_mod


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _patch_schema_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(schema_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(schema_mod, "VOCAB_REGISTRY_PATH", data_dir / "vocabularies.json")
    monkeypatch.delenv("MOOSE_USER_DATA_DIR", raising=False)
    return data_dir


def test_reload_schema_registry_loads_user_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _patch_schema_paths(tmp_path, monkeypatch)

    _write_json(data_dir / "vocabularies.json", [])
    _write_json(data_dir / "user" / "acme" / "type_ids.json", ["ACME:EMAIL", "ACME:PHONE"])
    _write_json(
        data_dir / "user_vocabularies.json",
        [
            {
                "name": "acme_schema",
                "label": "Acme Schema",
                "type_source": "user/acme/type_ids.json",
                "score_mode": "sparse",
                "supports_text": True,
                "supports_table": True,
            }
        ],
    )

    names = schema_mod.reload_schema_registry()
    assert "acme_schema" in names

    cfg = schema_mod.get_schema_config("acme_schema")
    assert cfg.label == "Acme Schema"
    assert cfg.load_type_ids() == ["ACME:EMAIL", "ACME:PHONE"]


def test_reload_schema_registry_rejects_duplicate_names_across_registries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_schema_paths(tmp_path, monkeypatch)

    _write_json(data_dir / "vocabularies.json", [{"name": "dup", "type_source": "base.json"}])
    _write_json(data_dir / "base.json", ["X:A"])
    _write_json(data_dir / "user_vocabularies.json", [{"name": "dup", "type_source": "user/dup/type_ids.json"}])
    _write_json(data_dir / "user" / "dup" / "type_ids.json", ["Y:B"])

    with pytest.raises(ValueError, match="Duplicate schema name: dup"):
        schema_mod.reload_schema_registry()


def test_get_schema_config_refreshes_stale_cache_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_schema_paths(tmp_path, monkeypatch)

    # Start with no user schemas so cache is initialized stale.
    _write_json(data_dir / "vocabularies.json", [])
    _write_json(data_dir / "user_vocabularies.json", [])
    schema_mod.reload_schema_registry()

    # Simulate a schema being ingested by another worker/process.
    _write_json(data_dir / "user" / "acme" / "type_ids.json", ["ACME:EMAIL", "ACME:PHONE"])
    _write_json(
        data_dir / "user_vocabularies.json",
        [
            {
                "name": "acme_schema",
                "label": "Acme Schema",
                "type_source": "user/acme/type_ids.json",
                "score_mode": "sparse",
                "supports_text": True,
                "supports_table": True,
            }
        ],
    )

    cfg = schema_mod.get_schema_config("acme_schema")
    assert cfg.label == "Acme Schema"
    assert cfg.load_type_ids() == ["ACME:EMAIL", "ACME:PHONE"]
