from __future__ import annotations

import json
from pathlib import Path

import pytest

import moose.ingest as ingest_mod
import moose.schema as schema_mod


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _patch_data_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(schema_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(schema_mod, "VOCAB_REGISTRY_PATH", data_dir / "vocabularies.json")
    monkeypatch.delenv("MOOSE_USER_DATA_DIR", raising=False)
    _write_json(data_dir / "vocabularies.json", [])
    return data_dir


@pytest.mark.asyncio
async def test_ingest_schema_payload_from_file_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source.json"
    _write_json(
        source_path,
        {
            "types": [
                {"id": "ACME:EMAIL", "label": "Email"},
                {"id": "ACME:PHONE", "label": "Phone"},
                {"id": "ACME:MRN", "label": "Medical Record Number"},
            ]
        },
    )

    result = await ingest_mod.ingest_schema_payload(
        {
            "name": "acme_schema",
            "source_type": "file",
            "source_path": str(source_path),
            "supports_text": True,
            "supports_table": True,
        }
    )

    assert result["activated"] is True
    assert result["mapping_strategy"] == "deterministic"
    assert (data_dir / "user" / "acme_schema" / "types.json").exists()
    cfg = schema_mod.get_schema_config("acme_schema")
    assert cfg.load_type_ids() == ["ACME:EMAIL", "ACME:PHONE", "ACME:MRN"]


class _FakeLLMClient:
    async def generate(self, prompt: str) -> str:
        assert "records_path" in prompt
        return json.dumps(
            {
                "records_path": "payload.entries",
                "id_field": "token",
                "label_field": "display",
                "description_field": "",
                "aliases_field": "",
            }
        )


@pytest.mark.asyncio
async def test_ingest_schema_payload_llm_fallback_for_ambiguous_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_ambiguous.json"
    _write_json(
        source_path,
        {
            "payload": {
                "entries": [
                    {"token": "Z:ONE", "display": "One"},
                    {"token": "Z:TWO", "display": "Two"},
                    {"token": "Z:THREE", "display": "Three"},
                ]
            }
        },
    )

    result = await ingest_mod.ingest_schema_payload(
        {
            "name": "z_schema",
            "source_type": "file",
            "source_path": str(source_path),
            "use_llm_fallback": True,
            "supports_text": True,
            "supports_table": True,
        },
        llm_client=_FakeLLMClient(),
    )

    assert result["activated"] is True
    assert result["mapping_strategy"] == "llm_fallback"
    cfg = schema_mod.get_schema_config("z_schema")
    assert cfg.load_type_ids() == ["Z:ONE", "Z:TWO", "Z:THREE"]


@pytest.mark.asyncio
async def test_update_and_delete_user_schema_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_lifecycle.json"
    _write_json(source_path, {"types": [{"id": "L:ONE"}, {"id": "L:TWO"}, {"id": "L:THREE"}]})

    await ingest_mod.ingest_schema_payload(
        {
            "name": "life_schema",
            "source_type": "file",
            "source_path": str(source_path),
            "supports_text": True,
            "supports_table": True,
        }
    )
    cfg_before = schema_mod.get_schema_config("life_schema")
    assert cfg_before.supports_table is True

    updated = ingest_mod.update_user_schema_metadata(
        "life_schema",
        {
            "label": "Lifecycle Schema",
            "supports_table": False,
            "description": "Updated desc",
        },
    )
    assert updated["schema"] == "life_schema"
    cfg_after = schema_mod.get_schema_config("life_schema")
    assert cfg_after.label == "Lifecycle Schema"
    assert cfg_after.supports_table is False

    removed = ingest_mod.delete_user_schema("life_schema", remove_files=True)
    assert removed["removed"] is True
    assert not (data_dir / "user" / "life_schema").exists()
    with pytest.raises(ValueError, match="Unknown schema"):
        schema_mod.get_schema_config("life_schema")


@pytest.mark.asyncio
async def test_preview_schema_payload_does_not_activate_or_write_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_preview.json"
    _write_json(
        source_path,
        {
            "types": [
                {"id": "PRE:ONE", "label": "One"},
                {"id": "PRE:TWO", "label": "Two"},
            ]
        },
    )

    result = await ingest_mod.preview_schema_payload(
        {
            "name": "preview_schema",
            "source_type": "file",
            "source_path": str(source_path),
            "supports_text": True,
            "supports_table": True,
        }
    )

    assert result["can_activate"] is False
    assert result["activated"] is False
    assert result["mapping_strategy"] == "deterministic"
    assert result["type_count"] == 2
    assert result["sample_type_ids"] == ["PRE:ONE", "PRE:TWO"]
    assert isinstance(result.get("mapping_sha256"), str)
    assert result.get("guardrails", {}).get("passed") is False
    assert not (data_dir / "user" / "preview_schema").exists()
    with pytest.raises(ValueError, match="Unknown schema"):
        schema_mod.get_schema_config("preview_schema")


@pytest.mark.asyncio
async def test_preview_schema_payload_guardrails_pass_for_valid_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_preview_pass.json"
    _write_json(
        source_path,
        {
            "types": [
                {"id": "PRE:ONE", "label": "One"},
                {"id": "PRE:TWO", "label": "Two"},
                {"id": "PRE:THREE", "label": "Three"},
            ]
        },
    )

    result = await ingest_mod.preview_schema_payload(
        {
            "name": "preview_pass",
            "source_type": "file",
            "source_path": str(source_path),
            "supports_text": True,
            "supports_table": True,
        }
    )

    assert result.get("guardrails", {}).get("passed") is True
    assert result["type_count"] == 3


@pytest.mark.asyncio
async def test_ingest_schema_payload_rejects_preview_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_hash.json"
    _write_json(
        source_path,
        {
            "types": [
                {"id": "HX:ONE", "label": "One"},
                {"id": "HX:TWO", "label": "Two"},
                {"id": "HX:THREE", "label": "Three"},
            ]
        },
    )

    with pytest.raises(ValueError, match="Source hash mismatch"):
        await ingest_mod.ingest_schema_payload(
            {
                "name": "hash_mismatch",
                "source_type": "file",
                "source_path": str(source_path),
                "expected_source_sha256": "deadbeef",
                "supports_text": True,
                "supports_table": True,
            }
        )


@pytest.mark.asyncio
async def test_ingest_schema_payload_rejects_uppercase_schema_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_paths(tmp_path, monkeypatch)
    source_path = tmp_path / "source_upper.json"
    _write_json(source_path, {"types": [{"id": "U:ONE"}, {"id": "U:TWO"}, {"id": "U:THREE"}]})

    with pytest.raises(ValueError, match="must be lowercase"):
        await ingest_mod.ingest_schema_payload(
            {
                "name": "UpperCaseSchema",
                "source_type": "file",
                "source_path": str(source_path),
                "supports_text": True,
                "supports_table": True,
            }
        )


def test_resolve_source_file_path_remaps_host_absolute_examples_path() -> None:
    path = ingest_mod._resolve_source_file_path(
        "/Users/someone/work/moose/examples/schema_ingest_samples/schemas/healthcare_pii.json"
    )
    assert path.name == "healthcare_pii.json"
    assert path.exists()
