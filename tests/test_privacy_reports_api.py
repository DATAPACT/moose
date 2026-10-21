from __future__ import annotations

import pytest
from fastapi import HTTPException

from moose_api.main import ADMIN_CLIENT, _current_client, app, get_job_privacy_report, get_privacy_report_schema_endpoint
from moose_api.queue import JobRecord


@pytest.fixture(autouse=True)
def _as_admin():
    # These tests call the endpoint directly, so set the client the auth dependency would set.
    token = _current_client.set(ADMIN_CLIENT)
    yield
    _current_client.reset(token)


class _StubStore:
    def __init__(self, jobs: dict[str, JobRecord]) -> None:
        self._jobs = jobs

    async def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)


@pytest.mark.asyncio
async def test_get_job_privacy_report_returns_machine_content() -> None:
    machine_content = {
        "report_type": "privacy_analysis",
        "schema_id": "moose.privacy.machine_report.v1",
        "schema_version": "1.0.0",
        "summary": {"task_count": 1, "finding_count": 0},
        "tasks": [],
    }
    app.state.job_store = _StubStore(
        {
            "j1": JobRecord(
                job_id="j1",
                endpoint_type="privacy_analyze",
                payload={},
                status="completed",
                created_at="2026-03-08T00:00:00+00:00",
                updated_at="2026-03-08T00:00:01+00:00",
                retries=0,
                result={"reports": {"machine_readable": {"content": machine_content}}},
            )
        }
    )

    out = await get_job_privacy_report("j1")
    assert out["schema_id"] == "moose.privacy.machine_report.v1"
    assert out["schema_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_get_job_privacy_report_requires_completed_status() -> None:
    app.state.job_store = _StubStore(
        {
            "j2": JobRecord(
                job_id="j2",
                endpoint_type="privacy_analyze",
                payload={},
                status="processing",
                created_at="2026-03-08T00:00:00+00:00",
                updated_at="2026-03-08T00:00:01+00:00",
                retries=0,
            )
        }
    )

    with pytest.raises(HTTPException) as exc:
        await get_job_privacy_report("j2")
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_privacy_report_schema_endpoint_returns_expected_schema() -> None:
    schema = await get_privacy_report_schema_endpoint()
    assert schema.get("$id") == "moose.privacy.machine_report.v1"
    assert schema.get("x-schema-version") == "1.0.0"
    assert isinstance(schema.get("properties"), dict)
