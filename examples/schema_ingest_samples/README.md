# Schema Ingest Samples

This folder provides 3 sample schemas and matching sample data for testing:

- `POST /schemas/ingest`
- `POST /schemas/{schema}/ner`
- `POST /schemas/{schema}/tabular/annotate`

## Layout

- `schemas/`: source schema files to ingest
- `data/`: sample text/table data to annotate after ingest
- `requests/`: ready-made JSON payloads for API calls

## Suggested flow

1. Ingest schema using one of the request payloads in `requests/ingest_*.json`.
2. Poll `GET /jobs/{job_id}` until completed.
3. Submit annotation requests using `requests/annotate_*.json`.

## Example commands

```bash
curl -s -X POST http://localhost:8000/schemas/ingest \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d @examples/schema_ingest_samples/requests/ingest_healthcare_file.json
```

```bash
curl -s -X POST http://localhost:8000/schemas/healthcare_pii/ner \
  -H "X-API-Key: test-dev-key" \
  -H "Content-Type: application/json" \
  -d @examples/schema_ingest_samples/requests/annotate_healthcare_ner.json
```

