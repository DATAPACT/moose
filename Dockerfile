FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples/schema_ingest_samples ./examples/schema_ingest_samples

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "moose_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
