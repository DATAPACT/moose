FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples/schema_ingest_samples ./examples/schema_ingest_samples

# MOOSE_EXTRAS=laya adds the local Laya linker (PyTorch, about 1 GB more).
ARG MOOSE_EXTRAS=""
# CPU-only PyTorch first: the default Linux wheel bundles CUDA (several GB) for no use here.
RUN case ",$MOOSE_EXTRAS," in *,laya,*) pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ;; esac \
    && if [ -n "$MOOSE_EXTRAS" ]; then pip install --no-cache-dir -e ".[$MOOSE_EXTRAS]"; else pip install --no-cache-dir -e .; fi

EXPOSE 8000

CMD ["uvicorn", "moose_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
