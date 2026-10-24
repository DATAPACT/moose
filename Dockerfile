FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY docs/library.md ./docs/library.md
COPY src ./src
COPY examples/schema_ingest_samples ./examples/schema_ingest_samples

# The System 1 model (Laya) gives every probability Moose reports, so the server image
# includes it by default (CPU PyTorch, about 1 GB). MOOSE_EXTRAS="" builds without it:
# decisions then carry no probability.
ARG MOOSE_EXTRAS="laya"
# CPU-only PyTorch first: the default Linux wheel bundles CUDA (several GB) for no use here.
RUN case ",$MOOSE_EXTRAS," in *,laya,*) pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ;; esac \
    && if [ -n "$MOOSE_EXTRAS" ]; then pip install --no-cache-dir -e ".[api,$MOOSE_EXTRAS]"; else pip install --no-cache-dir -e ".[api]"; fi

EXPOSE 8000

CMD ["uvicorn", "moose_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
