# Moose

Moose annotates personal data and privacy roles in text and tables with the W3C
[Data Privacy Vocabulary (DPV)](https://w3c.github.io/dpv/), or with your own schema.

- **An LLM finds and describes** the mentions or columns.
- **A local System 1 model,
  [Laya](https://huggingface.co/convaiinnovations/laya), gives every probability.** LLMs
  are over-confident, so Moose never asks them for a number.

As a library, Moose runs on your machine. LLM calls go straight from your process to your
provider with your key, and Laya runs locally. Nothing passes through a Moose server.

## Install

```bash
pip install "moose-annotate[laya]"
```

The package is imported as `moose`.

- **`laya`:** installs the System 1 model, which pulls in PyTorch. Its checkpoint, about
  800 MB, downloads from Hugging Face on first use. Without `laya`, results have no
  probability (`"confidence": null`).
- **`api`:** installs the REST API server (`moose_api`), as in the
  [Docker image](https://github.com/DATAPACT/moose).

Python 3.12 or later is required.

## Quick start

```python
import os
from moose import Annotator, OpenRouterClient

moose = Annotator(OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"], model="openai/gpt-oss-120b"))

result = moose.annotate_text("Agent: Thanks, Luca Bianchi. Your customer ID is USR 661 204.")
for e in result["entities"]:
    print(e["text"], e["type_id"], e["confidence"], e["decided_by"])
# Luca Bianchi  dpv-pd:Identifier  0.49  system1+llm
# USR 661 204   dpv-pd:Identifier  0.81  system1

print(moose.last_usage.calls, moose.last_usage.cost_usd)  # LLM calls and cost of that call
```

The key can also be a callable, called on every request, that reads a keychain or secrets
manager:

```python
import keyring
llm = OpenRouterClient(api_key=lambda: keyring.get_password("moose", "openrouter"), model="openai/gpt-oss-120b")
```

**Other LLM clients:**
- `OllamaClient(host="http://localhost:11434", token=None, model="qwen3:8b")` for a local
  model;
- `OpenAICompatibleClient(base_url=..., model=..., api_key=...)` for any OpenAI-compatible
  endpoint;
- your own subclass of `moose.LLMClient` (one `async generate(prompt) -> str` method).

## What it does

| Method | Input | Output |
|---|---|---|
| `annotate_text(text, schema="dpv_privacy")` | a text | entities: span, `type_id`, `confidence` |
| `annotate_texts(texts, ...)` | several texts | one result per text |
| `annotate_table(rows, schema="dpv_privacy")` | sample rows (dicts of column → value) | a type per column |
| `column_relationships(rows, subject_column, schema="cpa")` | sample rows | the relation from the subject column to each other column |
| `analyze_privacy(text=... or rows=...)` | a text or a table | findings and recommended actions (GDPR policy pack) |
| `schemas()` | | available schemas |
| `add_schema(payload)`, `remove_schema(name)` | your own vocabulary | bring your own schema |

Each method has an async twin with an `a` prefix, for example `await moose.aannotate_text(...)`.
The sync methods also work inside a running event loop, such as a Jupyter notebook. Use
either the sync or the async methods with one LLM client: its connections belong to one
event loop.

Results have the same shape as the REST API's job results.

### Schemas

- **`dpv_privacy`** (the default): about 260 DPV personal-data categories and privacy roles.
- **`dpv_pd`:** DPV's personal-data categories only, about 400.
- **`dpv`:** all of DPV, 8,000+ concepts.
- **`coarse`, `fine`:** general named-entity types.
- **`cpa`, `schemaorg_cpa_v1`:** column relationships.
- Your own schemas: see `add_schema`. They are stored under `MOOSE_USER_DATA_DIR`, by
  default `~/.moose`.

## Probabilities

Every `confidence` comes from Laya, never from the LLM:
- **Laya decided** (`decided_by` is `system1` or `system1+llm`): the chosen type's
  probability, exactly as in `ranking`.
- **The LLM decided** (`decided_by` is `llm`): Laya's probability of "yes" to a plain
  question, such as "Is the mention Name?". It sees only that type's definition and is not
  told the LLM chose it.

The `linker` argument picks who decides:
- **`system1+llm`:** Laya decides, and consults the LLM when unsure. It is the default for
  `dpv_privacy`.
- **`system1`:** Laya alone.
- **`llm`:** the LLM decides. It is the default for `dpv`.

**`calibrated`** says whether those probabilities matched accuracy on labelled data
(DPVCallBench, expected calibration error of 0.10 or less):

| Schema | `system1` | `system1+llm` | `llm` |
|---|---|---|---|
| `dpv_privacy` text | 0.075 ✓ | 0.052 ✓ | 0.133 |
| `dpv_pd` text | 0.067 ✓ | 0.067 ✓ | 0.219 |
| `dpv` text | 0.254 | 0.238 | 0.176 |

Tables, column relationships and other schemas have no labelled data yet, so their
results show `"calibrated": false`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MOOSE_USER_DATA_DIR` | `~/.moose` | where your own schemas are stored |
| `MOOSE_LAYA_MODEL` | `convaiinnovations/laya` | another Laya checkpoint (a Hugging Face id or a folder) |
| `MOOSE_LAYA_DEVICE` | automatic | `cpu`, `cuda` or `mps` |

## Links

- Source, REST API and dashboard: <https://github.com/DATAPACT/moose>
- Licence: Apache-2.0
