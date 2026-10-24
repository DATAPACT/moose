"""Write a self-contained Colab notebook that fine-tunes Laya on a GPU.

The notebook embeds the training and evaluation rows that finetune_laya.py builds (Laya's
`{state, questions, gold}` format, made with Moose's own Laya inputs), so it needs no other
file. Rebuild the rows first if the two-step runs change:

    ../.venv-laya/bin/python finetune_laya.py --help        # rows are written by a run
    ../.venv-laya/bin/python make_colab_notebook.py
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import bench

DATA = bench.RUNS / "laya_finetune"
FILES = {"train.jsonl": DATA / "train_train.jsonl", "dev.jsonl": DATA / "eval_dev.jsonl"}
OUT = Path(__file__).resolve().parent / "colab" / "finetune_laya_dpv_privacy_v2.ipynb"
LAYA_VERSION = "0.4.13"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main() -> None:
    payload, sums, counts = {}, {}, {}
    for name, path in FILES.items():
        raw = path.read_bytes()
        payload[name] = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        sums[name] = hashlib.sha256(raw).hexdigest()
        counts[name] = sum(1 for line in raw.splitlines() if line.strip())

    cells = [
        md(f"""
# Fine-tune Laya for DPV concept linking (Moose)

Fine-tunes [Laya](https://huggingface.co/convaiinnovations/laya) to choose a mention's DPV concept
among the candidates Moose's two-step mode retrieves, so it gives **calibrated probabilities**.

- **Data:** embedded below. It is {counts['train.jsonl']} training rows (DPVCallBench v0.1 train) and
  {counts['dev.jsonl']} evaluation rows (dev). Each mention appears twice: once as Laya sees it alone, and once
  with the LLM's suggestion, as Moose's default `system1+llm` asks it. All of it is synthetic call-transcript text, with no
  real personal data. Running this notebook uploads the data to Google Colab. The dataset's licence
  is not final yet, so don't share the notebook outside the project.
- **Runtime:** choose *Runtime → Change runtime type → T4 GPU* (free). It takes about 10–20 minutes.
- **Result:** a checkpoint folder of about 800 MB, saved to your Google Drive or downloaded as a zip.
  Moose loads it with `laya.load(path)`.

Run the cells from top to bottom.
"""),
        md("## 1. Check the GPU"),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "No GPU: choose Runtime > Change runtime type > T4 GPU")
"""),
        md("## 2. Install Laya"),
        code(f"""
%pip install -q "laya=={LAYA_VERSION}" protobuf sentencepiece
import laya, torch, transformers
print("laya", "{LAYA_VERSION}", "| torch", torch.__version__, "| transformers", transformers.__version__, "| CUDA", torch.cuda.is_available())
"""),
        md("""
## 3. Settings

- `TRAIN_ON`: use `["train"]` to measure on dev, which is what we compare against. Use `["train", "dev"]`
  only for a final model once the setup is fixed (then there is no held-out evaluation).
- `LOSS`: `"soft-ce"` is plain cross-entropy against the target distribution. `"rlcd"` is Laya's own
  objective with proper scoring rules; it is slower and needs more memory.
"""),
        code("""
TRAIN_ON = ["train"]
EVAL_ON = "dev"            # set to None when TRAIN_ON includes "dev"
LOSS = "soft-ce"
EPOCHS = 4
MICRO_BATCH = 8            # 8 x 8 accumulation = effective batch 64, as in Laya's own recipe
OUTPUT_DIR = "/content/laya-dpv-privacy"
SAVE_TO_DRIVE = True       # False: download a zip instead
"""),
        md("## 4. Write the embedded data"),
        code(f"""
import base64, gzip, hashlib, json, os
DATA = {json.dumps(payload)}
SHA256 = {json.dumps(sums)}
os.makedirs("/content/data", exist_ok=True)
for name, blob in DATA.items():
    raw = gzip.decompress(base64.b64decode(blob))
    assert hashlib.sha256(raw).hexdigest() == SHA256[name], f"{{name}} is corrupted"
    open(f"/content/data/{{name}}", "wb").write(raw)
    print(name, sum(1 for line in raw.splitlines() if line.strip()), "rows")

train_path = "/content/data/train_combined.jsonl"
with open(train_path, "w") as out:
    for split in TRAIN_ON:
        out.write(open(f"/content/data/{{split}}.jsonl").read())
eval_path = f"/content/data/{{EVAL_ON}}.jsonl" if EVAL_ON else None
row = json.loads(open("/content/data/dev.jsonl").readline())
print("\\nExample state:\\n" + row["state"])
print("Options:", list(row["questions"]["category"]["criteria"].values())[:4], "...")
"""),
        md("## 5. Fine-tune\n\nLaya fits per-option-count temperatures on a held-out slice of the training data and saves them with the checkpoint. When `EVAL_ON` is set, it prints accuracy and calibration before and after fine-tuning."),
        code("""
from laya.train import TrainConfig, finetune, resolve_checkpoint_dir, dry_run

config = TrainConfig(
    epochs=EPOCHS,
    loss=LOSS,
    micro_batch=MICRO_BATCH,
    grad_accum=max(1, 64 // MICRO_BATCH),
    shuffle_options=("choice",),   # Laya otherwise learns a bias for the first options
    eval_data=eval_path,
    log_every=20,
)
base = resolve_checkpoint_dir("convaiinnovations/laya")
dry_run(train_path, base, config)
summary = finetune(train_path, base, OUTPUT_DIR, config=config, device="cuda")
json.dump(summary, open(f"{OUTPUT_DIR}/moose_finetune_summary.json", "w"), indent=2, default=str)
"""),
        md("## 6. Quick check of the saved checkpoint"),
        code("""
agent = laya.load(OUTPUT_DIR)
row = json.loads(open("/content/data/dev.jsonl").readline())
answer = agent.predict(row["state"], row["questions"])["answers"]["category"]
best = sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])[:3]
criteria = row["questions"]["category"]["criteria"]
print(row["state"])
for key, p in best:
    print(f"  {p:.3f}  {criteria[key][:70]}")
print("gold:", [criteria[k][:40] for k, v in row["gold"]["category"]["probabilities"].items() if v > 0])
"""),
        md("## 7. Save the checkpoint\n\nWith `SAVE_TO_DRIVE`, the zip goes to *My Drive/moose/*. Download it from there to your Mac and give the path to Claude."),
        code("""
import shutil
zip_path = shutil.make_archive("/content/laya-dpv-privacy", "zip", OUTPUT_DIR)
print("zip:", zip_path, round(os.path.getsize(zip_path) / 1e6), "MB")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/moose", exist_ok=True)
    shutil.copy(zip_path, "/content/drive/MyDrive/moose/")
    print("saved to My Drive/moose/laya-dpv-privacy.zip")
else:
    from google.colab import files
    files.download(zip_path)
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {counts}")


if __name__ == "__main__":
    main()
