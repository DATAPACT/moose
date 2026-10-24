"""Fine-tune Laya to choose DPV concepts among two-step candidates.

Training rows are built with Moose's own Laya inputs (moose.linkers.laya_inputs) from a
two-step run's mentions, so the model learns exactly what the server will ask it: each
mention alone and with the LLM's suggestion, as system1+llm asks it. The
target spreads probability over every candidate that maps to the gold concept. Options
are shuffled each epoch (Laya otherwise learns a position prior), and Laya's training
loop fits per-option-count temperatures on a held-out slice, which the checkpoint
applies at prediction time.

    ../.venv-laya/bin/python finetune_laya.py --train train --eval dev --freeze-encoder

A full fine-tune (all 395M parameters) with the default RLCD loss needs more than the
18 GB of a MacBook M3 Pro and stalls in swap; --freeze-encoder trains only the decision
head, and --micro-batch 2 --gradient-checkpointing --loss soft-ce makes a full one fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import bench
from linking_data import correct, mentions
from moose.linkers import DEFAULT_LAYA_MODEL, laya_inputs

MODELS = Path(__file__).resolve().parents[1] / "models"


def rows(split: str) -> list[dict[str, Any]]:
    """One row per mention as Laya sees it alone, and one with the LLM's suggestion (as in
    system1+llm), so the model learns to use the suggestion without simply copying it."""
    out = []
    for item in mentions(split):
        mention = item["mention"]
        label = next((c["label"] for c in mention["candidates"] if c["id"] == item["llm_choice"]), None)
        for suggestion in (None, label) if label else (None,):
            state, questions = laya_inputs(item["text"], mention, suggestion)
            options = questions["category"]["criteria"]
            right = [key for key, record in zip(options, mention["candidates"]) if correct(record["id"], item["gold"])]
            if not right:
                continue  # gold concept not among the candidates: nothing to learn here
            probabilities = {key: (1.0 / len(right) if key in right else 0.0) for key in options}
            out.append({"state": state, "questions": questions, "gold": {"category": {"probabilities": probabilities}}})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", default=["train"])
    parser.add_argument("--eval", default="dev")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the decision head.")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    from laya.train import TrainConfig, finetune, resolve_checkpoint_dir

    data_dir = bench.RUNS / "laya_finetune"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [r for split in args.train for r in rows(split)]
    train_path = data_dir / f"train_{'+'.join(args.train)}.jsonl"
    bench.write_jsonl(train_path, train_rows)
    eval_path = None
    if args.eval:
        eval_path = data_dir / f"eval_{args.eval}.jsonl"
        bench.write_jsonl(eval_path, rows(args.eval))
    print(f"{len(train_rows)} training rows from {args.train}")

    variant = f"{args.loss}{'-head' if args.freeze_encoder else ''}"
    out_dir = MODELS / (args.name or f"laya-dpv-privacy-{'+'.join(args.train)}-{variant}")
    config = TrainConfig(
        epochs=args.epochs,
        loss=args.loss,
        freeze_encoder=args.freeze_encoder,
        micro_batch=args.micro_batch,
        grad_accum=max(1, 64 // args.micro_batch),
        gradient_checkpointing=args.gradient_checkpointing or None,
        log_every=10,
        shuffle_options=("choice",),
        eval_data=str(eval_path) if eval_path else None,
    )
    summary = finetune(str(train_path), resolve_checkpoint_dir(DEFAULT_LAYA_MODEL), str(out_dir), config=config, device="auto")
    (out_dir / "moose_finetune_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("history",)}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
