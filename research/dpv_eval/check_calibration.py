"""Check, per schema and linker, whether Laya's probabilities are calibrated; write it for Moose.

Every probability Moose reports is Laya's probability of the chosen type, as in Laya's
ranking, unchanged (moose.linkers). This measures how well it matches the accuracy
observed on labelled data, separately for each linker, because each linker's probability
means something different:
  system1      Laya's probability of its own choice
  system1+llm  the same, after consulting the LLM below the schema's threshold
  llm          Laya's P(yes) that the LLM's choice is right, asked as a yes/no question
               (verify_choice.py caches it)

Uses the Laya scores that fit_system1.py caches next to the train, dev and test runs (each
mention scored alone and with the LLM's choice), pooled. A linker is marked validated for
text when its ECE is at most --max-ece. Nothing is fitted, so no held-out split is needed.
Only mentions whose span matches gold are used, so it checks P(type right | span right).

    ../.venv-laya/bin/python check_calibration.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

import bench
from compare_linkers import metrics
from linking_data import correct
from moose.linkers import Calibration
from moose.schema import DATA_DIR, get_schema_config

RUNS = {
    "train": "moose_{schema}_llm_final_train",
    "dev": "moose_{schema}_llm_openai_gpt-oss-120b_dev",
    "test": "moose_{schema}_llm_final_test",
}
SCORES = "system1_scores_convaiinnovations_laya.jsonl"
# Laya's yes/no P(yes) on the LLM's choice (verify_choice.py), the llm linker's probability.
VERIFY = "verify_convaiinnovations_laya.jsonl"
LINKERS = ("system1", "system1+llm", "llm")


def decisions(rows: list[dict[str, Any]], linker: str, cal: Calibration) -> list[tuple[float, bool] | None]:
    """(raw Laya probability, right?) for each mention, as Moose's linker would decide;
    None where the linker makes no decision (llm without a choice among the candidates)."""
    out: list[tuple[float, bool] | None] = []
    for r in rows:
        if linker == "llm":
            if r.get("p_yes") is None:
                out.append(None)
                continue
            out.append((r["p_yes"], correct(r["llm_choice"], r["gold"])))
            continue
        probs = r["plain"]
        choice = max(probs, key=probs.get)
        if linker == "system1+llm" and r["hinted"] and probs[choice] < cal.consult_below:
            probs = r["hinted"]
            choice = max(probs, key=probs.get)
        out.append((probs[choice], correct(choice, r["gold"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schemas", nargs="+", default=["dpv_privacy", "dpv_pd", "dpv"])
    parser.add_argument("--max-ece", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for schema_name in args.schemas:
        schema = get_schema_config(schema_name)
        cal = Calibration.load(schema.linker_calibration)
        rows, sizes = [], {}
        for split, run in RUNS.items():
            split_rows = bench.read_jsonl(bench.RUNS / run.format(schema=schema_name) / SCORES)
            # verify_choice.py rows carry the index of the mention among the cached scores.
            verified = {v["index"]: v["p_yes"] for v in bench.read_jsonl(bench.RUNS / run.format(schema=schema_name) / VERIFY)}
            for index, r in enumerate(split_rows):
                r["p_yes"] = verified.get(index)
            if split_rows:  # else this split was not scored for the schema
                rows += split_rows
                sizes[split] = len(split_rows)
        linkers, report[schema_name] = {}, {"mentions": sizes}
        for linker in LINKERS:
            pairs = [d for d in decisions(rows, linker, cal) if d is not None]
            x = np.array([p for p, _ in pairs])
            y = np.array([right for _, right in pairs], float)
            m = metrics(x, y)
            validated = m["ece"] <= args.max_ece
            linkers[linker] = {
                "validated_for": ["text"] if validated else [],
                "checked_on": f"DPVCallBench v0.1 train+dev+test ({len(x)} mentions with exact gold spans): "
                              f"accuracy {m['accuracy']}, ECE {m['ece']}. research/dpv_eval/check_calibration.py",
            }
            report[schema_name][linker] = {"n": len(x), **m, "validated": validated}
            print(f"{schema_name:12} {linker:12} n={len(x)} acc {m['accuracy']:.3f} ECE {m['ece']:.3f}{'  VALIDATED' if validated else ''}")
        if not args.dry_run:
            path = schema.linker_calibration
            if path is None or path == DATA_DIR / "linkers" / "laya_zero_shot.json":
                raise SystemExit(f"{schema_name} has no calibration file of its own")
            content = json.loads(path.read_text(encoding="utf-8"))
            content.pop("calibrators", None)
            content["linkers"] = linkers
            path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    (bench.RUNS / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
