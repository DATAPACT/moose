"""Fit the confidence of anonymisation's masking decisions on TAB train; check it on test.

Laya alone judges masking poorly (on the test run, AUC 0.70 for P(mask) against TAB, and a
mean confidence of 0.56 in decisions that are 82% right), so its answer is combined with
what the decision was: a logistic regression on the entity's category and decision (masked
or left) and the log-odds of Laya's probability that the decision is right. The weights go
into src/moose/data/anonymise/laya_calibration.json, which moose.anonymise reads; the file
says the confidence is calibrated when the test ECE is at most --max-ece.

Inputs are laya_mask.py's scores (laya_mask.jsonl) of a train run and of a test run.

    PYTHONPATH=../../src ../.venv-laya/bin/python fit_mask_calibration.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from laya_mask import SCORES, auc, ece
from moose.anonymise import CALIBRATION_PATH, decision_key, laya_logit
from moose.linkers import DEFAULT_LAYA_MODEL

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "runs" / "anonymise_train_openai_gpt-oss-120b"
TEST = HERE / "runs" / "anonymise_test_openai_gpt-oss-120b"


def load(run: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run / SCORES).read_text(encoding="utf-8").splitlines()]


def features(rows: list[dict[str, Any]], keys: list[str], with_laya: bool) -> np.ndarray:
    out = []
    for r in rows:
        f = [float(decision_key(r["category"], r["llm_masked"]) == k) for k in keys]
        out.append(f + ([laya_logit(r["plain"], r["llm_masked"])] if with_laya else []))
    return np.array(out)


def check(conf: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {"auc": round(auc(conf, right), 3), "ece": round(ece(conf, right), 3), "mean_confidence": round(float(conf.mean()), 3),
            "accuracy": round(float(right.mean()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-ece", type=float, default=0.10)
    parser.add_argument("--write", action="store_true", help=f"write {CALIBRATION_PATH.name}")
    args = parser.parse_args()

    train, test = load(TRAIN), load(TEST)
    keys = sorted({decision_key(r["category"], m) for r in train + test for m in (True, False)})
    y_train = np.array([r["llm_masked"] == r["gold"] for r in train], dtype=float)
    y_test = np.array([r["llm_masked"] == r["gold"] for r in test], dtype=float)
    print(f"train: {len(train)} entities of {len({r['doc_id'] for r in train})} cases; "
          f"test: {len(test)} entities of {len({r['doc_id'] for r in test})} cases")

    raw = np.array([1 / (1 + np.exp(-laya_logit(r["plain"], r["llm_masked"]))) for r in test])
    print("Laya alone (test)          ", check(raw, y_test))
    results = {}
    for name, with_laya in (("category and decision", False), ("category, decision, Laya", True)):
        model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000).fit(features(train, keys, with_laya), y_train)
        conf = model.predict_proba(features(test, keys, with_laya))[:, 1]
        results[name] = (model, check(conf, y_test))
        print(f"{name:27}(test)", results[name][1])

    model, checked = results["category, decision, Laya"]
    weights = dict(zip(keys, (round(float(w), 4) for w in model.coef_[0][:-1])))
    validated = checked["ece"] <= args.max_ece
    cases = len({r["doc_id"] for r in train})
    data = {
        "model": DEFAULT_LAYA_MODEL,
        "weights": weights,
        "laya_weight": round(float(model.coef_[0][-1]), 4),
        "validated_for": ["text"] if validated else [],
        "fitted_on": f"TAB train, {cases} random cases ({len(train)} entities, first mentions), protecting the applicant; "
                     "decisions by openai/gpt-oss-120b. research/tab_eval/fit_mask_calibration.py",
        "checked_on": f"TAB test, {len({r['doc_id'] for r in test})} random cases ({len(test)} entities): decisions "
                      f"{checked['accuracy']} right, mean confidence {checked['mean_confidence']}, ECE {checked['ece']}, "
                      f"AUC {checked['auc']}.",
    }
    print(json.dumps(data, indent=2))
    if args.write:
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("written:", CALIBRATION_PATH)


if __name__ == "__main__":
    main()
