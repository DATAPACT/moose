"""Laya's probability that one class is right, asked as a plain yes/no question.

For each mention the LLM typed, Laya is asked whether the mention is that class (label and
definition), with no other candidates and no mention of the LLM: moose.linkers'
laya_verify_inputs, as Moose's llm linker asks it. P(yes) is checked against
whether the LLM's choice matches gold. Only mentions whose span matches gold are used.

    ../.venv-laya/bin/python verify_choice.py --split dev --limit 50
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import bench
from compare_linkers import metrics
from linking_data import correct, mentions
from moose.linkers import laya_scorer, laya_verify_inputs
from moose.schema import get_schema_config

from check_calibration import RUNS, SCORES, VERIFY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", default="dpv_privacy")
    parser.add_argument("--split", default="dev", choices=list(RUNS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = {r["id"]: r for r in get_schema_config(args.schema).load_type_records()}
    run = RUNS[args.split].format(schema=args.schema)
    items = [m for m in mentions(args.split, run.replace(f"_{args.split}", "_{split}"), args.schema)]
    # The same mentions as the cached Laya scores, for a like-for-like comparison.
    cached = bench.read_jsonl(bench.RUNS / run / SCORES)
    assert len(cached) == len(items), (len(cached), len(items))
    for index, (m, c) in enumerate(zip(items, cached)):
        m["index"], m["hinted"] = index, (c["hinted"] or {}).get(m["llm_choice"])
    items = [m for m in items if m["llm_choice"] in records and m["hinted"] is not None]
    if args.limit:
        items = items[: args.limit]
    agent = laya_scorer("convaiinnovations/laya")._agent
    rows, t = [], time.time()
    for m in items:
        state, q = laya_verify_inputs(m["text"], m["mention"], records[m["llm_choice"]])
        p = float(agent.predict(state, q)["answers"]["verdict"]["probabilities"]["yes"])
        rows.append({"index": m["index"], "id": m["id"], "llm_choice": m["llm_choice"], "gold": m["gold"], "p_yes": p, "p_hinted": m["hinted"], "right": correct(m["llm_choice"], m["gold"])})
    x = np.array([r["p_yes"] for r in rows]); y = np.array([r["right"] for r in rows], float)
    h = np.array([r["p_hinted"] for r in rows])
    print(f"{args.split}: {len(rows)} mentions in {time.time() - t:.0f}s")
    print("  yes/no  ", json.dumps(metrics(x, y)))
    print("  hinted  ", json.dumps(metrics(h, y)))
    bench.write_jsonl(bench.RUNS / run / VERIFY, rows)


if __name__ == "__main__":
    main()
