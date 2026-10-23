"""Choosing each item's type among its candidates, with a System 1 model deciding.

Moose's decisions are classifications: a mention's type, a column's type. A "System 1"
model, Laya, scores every candidate in one forward pass and gives probabilities; an LLM
adds what it knows. Three linkers:

  system1+llm  Laya decides. Items it is unsure about (top probability below the
               calibration's `consult_below`) go to the LLM, whose suggestion is added to
               Laya's input; Laya then decides again. The final probability is Laya's.
  system1      Laya alone; no LLM call for this step.
  llm          the LLM decides. With Laya installed, the probability of the LLM's choice
               is Laya's, given the LLM's suggestion; otherwise it is the number the LLM
               states. Neither is calibrated (`"calibrated": false`): on DPVCallBench,
               Laya's probability of the LLM's choice was far too low (ECE 0.23-0.36).

On DPVCallBench (two-step `dpv_privacy`), system1+llm with consult_below 0.6 matched or
beat both Laya and the LLM alone and halved the calibration error, asking the LLM about
one item in five or six. See research/dpv_eval.

Laya is optional (`pip install "moose[laya]"`); its checkpoint is downloaded from Hugging
Face on first use, or loaded from a local fine-tuned folder.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

LINKERS = ("system1+llm", "system1", "llm")
SYSTEM1_LINKERS = ("system1+llm", "system1")
DEFAULT_LAYA_MODEL = "convaiinnovations/laya"
# Characters of each candidate's definition shown to Laya: definitions help it a lot,
# but its options share a small token budget.
LAYA_DEFINITION_CHARS = 140

# ask_llm(items) -> {index in items: (type_id or "none", stated confidence)}, one LLM call.
AskLLM = Callable[[list[dict[str, Any]]], Awaitable[dict[int, tuple[str, float]]]]


class LinkerUnavailable(RuntimeError):
    """The requested linker cannot run here (e.g. Laya is not installed)."""


def laya_available() -> bool:
    try:
        import laya  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Calibration:
    """How a schema's System 1 decisions are made (see research/dpv_eval/fit_system1.py)."""

    # Hugging Face repo id or a local checkpoint folder (e.g. a fine-tuned Laya).
    # MOOSE_LAYA_MODEL overrides it.
    model: str = DEFAULT_LAYA_MODEL
    # system1+llm: ask the LLM about items whose top probability is below this.
    consult_below: float = 0.6
    # Tasks on which Laya's probabilities were checked against labelled data; elsewhere
    # they are reported as not calibrated.
    validated_for: tuple[str, ...] = ("text",)
    fitted_on: str | None = None

    @classmethod
    def load(cls, path: Path | None) -> "Calibration | None":
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: data[k] for k in ("model", "consult_below", "validated_for", "fitted_on") if k in data}
        if "validated_for" in known:
            known["validated_for"] = tuple(known["validated_for"])
        if os.getenv("MOOSE_LAYA_MODEL"):
            known["model"] = os.environ["MOOSE_LAYA_MODEL"]
        return cls(**known)


def system1_ready(calibration: Calibration | None) -> bool:
    """Laya is installed and its model has not just failed to load."""
    return calibration is not None and laya_available() and _recent_failure(calibration.model) is None


def _short(definition: str | None, limit: int = LAYA_DEFINITION_CHARS) -> str:
    text = (definition or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class Prompt:
    """How an item and its context are shown to Laya (text mentions, table columns, ...)."""

    context_label: str = "Call transcript"
    item_label: str = "Mention"
    instructions: str = "Which category of personal data or privacy role is the mention?"


TEXT_PROMPT = Prompt()
TABLE_PROMPT = Prompt(
    context_label="Table columns",
    item_label="Column and sample values",
    instructions="Which category of personal data or privacy role do the column's values hold?",
)


def laya_inputs(
    context: str, item: dict[str, Any], suggestion: str | None = None, prompt: Prompt = TEXT_PROMPT
) -> tuple[str, dict[str, Any]]:
    """Laya's state and question for one item. Candidates keep their retrieval order,
    which Laya uses as a cue (shuffling them lowers its accuracy). `suggestion` is the
    label the LLM chose, given to Laya as evidence."""
    state = f'{prompt.context_label}: "{context}"\n{prompt.item_label}: "{item["text"]}"'
    if item.get("category"):
        state += f' (described as: {item["category"]})'
    if suggestion:
        state += f"\nAn LLM suggested: {suggestion}"
    criteria = {}
    for i, record in enumerate(item["candidates"]):
        definition = _short(record.get("definition"))
        criteria[f"option_{i}"] = f"{record.get('label') or record['id']}" + (f": {definition}" if definition else "")
    return state, {"category": {"type": "choice", "instructions": prompt.instructions, "criteria": criteria}}


class LayaScorer:
    """One loaded Laya model, shared by all jobs. Predictions run one text at a time in a thread."""

    def __init__(self, model: str) -> None:
        if not laya_available():
            raise LinkerUnavailable('Laya is not installed on this server: install "moose[laya]".')
        import laya

        self._agent = laya.load(model, device=os.getenv("MOOSE_LAYA_DEVICE") or None)
        self._lock = threading.Lock()

    def _score_sync(
        self, context: str, items: list[dict[str, Any]], suggestions: list[str | None], prompt: Prompt
    ) -> list[dict[str, float]]:
        out = []
        with self._lock:
            for item, suggestion in zip(items, suggestions):
                state, questions = laya_inputs(context, item, suggestion, prompt)
                answer = self._agent.predict(state, questions)["answers"]["category"]
                ids = [c["id"] for c in item["candidates"]]
                out.append({ids[int(key.split("_")[1])]: float(p) for key, p in answer["probabilities"].items()})
        return out

    async def score(
        self,
        context: str,
        items: list[dict[str, Any]],
        suggestions: list[str | None] | None = None,
        prompt: Prompt = TEXT_PROMPT,
    ) -> list[dict[str, float]]:
        """Probability per candidate, for each item; `suggestions` are the LLM's labels.
        All items of one call are scored in one thread, so a text's items stay together."""
        return await asyncio.to_thread(self._score_sync, context, items, suggestions or [None] * len(items), prompt)


# Loaded models, and recent load failures (not retried for a while, so a broken download
# or a missing network does not make every job wait for another attempt).
_SCORERS: dict[str, LayaScorer] = {}
_FAILURES: dict[str, tuple[float, str]] = {}
_LOAD_LOCK = threading.Lock()
RETRY_LOAD_AFTER_SECONDS = 600


def _recent_failure(model: str) -> str | None:
    failure = _FAILURES.get(model)
    if failure and time.monotonic() - failure[0] < RETRY_LOAD_AFTER_SECONDS:
        return failure[1]
    return None


def laya_scorer(model: str = DEFAULT_LAYA_MODEL) -> LayaScorer:
    """The loaded scorer for `model`, loading it on first use (blocking: the first load can
    download about 800 MB). Async code uses get_scorer, which loads in a thread."""
    if model in _SCORERS:
        return _SCORERS[model]
    with _LOAD_LOCK:  # one load at a time; a second caller finds the first one's result
        if model in _SCORERS:
            return _SCORERS[model]
        failure = _recent_failure(model)
        if failure:
            raise LinkerUnavailable(failure)
        try:
            scorer = LayaScorer(model)
        except LinkerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - download, disk or memory problems
            message = f"The System 1 model {model!r} could not be loaded: {exc}"
            _FAILURES[model] = (time.monotonic(), message)
            raise LinkerUnavailable(message) from exc
        _FAILURES.pop(model, None)
        _SCORERS[model] = scorer
        return scorer


async def get_scorer(model: str) -> LayaScorer:
    """laya_scorer without blocking the event loop while the model downloads or loads."""
    if model in _SCORERS:
        return _SCORERS[model]
    return await asyncio.to_thread(laya_scorer, model)


def ranked(probabilities: dict[str, float], top: int = 3) -> list[dict[str, Any]]:
    best = sorted(probabilities.items(), key=lambda item: -item[1])[:top]
    return [{"type_id": type_id, "probability": round(p, 4)} for type_id, p in best]


@dataclass
class Decision:
    type_id: str  # or "none"
    confidence: float
    # "system1", "system1+llm" (System 1 after consulting the LLM) or "llm"
    decided_by: str
    calibrated: bool
    probabilities: dict[str, float] | None = None
    llm_suggestion: str | None = None


async def decide(
    linker: str,
    context: str,
    items: list[dict[str, Any]],
    calibration: Calibration | None,
    ask_llm: AskLLM,
    prompt: Prompt = TEXT_PROMPT,
    task: str = "text",
) -> list[Decision]:
    """Choose a type for each item. Each item has `text`, optionally `category`, and
    `candidates`: records with `id`, `label` and `definition`. `task` ("text", "table")
    says whether Laya's probabilities count as calibrated (see Calibration.validated_for)."""
    if not items:
        return []
    system1 = system1_ready(calibration)
    validated = system1 and task in calibration.validated_for
    if linker in SYSTEM1_LINKERS and not system1:
        failure = _recent_failure(calibration.model) if calibration else None
        raise LinkerUnavailable(failure or "This linker needs Laya and the schema's calibration; use the llm linker.")

    decisions: list[Decision | None] = [None] * len(items)
    to_llm = list(range(len(items)))
    if system1 and linker != "llm":
        scorer = await get_scorer(calibration.model)
        plain = await scorer.score(context, items, prompt=prompt)
        to_llm = []
        for i, probabilities in enumerate(plain):
            best = max(probabilities, key=probabilities.get)
            decisions[i] = Decision(best, probabilities[best], "system1", validated, probabilities)
            if linker == "system1+llm" and probabilities[best] < calibration.consult_below:
                to_llm.append(i)
    if not to_llm:
        return decisions  # type: ignore[return-value]

    answers = await ask_llm([items[i] for i in to_llm])
    choices = {i: answers.get(j, ("none", 0.0)) for j, i in enumerate(to_llm)}
    labels = {
        i: next(c.get("label") or c["id"] for c in items[i]["candidates"] if c["id"] == choice)
        for i, (choice, _) in choices.items()
        if choice != "none"
    }
    hinted: dict[int, dict[str, float]] = {}
    if system1 and labels:
        consulted = sorted(labels)
        scorer = await get_scorer(calibration.model)
        scores = await scorer.score(context, [items[i] for i in consulted], [labels[i] for i in consulted], prompt=prompt)
        hinted = dict(zip(consulted, scores))

    for i, (choice, stated) in choices.items():
        if linker == "llm":
            if choice == "none":
                decisions[i] = Decision("none", 0.0, "llm", False)
            elif i in hinted:
                decisions[i] = Decision(choice, hinted[i].get(choice, 0.0), "llm", False, hinted[i], labels[i])
            else:
                decisions[i] = Decision(choice, stated, "llm", False)
        elif i in hinted:  # system1+llm: Laya decides again, with the LLM's suggestion
            best = max(hinted[i], key=hinted[i].get)
            decisions[i] = Decision(best, hinted[i][best], "system1+llm", validated, hinted[i], labels[i])
        elif choice == "none":
            # Laya was unsure and the LLM found no candidate that fits: leave the item out.
            decisions[i] = Decision("none", 0.0, "system1+llm", False)
    return decisions  # type: ignore[return-value]
