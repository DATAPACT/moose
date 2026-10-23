"""Choosing each item's type among its candidates, with a System 1 model deciding.

Moose's decisions are classifications: a mention's type, a column's type. A "System 1"
model, Laya, scores every candidate in one forward pass and gives probabilities; an LLM
adds what it knows. Three linkers:

  system1+llm  Laya decides. Items it is unsure about (top probability below the
               calibration's `consult_below`) go to the LLM, whose suggestion is added to
               Laya's input; Laya then decides again. The final probability is Laya's.
  system1      Laya alone; no LLM call for this step.
  llm          the LLM decides; Laya gives the probability of the LLM's choice.

The LLM never states a probability: LLMs are over-confident (on DPVCallBench, a stated
0.97 for 77% right). Every probability is Laya's, mapped through the schema's calibrator
for that linker where one was fitted on labelled data, and marked calibrated only where
that was checked. Without Laya (the plain library), decisions carry no probability.

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
CONFIDENCE_MODES = ("top1", "family")
SYSTEM1_LINKERS = ("system1+llm", "system1")
DEFAULT_LAYA_MODEL = "convaiinnovations/laya"
# Characters of each candidate's definition shown to Laya: definitions help it a lot,
# but its options share a small token budget.
LAYA_DEFINITION_CHARS = 140

# ask_llm(items) -> {index in items: type_id or "none"}, one LLM call.
AskLLM = Callable[[list[dict[str, Any]]], Awaitable[dict[int, str]]]


class LinkerUnavailable(RuntimeError):
    """The requested linker cannot run here (e.g. Laya is not installed)."""


def laya_available() -> bool:
    try:
        import laya  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Calibrator:
    """Maps Laya's raw probability to the accuracy observed on labelled data (isotonic
    regression, stored as increasing [raw, calibrated] points, linearly interpolated)."""

    points: tuple[tuple[float, float], ...]
    validated_for: tuple[str, ...] = ()
    fitted_on: str | None = None

    def __call__(self, raw: float) -> float:
        points = self.points
        if raw <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if raw <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
        return points[-1][1]


@dataclass(frozen=True)
class Calibration:
    """How a schema's System 1 decisions are made (see research/dpv_eval/fit_system1.py)."""

    # Hugging Face repo id (a private one needs HF_TOKEN) or a local checkpoint folder.
    model: str = DEFAULT_LAYA_MODEL
    # Environment variable that overrides `model` (e.g. a local folder); MOOSE_LAYA_MODEL by
    # default. Fine-tuned models use their own, so they do not replace the base model elsewhere.
    model_env: str = "MOOSE_LAYA_MODEL"
    # system1+llm: ask the LLM about items whose top probability is below this.
    consult_below: float = 0.6
    # "top1": the chosen type's probability. "family": it plus its direct parents' and
    # children's among the candidates, for models trained to spread probability over a type
    # and its finer children (see research/dpv_eval, fine-tuned Laya).
    confidence: str = "top1"
    # Tasks on which Laya's raw probabilities were checked against labelled data (used when
    # the linker has no calibrator; never for the llm linker).
    validated_for: tuple[str, ...] = ("text",)
    # Per linker: a calibrator fitted on labelled data (see research/dpv_eval/fit_calibrators.py).
    calibrators: tuple[tuple[str, Calibrator], ...] = ()
    fitted_on: str | None = None

    @classmethod
    def load(cls, path: Path | None) -> "Calibration | None":
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = ("model", "model_env", "consult_below", "confidence", "validated_for", "fitted_on")
        known: dict[str, Any] = {k: data[k] for k in fields if k in data}
        if "validated_for" in known:
            known["validated_for"] = tuple(known["validated_for"])
        known["calibrators"] = tuple(
            (linker, Calibrator(
                points=tuple((float(x), float(y)) for x, y in entry["points"]),
                validated_for=tuple(entry.get("validated_for") or ()),
                fitted_on=entry.get("fitted_on"),
            ))
            for linker, entry in (data.get("calibrators") or {}).items()
        )
        if known.get("confidence", "top1") not in CONFIDENCE_MODES:
            raise ValueError(f"{path}: confidence must be one of {', '.join(CONFIDENCE_MODES)}.")
        override = os.getenv(known.get("model_env", "MOOSE_LAYA_MODEL"))
        if override:
            known["model"] = override
        return cls(**known)

    def validated_tasks(self, linker: str) -> tuple[str, ...]:
        """Tasks on which this linker's probabilities were checked against labelled data: its
        calibrator's, or the file's validated_for for Laya's own raw decisions only."""
        calibrator = dict(self.calibrators).get(linker)
        if calibrator is not None:
            return calibrator.validated_for
        return self.validated_for if linker in SYSTEM1_LINKERS else ()

    def probability(self, linker: str, raw: float, task: str) -> tuple[float, bool]:
        """The probability to report for a decision, and whether it counts as calibrated."""
        calibrator = dict(self.calibrators).get(linker)
        probability = calibrator(raw) if calibrator is not None else raw
        return probability, task in self.validated_tasks(linker)

    def confidence_of(self, probabilities: dict[str, float], chosen: str, candidates: list[dict[str, Any]]) -> float:
        if self.confidence != "family":
            return probabilities[chosen]
        parents = {c["id"]: set(c.get("parents") or []) for c in candidates}
        related = {c for c in parents if c != chosen and (c in parents.get(chosen, set()) or chosen in parents[c])}
        return min(1.0, probabilities[chosen] + sum(probabilities.get(c, 0.0) for c in related))


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
    # The probability that type_id is right (Laya's); None when Laya is not available.
    confidence: float | None
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
    selects which validations apply. Every probability comes from Laya."""
    if not items:
        return []
    system1 = system1_ready(calibration)
    if linker in SYSTEM1_LINKERS and not system1:
        failure = _recent_failure(calibration.model) if calibration else None
        raise LinkerUnavailable(failure or "This linker needs Laya and the schema's calibration; use the llm linker.")

    def decision(chosen: str, probabilities: dict[str, float], decided_by: str, i: int, suggestion: str | None = None) -> Decision:
        raw = calibration.confidence_of(probabilities, chosen, items[i]["candidates"])
        probability, calibrated = calibration.probability(linker, raw, task)
        return Decision(chosen, probability, decided_by, calibrated, probabilities, suggestion)

    decisions: list[Decision | None] = [None] * len(items)
    to_llm = list(range(len(items)))
    if system1 and linker != "llm":
        scorer = await get_scorer(calibration.model)
        plain = await scorer.score(context, items, prompt=prompt)
        to_llm = []
        for i, probabilities in enumerate(plain):
            best = max(probabilities, key=probabilities.get)
            decisions[i] = decision(best, probabilities, "system1", i)
            if linker == "system1+llm" and calibration.confidence_of(probabilities, best, items[i]["candidates"]) < calibration.consult_below:
                to_llm.append(i)
    if not to_llm:
        return decisions  # type: ignore[return-value]

    answers = await ask_llm([items[i] for i in to_llm])
    choices = {i: answers.get(j, "none") for j, i in enumerate(to_llm)}
    labels = {
        i: next(c.get("label") or c["id"] for c in items[i]["candidates"] if c["id"] == choice)
        for i, choice in choices.items()
        if choice != "none"
    }

    # Laya scores the candidates again, given the LLM's choice: it decides again
    # (system1+llm), or gives the probability of the LLM's choice (llm).
    hinted: dict[int, dict[str, float]] = {}
    if system1 and labels:
        consulted = sorted(labels)
        try:
            scorer = await get_scorer(calibration.model)
        except LinkerUnavailable:
            if linker != "llm":
                raise
            scorer = None  # the LLM's choice stands, without a probability
        if scorer is not None:
            scores = await scorer.score(context, [items[i] for i in consulted], [labels[i] for i in consulted], prompt=prompt)
            hinted = dict(zip(consulted, scores))

    for i, choice in choices.items():
        if choice == "none":
            # The LLM found no candidate that fits (for system1+llm: Laya was unsure too).
            decisions[i] = Decision("none", None, linker, False)
        elif linker == "llm":
            decisions[i] = decision(choice, hinted[i], "llm", i, labels[i]) if i in hinted else Decision(choice, None, "llm", False)
        else:
            best = max(hinted[i], key=hinted[i].get)
            decisions[i] = decision(best, hinted[i], "system1+llm", i, labels[i])
    return decisions  # type: ignore[return-value]
