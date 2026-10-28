"""Choosing each item's type among its candidates, with a System 1 model deciding.

Moose's decisions are classifications: a mention's type, a column's type. A "System 1"
model, Laya, scores every candidate in one forward pass and gives probabilities; an LLM
adds what it knows. Three linkers:

  system1+llm  Laya decides. Items it is unsure about (top probability below the
               calibration's `consult_below`) go to the LLM, whose suggestion is added to
               Laya's input; Laya then decides again. The final probability is Laya's.
  system1      Laya alone; no LLM call for this step.
  llm          the LLM decides; Laya, asked a plain yes/no question about that type (not
               told it came from an LLM), gives the probability that it is right.

The LLM never states a probability: LLMs are over-confident (on DPVCallBench, a stated
0.97 for 77% right). Every probability is Laya's: for its own decisions, the chosen type's
probability in Laya's ranking, unchanged; for the LLM's, Laya's P(yes) that the item is
that type. It is marked calibrated only where the schema's calibration file says
that linker's probabilities were checked against labelled data. Without Laya (the plain
library), decisions carry no probability.

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
class Calibration:
    """How a schema's System 1 decisions are made (see research/dpv_eval/fit_system1.py)."""

    # Hugging Face repo id (a private one needs HF_TOKEN) or a local checkpoint folder.
    model: str = DEFAULT_LAYA_MODEL
    # Environment variable that overrides `model` (e.g. a local folder); MOOSE_LAYA_MODEL by
    # default. Fine-tuned models use their own, so they do not replace the base model elsewhere.
    model_env: str = "MOOSE_LAYA_MODEL"
    # system1+llm: ask the LLM about items whose top probability is below this.
    consult_below: float = 0.6
    # Tasks on which Laya's probabilities were checked against labelled data (used for
    # Laya's own linkers when the file has no entry for the linker; never for llm).
    validated_for: tuple[str, ...] = ("text",)
    # Per linker: the tasks on which its probabilities were checked (see
    # research/dpv_eval/check_calibration.py).
    linkers: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fitted_on: str | None = None

    @classmethod
    def load(cls, path: Path | None) -> "Calibration | None":
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = ("model", "model_env", "consult_below", "validated_for", "fitted_on")
        known: dict[str, Any] = {k: data[k] for k in fields if k in data}
        if "validated_for" in known:
            known["validated_for"] = tuple(known["validated_for"])
        known["linkers"] = tuple(
            (linker, tuple(entry.get("validated_for") or ())) for linker, entry in (data.get("linkers") or {}).items()
        )
        override = os.getenv(known.get("model_env", "MOOSE_LAYA_MODEL"))
        if override:
            known["model"] = override
        return cls(**known)

    def validated_tasks(self, linker: str) -> tuple[str, ...]:
        """Tasks on which this linker's probabilities were checked against labelled data: its
        own entry, or the file's validated_for for Laya's own linkers only."""
        checked = dict(self.linkers)
        if linker in checked:
            return checked[linker]
        return self.validated_for if linker in SYSTEM1_LINKERS else ()


async def llm_choice_verdicts(
    calibration: Calibration | None, context: str, items: list[dict[str, Any]], records: list[dict[str, Any]], prompt: Prompt
) -> list[float | None]:
    """For types an LLM chose: Laya's yes/no probability that each is right (see
    laya_verify_inputs). None for every item when Laya is not available."""
    if not items or not system1_ready(calibration):
        return [None] * len(items)
    try:
        scorer = await get_scorer(calibration.model)
    except LinkerUnavailable:
        return [None] * len(items)
    return await scorer.verify(context, items, records, prompt)


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
    # The yes/no question for one type, chosen by the LLM.
    verify_question: str = "Is the mention {label}?"


TEXT_PROMPT = Prompt()
TABLE_PROMPT = Prompt(
    context_label="Table columns",
    item_label="Column and sample values",
    instructions="Which category of personal data or privacy role do the column's values hold?",
    verify_question="Do the column's values hold {label}?",
)
CELL_PROMPT = Prompt(context_label="Table cell")
CPA_PROMPT = Prompt(
    context_label="Table",
    item_label="Subject and target columns",
    instructions="What is the relationship between the subject column and the target column?",
    verify_question="Is the relationship between the subject column and the target column {label}?",
)


# Laya reads 512 tokens, the question and options included, and the context comes first: on a
# long text the mention and its line are cut off, and every mention of the text gets the same
# scores. A mention with a position in a longer text is shown with this much text around it.
# Laya was trained and calibrated on short texts (DPVCallBench's calls, at most 121 characters):
# with 800 characters it typed most names in court cases (TAB) dpv:Adult or Identifier, with 400
# it typed 66 of 87 Name and 3 Adult.
LAYA_CONTEXT_CHARS = 400


def _laya_context(context: str, item: dict[str, Any]) -> str:
    start, end = item.get("start"), item.get("end")
    if len(context) <= LAYA_CONTEXT_CHARS or not isinstance(start, int) or not isinstance(end, int):
        return context
    side = max(0, (LAYA_CONTEXT_CHARS - (end - start)) // 2)
    lo, hi = max(0, start - side), min(len(context), end + side)
    return ("…" if lo else "") + context[lo:hi] + ("…" if hi < len(context) else "")


def laya_inputs(
    context: str, item: dict[str, Any], suggestion: str | None = None, prompt: Prompt = TEXT_PROMPT
) -> tuple[str, dict[str, Any]]:
    """Laya's state and question for one item. Candidates keep their retrieval order,
    which Laya uses as a cue (shuffling them lowers its accuracy). `suggestion` is the
    label the LLM chose, given to Laya as evidence."""
    state = f'{prompt.context_label}: "{_laya_context(context, item)}"\n{prompt.item_label}: "{item["text"]}"'
    if item.get("category"):
        state += f' (described as: {item["category"]})'
    if suggestion:
        state += f"\nAn LLM suggested: {suggestion}"
    criteria = {}
    for i, record in enumerate(item["candidates"]):
        definition = _short(record.get("definition"))
        criteria[f"option_{i}"] = f"{record.get('label') or record['id']}" + (f": {definition}" if definition else "")
    return state, {"category": {"type": "choice", "instructions": prompt.instructions, "criteria": criteria}}


def laya_verify_inputs(context: str, item: dict[str, Any], record: dict[str, Any], prompt: Prompt = TEXT_PROMPT) -> tuple[str, dict[str, Any]]:
    """Laya's state and yes/no question for one type: the item alone, with no other
    candidates and nothing about who proposed the type, so Laya judges it on its own."""
    state = f'{prompt.context_label}: "{_laya_context(context, item)}"\n{prompt.item_label}: "{item["text"]}"'
    label = record.get("label") or record["id"]
    definition = _short(record.get("definition"))
    criteria = {"yes": f"Yes, it is {label}" + (f": {definition}" if definition else ""), "no": f"No, it is not {label}"}
    return state, {"verdict": {"type": "choice", "instructions": prompt.verify_question.format(label=label), "criteria": criteria}}


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

    def _verify_sync(self, context: str, items: list[dict[str, Any]], records: list[dict[str, Any]], prompt: Prompt) -> list[float]:
        out = []
        with self._lock:
            for item, record in zip(items, records):
                state, questions = laya_verify_inputs(context, item, record, prompt)
                out.append(float(self._agent.predict(state, questions)["answers"]["verdict"]["probabilities"]["yes"]))
        return out

    async def verify(
        self, context: str, items: list[dict[str, Any]], records: list[dict[str, Any]], prompt: Prompt = TEXT_PROMPT
    ) -> list[float]:
        """For each item, Laya's probability that it is the given type (a candidate record)."""
        return await asyncio.to_thread(self._verify_sync, context, items, records, prompt)

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


def ranked(probabilities: dict[str, float], chosen: str | None = None, top: int = 3) -> list[dict[str, Any]]:
    """Laya's top types, plus the chosen one if it is not among them (the LLM's choice can
    rank lower), so the reported confidence is always in the ranking."""
    best = sorted(probabilities.items(), key=lambda item: -item[1])[:top]
    if chosen in probabilities and chosen not in dict(best):
        best.append((chosen, probabilities[chosen]))
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

    def decision(chosen: str, probabilities: dict[str, float], decided_by: str, suggestion: str | None = None,
                 confidence: float | None = None) -> Decision:
        calibrated = task in calibration.validated_tasks(linker)
        confidence = probabilities[chosen] if confidence is None else confidence
        return Decision(chosen, confidence, decided_by, calibrated, probabilities, suggestion)

    decisions: list[Decision | None] = [None] * len(items)
    to_llm = list(range(len(items)))
    if system1 and linker != "llm":
        scorer = await get_scorer(calibration.model)
        plain = await scorer.score(context, items, prompt=prompt)
        to_llm = []
        for i, probabilities in enumerate(plain):
            best = max(probabilities, key=probabilities.get)
            decisions[i] = decision(best, probabilities, "system1")
            if linker == "system1+llm" and probabilities[best] < calibration.consult_below:
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

    # system1+llm: Laya scores the candidates again, given the LLM's choice, and decides again.
    # llm: Laya ranks the candidates on its own and answers yes/no on the LLM's choice.
    hinted: dict[int, dict[str, float]] = {}
    verdicts: dict[int, float] = {}
    if system1 and labels:
        consulted = sorted(labels)
        try:
            scorer = await get_scorer(calibration.model)
        except LinkerUnavailable:
            if linker != "llm":
                raise
            scorer = None  # the LLM's choice stands, without a probability
        if scorer is not None and linker == "llm":
            chosen = [next(c for c in items[i]["candidates"] if c["id"] == choices[i]) for i in consulted]
            scores = await scorer.score(context, [items[i] for i in consulted], prompt=prompt)
            yes = await scorer.verify(context, [items[i] for i in consulted], chosen, prompt=prompt)
            hinted, verdicts = dict(zip(consulted, scores)), dict(zip(consulted, yes))
        elif scorer is not None:
            scores = await scorer.score(context, [items[i] for i in consulted], [labels[i] for i in consulted], prompt=prompt)
            hinted = dict(zip(consulted, scores))

    for i, choice in choices.items():
        if choice == "none":
            # The LLM found no candidate that fits (for system1+llm: Laya was unsure too).
            decisions[i] = Decision("none", None, linker, False)
        elif linker == "llm":
            decisions[i] = (decision(choice, hinted[i], "llm", labels[i], verdicts[i]) if i in verdicts
                            else Decision(choice, None, "llm", False))
        else:
            best = max(hinted[i], key=hinted[i].get)
            decisions[i] = decision(best, hinted[i], "system1+llm", labels[i])
    return decisions  # type: ignore[return-value]
