from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class LLMUsage:
    """LLM calls made while tracking is active: which models answered, tokens and cost."""

    requested_model: str | None = None
    models_used: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    fallbacks: int = 0

    def record(
        self,
        *,
        model: str | None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        fallbacks: int = 0,
    ) -> None:
        self.calls += 1
        if model and model not in self.models_used:
            self.models_used.append(model)
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        if cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + float(cost_usd)
        self.fallbacks += fallbacks

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "models_used": list(self.models_used),
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8) if self.cost_usd is not None else None,
            "fallbacks": self.fallbacks,
        }


# Set per job, so one shared client can serve concurrent jobs. Tasks started inside a
# job copy the context and therefore add to the same LLMUsage.
_current_usage: ContextVar[LLMUsage | None] = ContextVar("moose_llm_usage", default=None)


@contextmanager
def track_llm_usage(requested_model: str | None = None) -> Iterator[LLMUsage]:
    usage = LLMUsage(requested_model=requested_model)
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)


def record_llm_call(**kwargs: Any) -> None:
    usage = _current_usage.get()
    if usage is not None:
        usage.record(**kwargs)
