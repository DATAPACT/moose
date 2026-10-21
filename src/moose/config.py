from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # "gateway" (OpenAI-compatible endpoint such as LiteLLM; used by the API service),
    # or "ollama" / "openrouter" for direct library use.
    MOOSE_LLM_PROVIDER: str = "gateway"
    # Model name or gateway alias used when a request does not name one.
    MOOSE_MODEL: str | None = None
    # Admin API key (full access). Partners use their own LiteLLM virtual keys instead.
    MOOSE_API_KEY: str | None = None
    MOOSE_MAX_RETRIES: int = 2
    MOOSE_WORKER_COUNT: int = 4
    MOOSE_QUEUE_MAXSIZE: int = 1000

    # LLM gateway (OpenAI-compatible). The key is the gateway's virtual key, not a provider key.
    MOOSE_LLM_BASE_URL: str = "http://localhost:4000/v1"
    MOOSE_LLM_API_KEY: str | None = None

    # Direct providers (library use).
    MOOSE_OLLAMA_HOST: str = "http://localhost:11434"
    MOOSE_OLLAMA_TOKEN: str | None = None
    MOOSE_OPENROUTER_API_KEY: str | None = None
    MOOSE_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    MOOSE_MONGO_URL: str | None = None
    MOOSE_MONGO_DB: str = "moose"
    MOOSE_MONGO_TIMEOUT_SECS: float = 2.0
    MOOSE_MAX_TASKS_PER_PROMPT: int = 10
    MOOSE_MAX_CHARS_PER_PROMPT: int = 12000
    MOOSE_TIMEOUT_SECS: float = 3600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()