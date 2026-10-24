"""Moose: privacy annotation of text and tables with the W3C Data Privacy Vocabulary (DPV).

Start with `Annotator` (see moose.annotator); `run_text_ner` and the other run_* functions
are the lower-level async entry points the API uses.
"""

from .annotator import Annotator
from .config import Settings, get_settings
from .llm import LLMClient, LLMUsage, OllamaClient, OpenAICompatibleClient, OpenRouterClient
from .ner import run_table_annotate, run_text_ner

__all__ = [
    "Annotator",
    "LLMClient",
    "LLMUsage",
    "OllamaClient",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "Settings",
    "get_settings",
    "run_table_annotate",
    "run_text_ner",
]
