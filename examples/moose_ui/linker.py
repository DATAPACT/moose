from __future__ import annotations

from typing import Any

import streamlit as st

LINKER_HELP = {
    "system1+llm": "System 1 + LLM: Laya decides and consults the LLM when unsure",
    "system1": "System 1 only: Laya decides, no extra LLM call",
    "llm": "LLM: the LLM decides; Laya gives the probability of its choice",
}


def linker_picker(schema_info: dict[str, Any], *, key: str, item: str = "item", task: str = "text") -> str | None:
    """The linker for a two-step schema, chosen among those the API server can run; None otherwise."""
    if schema_info.get("candidate_mode") != "two_step":
        st.caption("One-step schema: the LLM sees all of the schema's types and decides; Laya gives the probability.")
        return None
    st.caption(
        f"Two-step schema: the LLM first finds and describes each {item}, then a few candidate types are "
        f"retrieved and one is chosen for each {item}."
    )
    available = schema_info.get("linkers") or ["llm"]
    default = schema_info.get("linker") if schema_info.get("linker") in available else available[0]
    choice = st.radio(
        f"Who chooses each {item}'s type?",
        available,
        index=available.index(default),
        format_func=lambda name: LINKER_HELP.get(name, name),
        key=key,
    )
    if available == ["llm"]:
        st.caption(
            "The System 1 model (Laya) is not installed on this API server: only the LLM linker is available, "
            "and results carry no probability."
        )
    elif task not in schema_info.get("system1_validated_for", [task]):
        st.caption(
            f"Laya's probabilities for this schema's {task} annotations have not been checked against "
            "labelled data yet, so results show calibrated = false. dpv_privacy text is validated."
        )
    return choice


def ranking_text(ranking: Any) -> str | None:
    if not isinstance(ranking, list) or not ranking:
        return None
    return " · ".join(f"{r.get('type_id')} {float(r.get('probability') or 0):.2f}" for r in ranking if isinstance(r, dict))


def decision_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Columns for how a two-step type was decided (absent for other schemas)."""
    return {
        "calibrated": item.get("calibrated"),
        "decided_by": item.get("decided_by"),
        "llm_suggestion": item.get("llm_suggestion"),
        "category": item.get("category"),
        "ranking": ranking_text(item.get("ranking")),
    }


def decision_caption() -> None:
    st.caption(
        "confidence: the probability that the type is right, always from the System 1 model (Laya), never from "
        "the LLM, which tends to be over-confident. calibrated = true where that probability was checked against "
        "labelled data; empty when Laya is not available on the server. decided_by: system1 (Laya alone), "
        "system1+llm (Laya after the LLM's suggestion, shown in llm_suggestion) or llm (the LLM chose; Laya "
        "scored its choice). category: how the LLM described it. ranking: Laya's top candidates."
    )
