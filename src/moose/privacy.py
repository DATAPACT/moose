from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from moose.config import Settings, get_settings
from moose.ner import run_table_annotate, run_tabular_ner, run_text_ner
from moose.pipelines import resolve_privacy_defaults
from moose.schema import get_schema_config
from moose.validate import extract_json


DATA_DIR = Path(__file__).resolve().parent / "data"
POLICY_DIR = DATA_DIR / "policies"
REPORTS_DIR = DATA_DIR / "reports"
MACHINE_REPORT_SCHEMA_PATH = REPORTS_DIR / "privacy_machine_report_v1.schema.json"


def list_policy_packs() -> list[str]:
    """
    List available policy pack JSON files in moose/data/policies/*.json
    (excluding subdirectories).
    Returns names without .json extension.
    """
    if not POLICY_DIR.exists():
        return []
    packs = [p.stem for p in POLICY_DIR.glob("*.json") if p.is_file()]
    return sorted(set(packs))


def get_machine_report_json_schema() -> dict[str, Any]:
    if MACHINE_REPORT_SCHEMA_PATH.exists():
        try:
            payload = json.loads(MACHINE_REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            pass
    schema = MachinePrivacyReport.model_json_schema()
    schema["$id"] = "moose.privacy.machine_report.v1"
    schema["x-schema-version"] = "1.0.0"
    return schema


# -------------------------
# Policy pack models/schema
# -------------------------
Severity = Literal["low", "medium", "high"]
AssessmentLevel = Literal["compliance_risk", "possible_violation"]


class PolicyAction(BaseModel):
    action_id: str
    label: str
    description: str | None = None


class RuleMinCount(BaseModel):
    category: str
    count: int = Field(ge=1)


class RuleCondition(BaseModel):
    # Category-based conditions
    any_of: list[str] = Field(default_factory=list)
    all_of: list[str] = Field(default_factory=list)
    min_count: RuleMinCount | None = None

    # Signal-based conditions (deterministic lexical signals)
    any_signals: list[str] = Field(default_factory=list)
    all_signals: list[str] = Field(default_factory=list)

    # OR-list between categories and signals
    # Each item must be either "cat:<CATEGORY_ID>" or "sig:<SIGNAL_ID>"
    any_of_either: list[str] = Field(default_factory=list)


class PolicyRule(BaseModel):
    rule_id: str
    scope: Literal["text_task", "table_task", "any"] = "any"
    issue: str
    severity: Severity
    default_actions: list[str] = Field(default_factory=list)
    when: RuleCondition
    needs_llm: bool = False
    description: str | None = None


class LegalRefsConfig(BaseModel):
    """
    Configuration for legal references that map rules to regulatory concepts.
    """
    source: str  # e.g., "gdprtext", "hipaa"
    by_rule_id: dict[str, list[str]] = Field(default_factory=dict)


class LLMAssessorConfig(BaseModel):
    intro: str | None = None
    max_findings: int = 20
    max_actions_per_finding: int = 6
    max_rationale_chars: int = 300


class ConfidenceConfig(BaseModel):
    ignore_below: float = 0.05
    default_min: float = 0.40
    by_category: dict[str, float] = Field(default_factory=dict)


class PolicyPack(BaseModel):
    name: str
    label: str | None = None
    version: str | None = None

    actions: list[PolicyAction]

    type_categories: dict[str, list[str]] = Field(default_factory=dict)
    type_prefix_categories: dict[str, list[str]] = Field(default_factory=dict)

    rules: list[PolicyRule] = Field(default_factory=list)
    llm_assessor: LLMAssessorConfig = Field(default_factory=LLMAssessorConfig)

    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    signals: dict[str, list[str]] = Field(default_factory=dict)

    # Legal references mapping
    legal_refs: LegalRefsConfig | None = None

    max_evidence_per_finding: int = 5
    max_scan_examples_per_column: int = 3


@lru_cache
def load_policy_pack(name: str) -> PolicyPack:
    filename = name if name.endswith(".json") else f"{name}.json"
    path = (POLICY_DIR / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Policy pack not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    pack = PolicyPack.model_validate(data)

    action_ids = [a.action_id for a in pack.actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Policy pack actions must have unique action_id values.")
    action_set = set(action_ids)

    for rule in pack.rules:
        unknown = [a for a in rule.default_actions if a not in action_set]
        if unknown:
            raise ValueError(f"Policy pack rule '{rule.rule_id}' has unknown default_actions: {unknown}")

        for item in rule.when.any_of_either:
            if not (item.startswith("cat:") or item.startswith("sig:")):
                raise ValueError(
                    f"Policy pack rule '{rule.rule_id}' has invalid any_of_either item: {item!r}. "
                    "Must start with 'cat:' or 'sig:'."
                )

    if pack.confidence.ignore_below < 0 or pack.confidence.ignore_below > 1:
        raise ValueError("confidence.ignore_below must be in [0,1].")
    if pack.confidence.default_min < 0 or pack.confidence.default_min > 1:
        raise ValueError("confidence.default_min must be in [0,1].")
    for k, v in pack.confidence.by_category.items():
        if v < 0 or v > 1:
            raise ValueError(f"confidence.by_category[{k}] must be in [0,1].")

    for signal_id, phrases in pack.signals.items():
        if not isinstance(signal_id, str) or not signal_id.strip():
            raise ValueError("Policy pack signals must have non-empty signal IDs.")
        if not isinstance(phrases, list):
            raise ValueError(f"Policy pack signals[{signal_id}] must be a list.")
        for p in phrases:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Policy pack signals[{signal_id}] contains empty/non-string phrase: {p!r}")

    return pack


def _resolve_legal_refs_detail(source: str, ref_ids: list[str]) -> list[dict[str, Any]]:
    if not ref_ids:
        return []
    try:
        from moose.legal import resolve_legal_refs_detail
        return resolve_legal_refs_detail(source, ref_ids)
    except Exception:  # noqa: BLE001
        return []


# -------------------------
# Evidence + finding models
# -------------------------
class EvidenceEntity(BaseModel):
    kind: Literal["entity"] = "entity"
    task_id: str
    start: int
    end: int
    text: str
    type_id: str
    confidence: float | None  # None: no System 1 probability (Laya not available)
    categories: list[str] = Field(default_factory=list)

    confidence_threshold: float | None = None
    low_confidence: bool = False


class EvidenceColumn(BaseModel):
    kind: Literal["column"] = "column"
    task_id: str
    table_id: str
    column: str
    type_id: str
    confidence: float | None
    categories: list[str] = Field(default_factory=list)

    confidence_threshold: float | None = None
    low_confidence: bool = False

    sample_nonnull_count: int | None = None
    sample_unique_count: int | None = None
    sample_unique_ratio: float | None = None
    sample_avg_len: float | None = None
    sample_examples: list[str] = Field(default_factory=list)


class EvidenceScanSummary(BaseModel):
    kind: Literal["scan_summary"] = "scan_summary"
    task_id: str
    table_id: str
    column: str

    type_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    examples: list[dict[str, Any]] = Field(default_factory=list)


Evidence = EvidenceEntity | EvidenceColumn | EvidenceScanSummary


class MitigationStep(BaseModel):
    action_id: str
    label: str
    description: str | None = None
    priority: int
    targets: list[str] = Field(default_factory=list)


class FindingCandidate(BaseModel):
    rule_id: str
    issue: str
    severity: Severity
    default_actions: list[str]
    evidence: list[Evidence] = Field(default_factory=list)
    needs_llm: bool = False
    reason_needs_llm: str | None = None


class FindingOut(BaseModel):
    rule_id: str
    issue: str
    status: Literal["confirmed", "rejected", "uncertain"]
    severity: Severity
    recommended_actions: list[str]
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)
    assessment: AssessmentLevel = "compliance_risk"
    violation_reasons: list[str] = Field(default_factory=list)
    mitigation_plan: list[MitigationStep] = Field(default_factory=list)

    legal_refs: list[str] = Field(default_factory=list)
    legal_refs_detail: list[dict[str, Any]] = Field(default_factory=list)


class AssessedFinding(BaseModel):
    rule_id: str
    status: Literal["confirmed", "rejected", "uncertain"]
    severity: Severity
    recommended_actions: list[str] = Field(default_factory=list)
    rationale: str


class AssessorResponse(BaseModel):
    findings: list[AssessedFinding] = Field(default_factory=list)


class MachineReportSummary(BaseModel):
    task_count: int
    finding_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    assessment_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)


class MachineReportFinding(BaseModel):
    rule_id: str
    issue: str
    severity: str
    status: str
    assessment: str
    rationale: str
    recommended_actions: list[str] = Field(default_factory=list)
    mitigation_plan: list[dict[str, Any]] = Field(default_factory=list)
    violation_reasons: list[str] = Field(default_factory=list)
    legal_refs: list[str] = Field(default_factory=list)
    legal_refs_detail: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class MachineReportTask(BaseModel):
    task_id: str
    kind: str
    table_id: str | None = None
    finding_count: int
    findings: list[MachineReportFinding] = Field(default_factory=list)


class MachinePrivacyReport(BaseModel):
    report_type: Literal["privacy_analysis"] = "privacy_analysis"
    schema_id: Literal["moose.privacy.machine_report.v1"] = "moose.privacy.machine_report.v1"
    schema_version: Literal["1.0.0"] = "1.0.0"
    generated_at: str
    profile: str | None = None
    policy_pack: str
    summary: MachineReportSummary
    tasks: list[MachineReportTask] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    escalations: list[dict[str, Any]] = Field(default_factory=list)


class HumanReadableReport(BaseModel):
    format: Literal["markdown"] = "markdown"
    content: str


class MachineReadableReport(BaseModel):
    schema_id: Literal["moose.privacy.machine_report.v1"] = "moose.privacy.machine_report.v1"
    schema_version: Literal["1.0.0"] = "1.0.0"
    content: dict[str, Any]


class PrivacyReports(BaseModel):
    human_readable: HumanReadableReport
    machine_readable: MachineReadableReport


# -------------------------
# Categorization + confidence
# -------------------------
def categories_for_type(pack: PolicyPack, type_id: str) -> list[str]:
    cats: list[str] = []
    for prefix, prefix_cats in pack.type_prefix_categories.items():
        if type_id.startswith(prefix):
            cats.extend(prefix_cats)
    cats.extend(pack.type_categories.get(type_id, []))

    out: list[str] = []
    seen: set[str] = set()
    for c in cats:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def confidence_threshold_for_categories(pack: PolicyPack, categories: list[str]) -> float:
    threshold = pack.confidence.default_min
    for c in categories:
        if c in pack.confidence.by_category:
            threshold = max(threshold, pack.confidence.by_category[c])
    return threshold


def _probability(value: Any) -> float | None:
    """An item's probability; None when it has none (annotated without the System 1 model)."""
    return None if value is None else float(value)


def is_ignored(pack: PolicyPack, confidence: float | None) -> bool:
    """Dropped as noise. Items without a probability are kept, as low-confidence evidence."""
    return confidence is not None and confidence < pack.confidence.ignore_below


def is_low_confidence(pack: PolicyPack, confidence: float | None, categories: list[str]) -> tuple[bool, float]:
    if confidence is None or confidence < pack.confidence.ignore_below:
        return True, confidence_threshold_for_categories(pack, categories)
    threshold = confidence_threshold_for_categories(pack, categories)
    return confidence < threshold, threshold


# -------------------------
# Facts + signals + profiling
# -------------------------
@dataclass
class Facts:
    kind: Literal["text", "table"]
    task_id: str
    text: str | None = None
    table_id: str | None = None

    evidence: list[Evidence] | None = None
    signals_present: set[str] | None = None

    categories_present_any: set[str] | None = None
    categories_present_confident: set[str] | None = None

    category_counts_any: dict[str, int] | None = None
    category_counts_confident: dict[str, int] | None = None


def _profile_column(sampled_rows: list[dict[str, Any]], column: str, max_examples: int = 3) -> dict[str, Any]:
    values: list[str] = []
    lengths: list[int] = []

    for row in sampled_rows:
        v = row.get(column)
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v:
            continue
        values.append(v)
        lengths.append(len(v))

    nonnull = len(values)
    unique_count = len(set(values))
    unique_ratio = (unique_count / nonnull) if nonnull else 0.0
    avg_len = (sum(lengths) / nonnull) if nonnull else 0.0

    examples: list[str] = []
    for v in values:
        if v not in examples:
            examples.append(v)
        if len(examples) >= max_examples:
            break

    return {
        "sample_nonnull_count": nonnull,
        "sample_unique_count": unique_count,
        "sample_unique_ratio": unique_ratio,
        "sample_avg_len": avg_len,
        "sample_examples": examples,
    }


def _extract_signals(pack: PolicyPack, text: str) -> set[str]:
    if not pack.signals:
        return set()

    t = (text or "").lower()
    out: set[str] = set()

    for signal_id, phrases in pack.signals.items():
        for phrase in phrases:
            p = (phrase or "").strip().lower()
            if not p:
                continue
            if " " in p:
                if p in t:
                    out.add(signal_id)
                    break
            else:
                if re.search(rf"\b{re.escape(p)}\b", t):
                    out.add(signal_id)
                    break

    return out


def build_text_facts(pack: PolicyPack, task_id: str, text: str, entities: list[dict[str, Any]]) -> Facts:
    evidence: list[Evidence] = []
    signals_present = _extract_signals(pack, text)

    category_counts_any: dict[str, int] = {}
    category_counts_conf: dict[str, int] = {}
    cats_any: set[str] = set()
    cats_conf: set[str] = set()

    for e in entities:
        type_id = e["type_id"]
        conf = _probability(e.get("confidence"))
        cats = categories_for_type(pack, type_id)

        if is_ignored(pack, conf):
            continue

        low, threshold = is_low_confidence(pack, conf, cats)

        for c in cats:
            cats_any.add(c)
            category_counts_any[c] = category_counts_any.get(c, 0) + 1
            if not low:
                cats_conf.add(c)
                category_counts_conf[c] = category_counts_conf.get(c, 0) + 1

        evidence.append(
            EvidenceEntity(
                task_id=task_id,
                start=int(e["start"]),
                end=int(e["end"]),
                text=str(e["text"]),
                type_id=type_id,
                confidence=conf,
                categories=cats,
                confidence_threshold=threshold,
                low_confidence=low,
            )
        )

    return Facts(
        kind="text",
        task_id=task_id,
        text=text,
        evidence=evidence,
        signals_present=signals_present,
        categories_present_any=cats_any,
        categories_present_confident=cats_conf,
        category_counts_any=category_counts_any,
        category_counts_confident=category_counts_conf,
    )


def build_table_facts(
    pack: PolicyPack,
    task_id: str,
    table_id: str,
    sampled_rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    scan_result: dict[str, Any] | None = None,
) -> Facts:
    evidence: list[Evidence] = []
    signals_present: set[str] = set()

    cats_any: set[str] = set()
    cats_conf: set[str] = set()

    category_counts_any: dict[str, int] = {}
    category_counts_conf: dict[str, int] = {}

    for col in columns:
        type_id = col["type_id"]
        conf = _probability(col.get("confidence"))
        cats = categories_for_type(pack, type_id)

        if is_ignored(pack, conf):
            continue

        low, threshold = is_low_confidence(pack, conf, cats)

        for c in set(cats):
            cats_any.add(c)
            category_counts_any[c] = category_counts_any.get(c, 0) + 1
            if not low:
                cats_conf.add(c)
                category_counts_conf[c] = category_counts_conf.get(c, 0) + 1

        profile = _profile_column(sampled_rows, str(col["column"]), max_examples=3)

        evidence.append(
            EvidenceColumn(
                task_id=task_id,
                table_id=table_id,
                column=str(col["column"]),
                type_id=type_id,
                confidence=conf,
                categories=cats,
                confidence_threshold=threshold,
                low_confidence=low,
                **profile,
            )
        )

    if scan_result:
        per_col_type_counts: dict[str, dict[str, int]] = {}
        per_col_cat_counts: dict[str, dict[str, int]] = {}
        per_col_examples: dict[str, list[dict[str, Any]]] = {}

        rows = scan_result.get("rows", [])
        for row in rows:
            for cell in row.get("cells", []):
                col_name = cell.get("column")
                if not isinstance(col_name, str):
                    continue

                entities = cell.get("entities", [])
                for ent in entities:
                    type_id = ent.get("type_id")
                    if not isinstance(type_id, str):
                        continue

                    conf = _probability(ent.get("confidence"))
                    if is_ignored(pack, conf):
                        continue

                    cats = categories_for_type(pack, type_id)
                    low, _threshold = is_low_confidence(pack, conf, cats)

                    for c in cats:
                        cats_any.add(c)
                        category_counts_any[c] = category_counts_any.get(c, 0) + 1
                        if not low:
                            cats_conf.add(c)
                            category_counts_conf[c] = category_counts_conf.get(c, 0) + 1

                    per_col_type_counts.setdefault(col_name, {})
                    per_col_type_counts[col_name][type_id] = per_col_type_counts[col_name].get(type_id, 0) + 1

                    per_col_cat_counts.setdefault(col_name, {})
                    for c in cats:
                        per_col_cat_counts[col_name][c] = per_col_cat_counts[col_name].get(c, 0) + 1

                    ex_list = per_col_examples.setdefault(col_name, [])
                    if len(ex_list) < pack.max_scan_examples_per_column:
                        ex_list.append({"text": ent.get("text"), "type_id": type_id, "confidence": conf})

        for col_name, type_counts in per_col_type_counts.items():
            evidence.append(
                EvidenceScanSummary(
                    task_id=task_id,
                    table_id=table_id,
                    column=col_name,
                    type_counts=type_counts,
                    category_counts=per_col_cat_counts.get(col_name, {}),
                    examples=per_col_examples.get(col_name, []),
                )
            )

    return Facts(
        kind="table",
        task_id=task_id,
        table_id=table_id,
        evidence=evidence,
        signals_present=signals_present,
        categories_present_any=cats_any,
        categories_present_confident=cats_conf,
        category_counts_any=category_counts_any,
        category_counts_confident=category_counts_conf,
    )


# -------------------------
# Deterministic rule engine
# -------------------------
def _rule_applies(rule: PolicyRule, facts: Facts) -> bool:
    if rule.scope == "any":
        return True
    if rule.scope == "text_task" and facts.kind == "text":
        return True
    if rule.scope == "table_task" and facts.kind == "table":
        return True
    return False


def _condition_matches(cond: RuleCondition, facts: Facts, mode: Literal["any", "confident"]) -> bool:
    if mode == "any":
        present = facts.categories_present_any or set()
        counts = facts.category_counts_any or {}
    else:
        present = facts.categories_present_confident or set()
        counts = facts.category_counts_confident or {}

    signals = facts.signals_present or set()

    if cond.any_of and not (set(cond.any_of) & present):
        return False
    if cond.all_of and not set(cond.all_of).issubset(present):
        return False
    if cond.min_count is not None:
        if counts.get(cond.min_count.category, 0) < cond.min_count.count:
            return False

    if cond.any_signals and not (set(cond.any_signals) & signals):
        return False
    if cond.all_signals and not set(cond.all_signals).issubset(signals):
        return False

    if cond.any_of_either:
        ok = False
        for item in cond.any_of_either:
            if item.startswith("cat:"):
                cat = item[len("cat:") :]
                if cat in present:
                    ok = True
                    break
            elif item.startswith("sig:"):
                sig = item[len("sig:") :]
                if sig in signals:
                    ok = True
                    break
        if not ok:
            return False

    return True


def _relevant_categories_for_rule(rule: PolicyRule) -> set[str]:
    relevant: set[str] = set()
    relevant.update(rule.when.any_of)
    relevant.update(rule.when.all_of)
    if rule.when.min_count:
        relevant.add(rule.when.min_count.category)
    for item in rule.when.any_of_either:
        if item.startswith("cat:"):
            relevant.add(item[len("cat:") :])
    return relevant


def _select_evidence_for_rule(pack: PolicyPack, rule: PolicyRule, facts: Facts) -> list[Evidence]:
    relevant = _relevant_categories_for_rule(rule)
    if not relevant:
        return (facts.evidence or [])[: pack.max_evidence_per_finding]

    selected: list[Evidence] = []
    for ev in facts.evidence or []:
        if isinstance(ev, EvidenceScanSummary):
            ev_cats = set(ev.category_counts.keys())
        else:
            ev_cats = set(getattr(ev, "categories", []) or [])
        if ev_cats & relevant:
            selected.append(ev)
        if len(selected) >= pack.max_evidence_per_finding:
            break

    if not selected:
        selected = (facts.evidence or [])[: pack.max_evidence_per_finding]
    return selected


def generate_candidates(pack: PolicyPack, facts: Facts) -> list[FindingCandidate]:
    out: list[FindingCandidate] = []
    for rule in pack.rules:
        if not _rule_applies(rule, facts):
            continue

        matches_any = _condition_matches(rule.when, facts, mode="any")
        if not matches_any:
            continue

        matches_conf = _condition_matches(rule.when, facts, mode="confident")
        needs_llm_due_to_conf = matches_any and (not matches_conf)

        ev = _select_evidence_for_rule(pack, rule, facts)

        reason = None
        if needs_llm_due_to_conf:
            reason = "Condition matched using low-confidence evidence; requires LLM confirmation."

        out.append(
            FindingCandidate(
                rule_id=rule.rule_id,
                issue=rule.issue,
                severity=rule.severity,
                default_actions=list(rule.default_actions),
                evidence=ev,
                needs_llm=bool(rule.needs_llm or needs_llm_due_to_conf),
                reason_needs_llm=reason,
            )
        )

    return out


# -------------------------
# LLM assessor
# -------------------------
def build_llm_assessor_prompt(pack: PolicyPack, facts: Facts, candidates: list[FindingCandidate], context: Any | None = None) -> str:
    actions_payload = [{"action_id": a.action_id, "label": a.label, "description": a.description} for a in pack.actions]

    candidates_payload = []
    for c in candidates:
        candidates_payload.append(
            {
                "rule_id": c.rule_id,
                "issue": c.issue,
                "severity": c.severity,
                "default_actions": c.default_actions,
                "needs_llm": c.needs_llm,
                "reason_needs_llm": c.reason_needs_llm,
                "evidence": [e.model_dump() for e in c.evidence],
            }
        )

    facts_summary = {
        "kind": facts.kind,
        "task_id": facts.task_id,
        "table_id": facts.table_id,
        "signals_present": sorted(facts.signals_present or []),
        "categories_present_any": sorted(facts.categories_present_any or []),
        "categories_present_confident": sorted(facts.categories_present_confident or []),
        "category_counts_any": facts.category_counts_any or {},
        "category_counts_confident": facts.category_counts_confident or {},
    }

    payload = {
        "policy_pack": pack.name,
        "allowed_actions": actions_payload,
        "facts_summary": facts_summary,
        "context": context,
        "candidate_findings": candidates_payload,
    }

    intro = pack.llm_assessor.intro or "You are a privacy risk assessor. Confirm or reject candidate findings using the evidence and context."

    return "\n".join(
        [
            intro,
            "Return ONLY valid JSON.",
            "Output format:",
            "{",
            '  "findings": [',
            "    {",
            '      "rule_id": "string (must match one of the candidate rule_id values)",',
            '      "status": "confirmed|rejected|uncertain",',
            '      "severity": "low|medium|high",',
            '      "recommended_actions": ["action_id", "..."],',
            '      "rationale": "short explanation"',
            "    }",
            "  ]",
            "}",
            "Rules:",
            "- Only output rule_id values that appear in candidate_findings.",
            "- recommended_actions must contain ONLY action_id values from allowed_actions.",
            "- Do not invent new actions.",
            "- Keep rationale short and concrete.",
            "No extra text around the JSON.",
            "",
            "Input JSON:",
            json.dumps(payload, ensure_ascii=True),
        ]
    )


def validate_llm_assessor_response(
    raw_text: str,
    candidate_rule_ids: set[str],
    allowed_action_ids: set[str],
    max_findings: int,
    max_actions_per_finding: int,
    max_rationale_chars: int,
) -> list[AssessedFinding]:
    data = extract_json(raw_text)
    adapter = TypeAdapter(AssessorResponse)
    parsed: AssessorResponse = adapter.validate_python(data)

    findings = parsed.findings[:max_findings]

    seen: set[str] = set()
    out: list[AssessedFinding] = []
    for f in findings:
        if f.rule_id not in candidate_rule_ids:
            continue
        if f.rule_id in seen:
            continue
        seen.add(f.rule_id)

        cleaned_actions: list[str] = []
        for a in f.recommended_actions:
            if a in allowed_action_ids and a not in cleaned_actions:
                cleaned_actions.append(a)
            if len(cleaned_actions) >= max_actions_per_finding:
                break

        rationale = (f.rationale or "").strip()
        if len(rationale) > max_rationale_chars:
            rationale = rationale[:max_rationale_chars].rstrip()

        out.append(
            AssessedFinding(
                rule_id=f.rule_id,
                status=f.status,
                severity=f.severity,
                recommended_actions=cleaned_actions,
                rationale=rationale,
            )
        )

    return out


async def _run_with_retries(llm_client, prompt: str, validator, max_retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt == 0:
            response = await llm_client.generate(prompt)
        else:
            correction = (
                "\n\nThe previous output was invalid: "
                f"{last_error}. Return ONLY valid JSON following the schema."
            )
            response = await llm_client.generate(prompt + correction)
        try:
            return validator(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"LLM output invalid after {max_retries} retries: {last_error}")


async def assess_candidates_with_llm(
    pack: PolicyPack,
    facts: Facts,
    candidates: list[FindingCandidate],
    llm_client,
    settings: Settings,
    context: Any | None = None,
) -> list[AssessedFinding]:
    candidate_rule_ids = {c.rule_id for c in candidates}
    allowed_action_ids = {a.action_id for a in pack.actions}

    prompt = build_llm_assessor_prompt(pack, facts, candidates, context=context)

    def validator(raw_text: str):
        return validate_llm_assessor_response(
            raw_text,
            candidate_rule_ids=candidate_rule_ids,
            allowed_action_ids=allowed_action_ids,
            max_findings=pack.llm_assessor.max_findings,
            max_actions_per_finding=pack.llm_assessor.max_actions_per_finding,
            max_rationale_chars=pack.llm_assessor.max_rationale_chars,
        )

    return await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)


# -------------------------
# Helpers: compute findings from extraction (reusable for escalation)
# -------------------------
def _build_legal_for_rule(pack: PolicyPack, rule_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    if not pack.legal_refs:
        return [], []
    ids = pack.legal_refs.by_rule_id.get(rule_id, [])
    if not ids:
        return [], []
    details = _resolve_legal_refs_detail(pack.legal_refs.source, ids)
    return ids, details


def _severity_priority(severity: Severity) -> int:
    return {"high": 1, "medium": 2, "low": 3}.get(severity, 3)


def _summarize_targets(evidence: list[Evidence]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for ev in evidence:
        if isinstance(ev, EvidenceEntity):
            candidate = f"text[{ev.start}:{ev.end}] {ev.type_id}"
        elif isinstance(ev, EvidenceColumn):
            candidate = f"column:{ev.column} ({ev.type_id})"
        elif isinstance(ev, EvidenceScanSummary):
            candidate = f"column:{ev.column} (scan)"
        else:
            candidate = "dataset"
        if candidate in seen:
            continue
        seen.add(candidate)
        targets.append(candidate)
    if not targets:
        return ["dataset"]
    return targets


def _build_mitigation_plan(
    pack: PolicyPack,
    recommended_actions: list[str],
    evidence: list[Evidence],
    severity: Severity,
) -> list[MitigationStep]:
    action_map = {a.action_id: a for a in pack.actions}
    priority = _severity_priority(severity)
    targets = _summarize_targets(evidence)
    steps: list[MitigationStep] = []
    for action_id in recommended_actions:
        action = action_map.get(action_id)
        if action is None:
            continue
        steps.append(
            MitigationStep(
                action_id=action.action_id,
                label=action.label,
                description=action.description,
                priority=priority,
                targets=targets,
            )
        )
    return steps


def _as_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _classify_assessment(task: dict[str, Any], status: str) -> tuple[AssessmentLevel, list[str]]:
    if status != "confirmed":
        return "compliance_risk", []

    context = task.get("context")
    if not isinstance(context, dict):
        return "compliance_risk", []

    flags = {
        "has_lawful_basis": "Missing lawful basis in processing context.",
        "has_dpa": "Missing processor agreement (DPA) in processing context.",
        "purpose_limited": "Purpose limitation not confirmed in processing context.",
        "has_access_controls": "Access controls not confirmed in processing context.",
        "has_retention_policy": "Retention policy not confirmed in processing context.",
    }
    reasons: list[str] = []
    for key, reason in flags.items():
        if key not in context:
            continue
        parsed = _as_boolish(context.get(key))
        if parsed is False:
            reasons.append(reason)

    if reasons:
        return "possible_violation", reasons
    return "compliance_risk", []


def _build_finding_out(
    *,
    pack: PolicyPack,
    task: dict[str, Any],
    candidate: FindingCandidate,
    status: Literal["confirmed", "rejected", "uncertain"],
    severity: Severity,
    recommended_actions: list[str],
    rationale: str,
) -> FindingOut:
    legal_refs, legal_refs_detail = _build_legal_for_rule(pack, candidate.rule_id)
    assessment, violation_reasons = _classify_assessment(task, status)
    mitigation_plan = _build_mitigation_plan(pack, recommended_actions, candidate.evidence, severity)
    return FindingOut(
        rule_id=candidate.rule_id,
        issue=candidate.issue,
        status=status,
        severity=severity,
        recommended_actions=recommended_actions,
        rationale=rationale,
        evidence=candidate.evidence,
        assessment=assessment,
        violation_reasons=violation_reasons,
        mitigation_plan=mitigation_plan,
        legal_refs=legal_refs,
        legal_refs_detail=legal_refs_detail,
    )


def _infer_table_columns(sampled_rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in sampled_rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if not isinstance(key, str):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _resolve_scan_columns(task: dict[str, Any]) -> list[str]:
    explicit = task.get("scan_columns")
    if isinstance(explicit, list):
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in explicit:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cleaned.append(name)
        if cleaned:
            return cleaned
    sampled_rows = task.get("sampled_rows")
    if isinstance(sampled_rows, list):
        return _infer_table_columns(sampled_rows)
    return []


async def _compute_findings_for_text_task(
    *,
    pack: PolicyPack,
    settings: Settings,
    llm_client,
    analysis_mode: Literal["rules", "hybrid"],
    task: dict[str, Any],
    extraction: dict[str, Any],
) -> list[FindingOut]:
    facts = build_text_facts(pack, task_id=task["task_id"], text=task.get("text", ""), entities=extraction.get("entities", []))
    candidates = generate_candidates(pack, facts)

    # baseline
    findings: list[FindingOut] = []
    for c in candidates:
        findings.append(
            _build_finding_out(
                pack=pack,
                task=task,
                candidate=c,
                status=("uncertain" if c.needs_llm else "confirmed"),
                severity=c.severity,
                recommended_actions=list(c.default_actions),
                rationale=(
                    c.reason_needs_llm
                    if c.needs_llm and c.reason_needs_llm
                    else "Generated by deterministic policy rule."
                ),
            )
        )

    # assessor
    if analysis_mode == "hybrid" and candidates:
        assessed = await assess_candidates_with_llm(pack, facts, candidates, llm_client, settings=settings, context=task.get("context"))
        assessed_by_id = {a.rule_id: a for a in assessed}

        merged: list[FindingOut] = []
        for c in candidates:
            a = assessed_by_id.get(c.rule_id)

            if a is None:
                merged.append(
                    _build_finding_out(
                        pack=pack,
                        task=task,
                        candidate=c,
                        status=("uncertain" if c.needs_llm else "confirmed"),
                        severity=c.severity,
                        recommended_actions=list(c.default_actions),
                        rationale="No LLM assessment returned; using deterministic default.",
                    )
                )
                continue

            if a.status == "rejected":
                continue

            merged.append(
                _build_finding_out(
                    pack=pack,
                    task=task,
                    candidate=c,
                    status=a.status,
                    severity=a.severity,
                    recommended_actions=a.recommended_actions or list(c.default_actions),
                    rationale=a.rationale or "Assessed by LLM.",
                )
            )

        findings = merged

    return findings


async def _compute_findings_for_table_task(
    *,
    pack: PolicyPack,
    settings: Settings,
    llm_client,
    analysis_mode: Literal["rules", "hybrid"],
    task: dict[str, Any],
    typing_extraction: dict[str, Any],
    scan_result: dict[str, Any] | None,
) -> list[FindingOut]:
    facts = build_table_facts(
        pack,
        task_id=task["task_id"],
        table_id=task.get("table_id", ""),
        sampled_rows=task.get("sampled_rows", []),
        columns=typing_extraction.get("columns", []),
        scan_result=scan_result,
    )
    candidates = generate_candidates(pack, facts)

    findings: list[FindingOut] = []
    for c in candidates:
        findings.append(
            _build_finding_out(
                pack=pack,
                task=task,
                candidate=c,
                status=("uncertain" if c.needs_llm else "confirmed"),
                severity=c.severity,
                recommended_actions=list(c.default_actions),
                rationale=(
                    c.reason_needs_llm
                    if c.needs_llm and c.reason_needs_llm
                    else "Generated by deterministic policy rule."
                ),
            )
        )

    if analysis_mode == "hybrid" and candidates:
        assessed = await assess_candidates_with_llm(pack, facts, candidates, llm_client, settings=settings, context=task.get("context"))
        assessed_by_id = {a.rule_id: a for a in assessed}

        merged: list[FindingOut] = []
        for c in candidates:
            a = assessed_by_id.get(c.rule_id)

            if a is None:
                merged.append(
                    _build_finding_out(
                        pack=pack,
                        task=task,
                        candidate=c,
                        status=("uncertain" if c.needs_llm else "confirmed"),
                        severity=c.severity,
                        recommended_actions=list(c.default_actions),
                        rationale="No LLM assessment returned; using deterministic default.",
                    )
                )
                continue

            if a.status == "rejected":
                continue

            merged.append(
                _build_finding_out(
                    pack=pack,
                    task=task,
                    candidate=c,
                    status=a.status,
                    severity=a.severity,
                    recommended_actions=a.recommended_actions or list(c.default_actions),
                    rationale=a.rationale or "Assessed by LLM.",
                )
            )
        findings = merged

    return findings


def _has_uncertain(findings: list[dict[str, Any]] | list[FindingOut]) -> bool:
    for f in findings:
        status = f.status if isinstance(f, FindingOut) else f.get("status")
        if status == "uncertain":
            return True
    return False


def _extract_profile_escalation(profile_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(profile_cfg, dict):
        return []
    esc = profile_cfg.get("escalation") or []
    return esc if isinstance(esc, list) else []


def _inc_count(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _sorted_counts(counter: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _build_machine_privacy_report(
    *,
    generated_at: str,
    profile_name: str | None,
    policy_pack: str,
    results_out: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    escalations_out: list[dict[str, Any]],
) -> dict[str, Any]:
    severity_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    assessment_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}

    tasks_out: list[MachineReportTask] = []
    total_findings = 0

    for task in results_out:
        task_id = str(task.get("task_id") or "")
        kind = str(task.get("kind") or "unknown")
        table_id = task.get("table_id")
        table_id_val = str(table_id) if isinstance(table_id, str) and table_id.strip() else None

        findings = task.get("findings") or []
        finding_items: list[MachineReportFinding] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity") or "unknown")
            status = str(finding.get("status") or "unknown")
            assessment = str(finding.get("assessment") or "compliance_risk")
            actions = [str(a) for a in (finding.get("recommended_actions") or [])]

            _inc_count(severity_counts, severity)
            _inc_count(status_counts, status)
            _inc_count(assessment_counts, assessment)
            for action in actions:
                _inc_count(action_counts, action)

            evidence = finding.get("evidence") or []
            legal_refs_detail = finding.get("legal_refs_detail") or []
            mitigation_plan = finding.get("mitigation_plan") or []
            violation_reasons = finding.get("violation_reasons") or []
            legal_refs = finding.get("legal_refs") or []

            finding_items.append(
                MachineReportFinding(
                    rule_id=str(finding.get("rule_id") or ""),
                    issue=str(finding.get("issue") or ""),
                    severity=severity,
                    status=status,
                    assessment=assessment,
                    rationale=str(finding.get("rationale") or ""),
                    recommended_actions=actions,
                    mitigation_plan=[m for m in mitigation_plan if isinstance(m, dict)],
                    violation_reasons=[str(v) for v in violation_reasons],
                    legal_refs=[str(r) for r in legal_refs],
                    legal_refs_detail=[d for d in legal_refs_detail if isinstance(d, dict)],
                    evidence=[e for e in evidence if isinstance(e, dict)],
                )
            )

        total_findings += len(finding_items)
        tasks_out.append(
            MachineReportTask(
                task_id=task_id,
                kind=kind,
                table_id=table_id_val,
                finding_count=len(finding_items),
                findings=finding_items,
            )
        )

    report = MachinePrivacyReport(
        generated_at=generated_at,
        profile=profile_name,
        policy_pack=policy_pack,
        summary=MachineReportSummary(
            task_count=len(tasks_out),
            finding_count=total_findings,
            severity_counts=_sorted_counts(severity_counts),
            status_counts=_sorted_counts(status_counts),
            assessment_counts=_sorted_counts(assessment_counts),
            action_counts=_sorted_counts(action_counts),
        ),
        tasks=tasks_out,
        warnings=warnings,
        escalations=escalations_out,
    )
    return report.model_dump()


def _format_counts_for_md(counter: dict[str, int]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{k}: {v}" for k, v in counter.items())


def _summarize_evidence_for_md(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "unknown")
    if kind == "entity":
        text = str(item.get("text") or "")
        type_id = str(item.get("type_id") or "")
        conf = item.get("confidence")
        return f"entity `{type_id}` (confidence={conf}) text=\"{text}\""
    if kind == "column":
        column = str(item.get("column") or "")
        type_id = str(item.get("type_id") or "")
        conf = item.get("confidence")
        return f"column `{column}` typed as `{type_id}` (confidence={conf})"
    if kind == "scan_summary":
        column = str(item.get("column") or "")
        cats = item.get("category_counts") or {}
        return f"scan summary for column `{column}` categories={cats}"
    return json.dumps(item, ensure_ascii=True)


def _build_human_privacy_markdown(machine_report: dict[str, Any]) -> str:
    summary = machine_report.get("summary") or {}
    lines: list[str] = []
    lines.append("# Privacy Analysis Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): {machine_report.get('generated_at')}")
    lines.append(f"- Profile: `{machine_report.get('profile')}`")
    lines.append(f"- Policy pack: `{machine_report.get('policy_pack')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Tasks analyzed: {summary.get('task_count', 0)}")
    lines.append(f"- Findings identified: {summary.get('finding_count', 0)}")
    lines.append(f"- Severity distribution: {_format_counts_for_md(summary.get('severity_counts') or {})}")
    lines.append(f"- Status distribution: {_format_counts_for_md(summary.get('status_counts') or {})}")
    lines.append(f"- Assessment distribution: {_format_counts_for_md(summary.get('assessment_counts') or {})}")
    lines.append(f"- Action frequency: {_format_counts_for_md(summary.get('action_counts') or {})}")
    lines.append("")

    for task in machine_report.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        kind = task.get("kind")
        table_id = task.get("table_id")
        title = f"## Task `{task_id}` ({kind})"
        if table_id:
            title += f" table_id=`{table_id}`"
        lines.append(title)
        lines.append(f"- Findings: {task.get('finding_count', 0)}")

        findings = task.get("findings") or []
        if not findings:
            lines.append("- No findings for this task.")
            lines.append("")
            continue

        for idx, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue
            lines.append(
                f"### {idx}. {str(finding.get('severity') or '').upper()} | "
                f"{finding.get('issue')} (`{finding.get('rule_id')}`)"
            )
            lines.append(f"- Status: `{finding.get('status')}`")
            lines.append(f"- Assessment: `{finding.get('assessment')}`")
            lines.append(f"- Rationale: {finding.get('rationale')}")

            actions = finding.get("recommended_actions") or []
            if actions:
                lines.append(f"- Recommended actions: {', '.join(str(a) for a in actions)}")

            reasons = finding.get("violation_reasons") or []
            if reasons:
                lines.append(f"- Violation signals: {', '.join(str(r) for r in reasons)}")

            mitigation = finding.get("mitigation_plan") or []
            if mitigation:
                lines.append("- Mitigation plan:")
                for step in mitigation:
                    if not isinstance(step, dict):
                        continue
                    lines.append(
                        f"  - (p{step.get('priority')}) `{step.get('action_id')}`: "
                        f"{step.get('label')} targets={step.get('targets') or []}"
                    )

            legal_detail = finding.get("legal_refs_detail") or []
            legal_refs = finding.get("legal_refs") or []
            if legal_detail:
                lines.append("- GDPR/legal references:")
                for ref in legal_detail:
                    if not isinstance(ref, dict):
                        continue
                    label = ref.get("label") or ref.get("id")
                    lines.append(f"  - {label} ({ref.get('id')})")
            elif legal_refs:
                lines.append(f"- GDPR/legal references: {', '.join(str(x) for x in legal_refs)}")

            evidence = finding.get("evidence") or []
            if evidence:
                lines.append("- Evidence:")
                for e in evidence:
                    if not isinstance(e, dict):
                        continue
                    lines.append(f"  - {_summarize_evidence_for_md(e)}")
            lines.append("")

    warnings = machine_report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            if not isinstance(w, dict):
                continue
            code = w.get("code") or "warning"
            lines.append(f"- {code}: {json.dumps(w, ensure_ascii=True)}")
        lines.append("")

    escalations = machine_report.get("escalations") or []
    if escalations:
        lines.append("## Escalations")
        for e in escalations:
            if not isinstance(e, dict):
                continue
            lines.append(f"- {json.dumps(e, ensure_ascii=True)}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_privacy_reports(
    *,
    profile_name: str | None,
    policy_pack: str,
    results_out: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    escalations_out: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    machine_content = _build_machine_privacy_report(
        generated_at=generated_at,
        profile_name=profile_name,
        policy_pack=policy_pack,
        results_out=results_out,
        warnings=warnings,
        escalations_out=escalations_out,
    )
    human_md = _build_human_privacy_markdown(machine_content)
    reports = PrivacyReports(
        human_readable=HumanReadableReport(content=human_md),
        machine_readable=MachineReadableReport(content=machine_content),
    )
    return reports.model_dump()


# -------------------------
# Main orchestration function (with progressive escalation)
# -------------------------
async def run_privacy_analyze(
    tasks: list[dict[str, Any]],
    policy_pack: str | None,
    llm_client,
    profile: str | None = None,
    analysis_mode: Literal["rules", "hybrid"] | None = None,
    text_schema: str | None = None,
    table_schema: str | None = None,
    scan_schema: str | None = None,
    include_extraction: bool | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()

    resolved = resolve_privacy_defaults(
        profile,
        overrides={
            "policy_pack": policy_pack,
            "analysis_mode": analysis_mode,
            "text_schema": text_schema,
            "table_schema": table_schema,
            "scan_schema": scan_schema,
            "include_extraction": include_extraction,
        },
    )

    profile_name = resolved.get("_profile")
    profile_cfg = resolved.get("_profile_config")

    policy_pack = resolved["policy_pack"]
    analysis_mode = resolved["analysis_mode"]
    text_schema = resolved["text_schema"]
    table_schema = resolved["table_schema"]
    scan_schema = resolved["scan_schema"]
    include_extraction = resolved["include_extraction"]

    if analysis_mode not in {"rules", "hybrid"}:
        raise ValueError(f"Invalid analysis_mode: {analysis_mode}")

    pack = load_policy_pack(policy_pack)

    # Schema sanity checks (AFTER profile resolution)
    if not get_schema_config(text_schema).supports_text:
        raise ValueError(f"text_schema '{text_schema}' does not support text annotation.")
    if not get_schema_config(table_schema).supports_table:
        raise ValueError(f"table_schema '{table_schema}' does not support tabular annotation.")
    if not get_schema_config(scan_schema).supports_text:
        raise ValueError(f"scan_schema '{scan_schema}' does not support text annotation.")

    text_tasks = [t for t in tasks if t.get("kind") == "text"]
    table_tasks = [t for t in tasks if t.get("kind") == "table"]

    warnings: list[dict[str, Any]] = []
    escalations_out: list[dict[str, Any]] = []

    # -------------------------
    # Extraction pass 1
    # -------------------------
    text_extraction_by_id: dict[str, dict[str, Any]] = {}
    if text_tasks:
        ner_input = [{"task_id": t["task_id"], "text": t["text"]} for t in text_tasks]
        ner_out = await run_text_ner(
            ner_input,
            text_schema,
            llm_client,
            settings=settings,
        )
        for item in ner_out.get("results", []):
            text_extraction_by_id[item["task_id"]] = item
        warnings.extend(ner_out.get("warnings", []))

    table_extraction_by_id: dict[str, dict[str, Any]] = {}
    if table_tasks:
        tab_input = [
            {"task_id": t["task_id"], "table_id": t["table_id"], "sampled_rows": t["sampled_rows"]}
            for t in table_tasks
        ]
        tab_out = await run_table_annotate(
            tab_input,
            table_schema,
            llm_client,
            settings=settings,
        )
        for item in tab_out.get("results", []):
            table_extraction_by_id[item["task_id"]] = item

    # optional scan pass 1 (only if scan_columns provided)
    scan_tasks = []
    resolved_scan_columns: dict[str, list[str]] = {}
    for t in table_tasks:
        scan_cols = _resolve_scan_columns(t)
        task_id = t.get("task_id")
        if isinstance(task_id, str):
            resolved_scan_columns[task_id] = scan_cols
        if scan_cols:
            scan_tasks.append(
                {
                    "task_id": t["task_id"],
                    "table_id": t["table_id"],
                    "sampled_rows": t["sampled_rows"],
                    "target_columns": scan_cols,
                    "strings_only": True,
                    "skip_structured_literals": True,
                }
            )

    table_scan_by_id: dict[str, dict[str, Any]] = {}
    if scan_tasks:
        try:
            scan_out = await run_tabular_ner(
                scan_tasks,
                scan_schema,
                llm_client,
                settings=settings,
            )
            for item in scan_out.get("results", []):
                table_scan_by_id[item["task_id"]] = item
            warnings.extend(scan_out.get("warnings", []))
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                {
                    "code": "table_scan_failed",
                    "schema": scan_schema,
                    "task_ids": [t.get("task_id") for t in scan_tasks if isinstance(t, dict)],
                    "error": str(exc),
                }
            )

    # -------------------------
    # Analysis pass 1
    # -------------------------
    results_out: list[dict[str, Any]] = []
    results_by_task_id: dict[str, dict[str, Any]] = {}

    for t in tasks:
        kind = t.get("kind")
        task_id = t.get("task_id")

        if kind == "text":
            extraction = text_extraction_by_id.get(task_id, {"task_id": task_id, "entities": []})
            try:
                findings_models = await _compute_findings_for_text_task(
                    pack=pack,
                    settings=settings,
                    llm_client=llm_client,
                    analysis_mode=analysis_mode,
                    task=t,
                    extraction=extraction,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append({"code": "findings_failed", "task_id": task_id, "error": str(exc)})
                findings_models = []

            out_item: dict[str, Any] = {
                "task_id": task_id,
                "kind": "text",
                "policy_pack": pack.name,
                "findings": [f.model_dump() for f in findings_models],
            }
            if include_extraction:
                out_item["extraction"] = extraction

        elif kind == "table":
            typing_extraction = table_extraction_by_id.get(
                task_id,
                {"task_id": task_id, "table_id": t.get("table_id"), "columns": []},
            )
            scan_result = table_scan_by_id.get(task_id)

            try:
                findings_models = await _compute_findings_for_table_task(
                    pack=pack,
                    settings=settings,
                    llm_client=llm_client,
                    analysis_mode=analysis_mode,
                    task=t,
                    typing_extraction=typing_extraction,
                    scan_result=scan_result,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append({"code": "findings_failed", "task_id": task_id, "error": str(exc)})
                findings_models = []

            out_item = {
                "task_id": task_id,
                "kind": "table",
                "policy_pack": pack.name,
                "findings": [f.model_dump() for f in findings_models],
            }
            if include_extraction:
                out_item["extraction"] = typing_extraction

        else:
            raise ValueError(f"Unknown task kind: {kind!r}")

        results_out.append(out_item)
        results_by_task_id[task_id] = out_item

    # -------------------------
    # Progressive escalation helpers (best-of-two merge for text)
    # -------------------------
    def _merge_entities_best_of_two(
        base_entities: list[dict[str, Any]] | None,
        esc_entities: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Merge two entity lists:
        - keep baseline order
        - add new entities from escalated pass
        - for exact key match (start,end,type_id), keep higher confidence
        """
        base_entities = base_entities or []
        esc_entities = esc_entities or []

        def key(e: dict[str, Any]) -> tuple[int, int, str]:
            return (int(e.get("start", -1)), int(e.get("end", -1)), str(e.get("type_id", "")))

        merged: list[dict[str, Any]] = []
        index: dict[tuple[int, int, str], int] = {}

        # start with baseline (preserve order)
        for e in base_entities:
            if not isinstance(e, dict):
                continue
            k = key(e)
            index[k] = len(merged)
            merged.append(e)

        # overlay + append escalated entities
        for e in esc_entities:
            if not isinstance(e, dict):
                continue
            k = key(e)
            if k in index:
                i = index[k]
                # keep higher confidence, but preserve baseline ordering
                try:
                    base_conf = float(merged[i].get("confidence", 0.0))
                except Exception:  # noqa: BLE001
                    base_conf = 0.0
                try:
                    new_conf = float(e.get("confidence", 0.0))
                except Exception:  # noqa: BLE001
                    new_conf = 0.0
                if new_conf > base_conf:
                    merged[i] = {**merged[i], **e}
            else:
                index[k] = len(merged)
                merged.append(e)

        return merged

    # -------------------------
    # Progressive escalation (text-only v1; optional table scan v1.1 if profile provides scan_schema)
    # -------------------------
    escalation_rules = _extract_profile_escalation(profile_cfg)

    if escalation_rules:
        task_order = [t.get("task_id") for t in tasks if isinstance(t, dict)]
        task_by_id = {t.get("task_id"): t for t in tasks if isinstance(t, dict)}

        for esc in escalation_rules:
            if not isinstance(esc, dict):
                continue
            if esc.get("when") != "uncertain_findings":
                continue

            max_extra_passes = int(esc.get("max_extra_passes", 1) or 1)
            if max_extra_passes <= 0:
                continue

            max_escalated_tasks = int(esc.get("max_escalated_tasks", 10) or 10)
            max_escalated_tasks = max(1, min(max_escalated_tasks, 50))

            # --------
            # Text escalation with best-of-two merge
            # --------
            esc_text_schema = esc.get("text_schema")
            if isinstance(esc_text_schema, str) and esc_text_schema.strip():
                esc_text_schema = esc_text_schema.strip()

                uncertain_text_ids: list[str] = []
                for tid in task_order:
                    item = results_by_task_id.get(tid)
                    if not item or item.get("kind") != "text":
                        continue
                    if _has_uncertain(item.get("findings", [])):
                        uncertain_text_ids.append(tid)

                if uncertain_text_ids:
                    uncertain_text_ids = uncertain_text_ids[:max_escalated_tasks]

                    try:
                        rerun_tasks = [{"task_id": tid, "text": task_by_id[tid]["text"]} for tid in uncertain_text_ids if tid in task_by_id]
                        ner_out2 = await run_text_ner(
                            rerun_tasks,
                            esc_text_schema,
                            llm_client,
                            settings=settings,
                        )
                        warnings.extend(ner_out2.get("warnings", []))

                        rerun_map = {
                            r["task_id"]: r
                            for r in ner_out2.get("results", [])
                            if isinstance(r, dict) and "task_id" in r
                        }

                        for tid in uncertain_text_ids:
                            t = task_by_id.get(tid)
                            if not t:
                                continue

                            base_extraction = text_extraction_by_id.get(tid) or {"task_id": tid, "entities": []}
                            esc_extraction = rerun_map.get(tid) or {"task_id": tid, "entities": []}

                            merged_entities = _merge_entities_best_of_two(
                                base_extraction.get("entities", []),
                                esc_extraction.get("entities", []),
                            )
                            merged_extraction = {"task_id": tid, "entities": merged_entities}

                            findings_models = await _compute_findings_for_text_task(
                                pack=pack,
                                settings=settings,
                                llm_client=llm_client,
                                analysis_mode=analysis_mode,
                                task=t,
                                extraction=merged_extraction,
                            )

                            updated_item: dict[str, Any] = {
                                "task_id": tid,
                                "kind": "text",
                                "policy_pack": pack.name,
                                "findings": [f.model_dump() for f in findings_models],
                            }
                            if include_extraction:
                                updated_item["extraction"] = merged_extraction

                            results_by_task_id[tid] = updated_item
                            for i, old in enumerate(results_out):
                                if old.get("task_id") == tid:
                                    results_out[i] = updated_item
                                    break

                            escalations_out.append(
                                {
                                    "task_id": tid,
                                    "kind": "text",
                                    "reason": "uncertain_findings",
                                    "stage": "text_schema",
                                    "from": text_schema,
                                    "to": esc_text_schema,
                                    "status": "applied",
                                    "merge": "best_of_two",
                                }
                            )

                    except Exception as exc:  # noqa: BLE001
                        warnings.append({"code": "escalation_failed", "stage": "text_schema", "error": str(exc)})
                        for tid in uncertain_text_ids:
                            escalations_out.append(
                                {
                                    "task_id": tid,
                                    "kind": "text",
                                    "reason": "uncertain_findings",
                                    "stage": "text_schema",
                                    "from": text_schema,
                                    "to": esc_text_schema,
                                    "status": "failed",
                                    "error": str(exc),
                                }
                            )

            # --------
            # Optional table scan escalation (only if profile provides scan_schema)
            # --------
            esc_scan_schema = esc.get("scan_schema")
            only_if_scan_columns_present = bool(esc.get("only_if_scan_columns_present", True))
            if isinstance(esc_scan_schema, str) and esc_scan_schema.strip():
                esc_scan_schema = esc_scan_schema.strip()

                uncertain_table_ids: list[str] = []
                for tid in task_order:
                    item = results_by_task_id.get(tid)
                    if not item or item.get("kind") != "table":
                        continue
                    if _has_uncertain(item.get("findings", [])):
                        if not only_if_scan_columns_present:
                            uncertain_table_ids.append(tid)
                        else:
                            scan_cols = resolved_scan_columns.get(tid) or []
                            if scan_cols:
                                uncertain_table_ids.append(tid)

                if uncertain_table_ids:
                    uncertain_table_ids = uncertain_table_ids[:max_escalated_tasks]
                    scan_tasks2 = []
                    for tid in uncertain_table_ids:
                        t = task_by_id.get(tid)
                        if not t:
                            continue
                        scan_cols = resolved_scan_columns.get(tid) or _resolve_scan_columns(t)
                        if not scan_cols and only_if_scan_columns_present:
                            continue
                        scan_tasks2.append(
                            {
                                "task_id": t["task_id"],
                                "table_id": t["table_id"],
                                "sampled_rows": t["sampled_rows"],
                                "target_columns": scan_cols,
                                "strings_only": True,
                                "skip_structured_literals": True,
                            }
                        )

                    if scan_tasks2:
                        try:
                            scan_out2 = await run_tabular_ner(
                                scan_tasks2,
                                esc_scan_schema,
                                llm_client,
                                settings=settings,
                            )
                            warnings.extend(scan_out2.get("warnings", []))
                            rerun_scan_map = {
                                r["task_id"]: r
                                for r in scan_out2.get("results", [])
                                if isinstance(r, dict) and "task_id" in r
                            }

                            for tid in uncertain_table_ids:
                                t = task_by_id.get(tid)
                                if not t:
                                    continue
                                typing_extraction = table_extraction_by_id.get(
                                    tid,
                                    {"task_id": tid, "table_id": t.get("table_id"), "columns": []},
                                )
                                new_scan_result = rerun_scan_map.get(tid)

                                findings_models = await _compute_findings_for_table_task(
                                    pack=pack,
                                    settings=settings,
                                    llm_client=llm_client,
                                    analysis_mode=analysis_mode,
                                    task=t,
                                    typing_extraction=typing_extraction,
                                    scan_result=new_scan_result,
                                )

                                updated_item = {
                                    "task_id": tid,
                                    "kind": "table",
                                    "policy_pack": pack.name,
                                    "findings": [f.model_dump() for f in findings_models],
                                }
                                if include_extraction:
                                    updated_item["extraction"] = typing_extraction

                                results_by_task_id[tid] = updated_item
                                for i, old in enumerate(results_out):
                                    if old.get("task_id") == tid:
                                        results_out[i] = updated_item
                                        break

                                escalations_out.append(
                                    {
                                        "task_id": tid,
                                        "kind": "table",
                                        "reason": "uncertain_findings",
                                        "stage": "scan_schema",
                                        "from": scan_schema,
                                        "to": esc_scan_schema,
                                        "status": "applied",
                                    }
                                )

                        except Exception as exc:  # noqa: BLE001
                            warnings.append({"code": "escalation_failed", "stage": "scan_schema", "error": str(exc)})
                            for tid in uncertain_table_ids:
                                escalations_out.append(
                                    {
                                        "task_id": tid,
                                        "kind": "table",
                                        "reason": "uncertain_findings",
                                        "stage": "scan_schema",
                                        "from": scan_schema,
                                        "to": esc_scan_schema,
                                        "status": "failed",
                                        "error": str(exc),
                                    }
                                )

            # v1: apply first matching rule once
            break

    response: dict[str, Any] = {
        "profile": profile_name,
        "policy_pack": pack.name,
        "action_catalog": [a.model_dump() for a in pack.actions],
        "results": results_out,
        "reports": _build_privacy_reports(
            profile_name=profile_name,
            policy_pack=pack.name,
            results_out=results_out,
            warnings=warnings,
            escalations_out=escalations_out,
        ),
    }
    if warnings:
        response["warnings"] = warnings
    if escalations_out:
        response["escalations"] = escalations_out
    return response
