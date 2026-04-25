from __future__ import annotations

"""Entity-aware validation helpers for companies, papers, and people."""

from typing import List

from core.evidence import ValidationDecision, country_evidence, keyword_evidence
from core.geography import contains_country_or_city
from core.task_models import TaskSpec


def _text_blob(record) -> str:
    parts = [
        getattr(record, "company_name", ""),
        getattr(record, "description", ""),
        getattr(record, "website", ""),
        getattr(record, "source_url", ""),
        getattr(record, "notes", ""),
        getattr(record, "authors", ""),
        getattr(record, "job_title", ""),
        getattr(record, "employer_name", ""),
        getattr(record, "linkedin_profile", ""),
        getattr(record, "linkedin_url", ""),
        getattr(record, "hq_country", ""),
        " ".join(getattr(record, "presence_countries", []) or []),
    ]
    return " ".join(str(p or "") for p in parts).strip()


def _geo_validate(record, task_spec: TaskSpec, decision: ValidationDecision) -> None:
    geo = getattr(task_spec, "geography", None)
    if not geo:
        return
    text = _text_blob(record)
    if geo.include_countries:
        include_hits = [c for c in geo.include_countries if contains_country_or_city(text, c)]
        if include_hits:
            decision.evidence.extend(country_evidence(text, include_hits, getattr(record, "source_url", ""), "record"))
        elif getattr(geo, "strict_mode", False) and getattr(geo, "require_geo_evidence", True):
            decision.accepted = False
            decision.reasons.append("missing_include_country_evidence")
            decision.score_delta -= 20

    excluded = list(getattr(geo, "exclude_countries", []) or []) + list(getattr(geo, "exclude_presence_countries", []) or [])
    matched_excluded = [c for c in excluded if contains_country_or_city(text, c)]
    if matched_excluded:
        decision.accepted = False
        decision.reasons.append(f"matched_excluded_geo:{','.join(matched_excluded)}")
        decision.score_delta -= 25
        decision.evidence.extend(country_evidence(text, matched_excluded, getattr(record, "source_url", ""), "record"))


def validate_company_record(record, task_spec: TaskSpec) -> ValidationDecision:
    decision = ValidationDecision(accepted=True)
    text = _text_blob(record)

    if not getattr(record, "company_name", "") and not getattr(record, "website", ""):
        decision.accepted = False
        decision.reasons.append("missing_identity")
        decision.score_delta -= 20

    category = getattr(task_spec, "target_category", "general")
    if category == "software_company":
        hints = ["software", "platform", "saas", "analytics", "automation", "digital", "ai", "cloud"]
        ev = keyword_evidence(text, hints, getattr(record, "source_url", ""), "record", "software")
        if ev:
            decision.evidence.extend(ev)
        else:
            decision.warnings.append("weak_software_signal")
            decision.score_delta -= 5
    elif category == "service_company":
        hints = ["service", "services", "contractor", "engineering", "maintenance", "inspection", "wireline", "logging"]
        ev = keyword_evidence(text, hints, getattr(record, "source_url", ""), "record", "service")
        if ev:
            decision.evidence.extend(ev)
        else:
            decision.warnings.append("weak_service_signal")
            decision.score_delta -= 5

    if getattr(task_spec, "solution_keywords", None):
        decision.evidence.extend(keyword_evidence(text, task_spec.solution_keywords, getattr(record, "source_url", ""), "record", "solution"))
    if getattr(task_spec, "domain_keywords", None):
        decision.evidence.extend(keyword_evidence(text, task_spec.domain_keywords, getattr(record, "source_url", ""), "record", "domain"))

    _geo_validate(record, task_spec, decision)
    return decision


def validate_paper_record(record, task_spec: TaskSpec) -> ValidationDecision:
    decision = ValidationDecision(accepted=True)
    if not getattr(record, "company_name", "") and not getattr(record, "description", ""):
        decision.accepted = False
        decision.reasons.append("missing_title_or_abstract")
        decision.score_delta -= 20
    if not getattr(record, "website", ""):
        decision.warnings.append("missing_source_url")
    text = _text_blob(record)
    if getattr(task_spec, "industry", "") and task_spec.industry.lower() not in text.lower():
        decision.warnings.append("industry_not_explicit_in_paper_text")
    return decision


def validate_person_record(record, task_spec: TaskSpec) -> ValidationDecision:
    decision = ValidationDecision(accepted=True)
    if not getattr(record, "linkedin_profile", "") and not getattr(record, "linkedin_url", ""):
        decision.accepted = False
        decision.reasons.append("missing_profile_url")
        decision.score_delta -= 20
    if not getattr(record, "company_name", ""):
        decision.warnings.append("missing_person_name")
    _geo_validate(record, task_spec, decision)
    return decision


def validate_record(record, task_spec: TaskSpec) -> ValidationDecision:
    entity = (getattr(task_spec, "target_entity_types", []) or ["company"])[0]
    if entity == "paper" or getattr(task_spec, "task_type", "") == "document_research":
        return validate_paper_record(record, task_spec)
    if entity == "person" or getattr(task_spec, "task_type", "") == "people_search":
        return validate_person_record(record, task_spec)
    return validate_company_record(record, task_spec)
