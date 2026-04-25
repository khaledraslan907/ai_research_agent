from __future__ import annotations

"""Lightweight critic helpers for task specs, query plans, and result sets."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.task_models import TaskSpec


@dataclass
class CriticIssue:
    severity: str = "info"  # info | warning | error
    code: str = ""
    message: str = ""
    recommendation: str = ""


def review_task_spec(task_spec: TaskSpec) -> List[CriticIssue]:
    issues: List[CriticIssue] = []
    if not getattr(task_spec, "industry", ""):
        issues.append(CriticIssue("error", "missing_industry", "No clear topic/industry detected.", "Add an industry or subject phrase to the prompt."))
    if not getattr(task_spec, "target_entity_types", None):
        issues.append(CriticIssue("warning", "missing_entity", "No explicit entity type detected.", "Default to companies or specify papers/people."))
    geo = getattr(task_spec, "geography", None)
    if geo and geo.strict_mode and not geo.include_countries and not geo.exclude_countries and not geo.exclude_presence_countries:
        issues.append(CriticIssue("warning", "strict_without_geo", "Strict geography mode is on but no country filters exist.", "Disable strict geography or add include/exclude countries."))
    if getattr(task_spec, "commercial_intent", "general") != "general" and getattr(task_spec, "primary_entity_type", lambda: "company")() != "company":
        issues.append(CriticIssue("info", "commercial_non_company", "Commercial intent was set for a non-company search.", "This is okay, but ranking may be weaker."))
    return issues


def review_query_plan(plan: Dict[str, List], task_spec: TaskSpec) -> List[CriticIssue]:
    issues: List[CriticIssue] = []
    total = sum(len(v or []) for v in (plan or {}).values())
    if total == 0:
        issues.append(CriticIssue("error", "no_queries", "No queries were generated.", "Fallback to template queries."))
        return issues

    include_countries = list(getattr(getattr(task_spec, "geography", None), "include_countries", []) or [])
    geo_texts = [c.lower() for c in include_countries]
    geo_hits = 0
    for provider_queries in (plan or {}).values():
        for q in provider_queries or []:
            text = getattr(q, "text", str(q)).lower()
            if any(g in text for g in geo_texts):
                geo_hits += 1
    if include_countries and geo_hits == 0:
        issues.append(CriticIssue("warning", "missing_geo_in_queries", "Generated queries do not mention included countries.", "Inject geography phrases into at least some queries."))

    if getattr(task_spec, "domain_keywords", None):
        dom_hits = 0
        for provider_queries in (plan or {}).values():
            for q in provider_queries or []:
                text = getattr(q, "text", str(q)).lower()
                if any(str(k).lower() in text for k in task_spec.domain_keywords):
                    dom_hits += 1
        if dom_hits == 0:
            issues.append(CriticIssue("warning", "missing_domain_keywords", "Generated queries dropped the domain keywords.", "Add niche terms explicitly to query families."))
    return issues


def review_results(records: Iterable, task_spec: TaskSpec) -> List[CriticIssue]:
    issues: List[CriticIssue] = []
    rows = list(records or [])
    if not rows:
        issues.append(CriticIssue("error", "no_results", "No accepted results.", "Relax confidence, widen queries, or add providers."))
        return issues

    low_conf = [r for r in rows if float(getattr(r, "confidence_score", 0.0) or 0.0) < 40]
    if len(low_conf) > len(rows) * 0.5:
        issues.append(CriticIssue("warning", "many_low_confidence", "More than half the accepted results are low confidence.", "Tighten validators or raise minimum score."))

    include_countries = list(getattr(getattr(task_spec, "geography", None), "include_countries", []) or [])
    if include_countries:
        weak_geo = 0
        for rec in rows:
            text = " ".join([
                str(getattr(rec, "hq_country", "") or ""),
                " ".join(getattr(rec, "presence_countries", []) or []),
                str(getattr(rec, "description", "") or ""),
                str(getattr(rec, "notes", "") or ""),
            ]).lower()
            if not any(c.lower() in text for c in include_countries):
                weak_geo += 1
        if weak_geo > len(rows) * 0.5:
            issues.append(CriticIssue("warning", "weak_geo_evidence", "Most accepted results have weak geography evidence.", "Require geography evidence in validation or enrich from contact/location pages."))
    return issues


def summarize_issues(issues: List[CriticIssue]) -> str:
    if not issues:
        return "No major critic issues."
    return " | ".join(f"[{i.severity}] {i.code}: {i.message}" for i in issues)
