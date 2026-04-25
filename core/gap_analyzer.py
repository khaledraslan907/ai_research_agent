from __future__ import annotations

"""Simple result gap analysis for the research agent."""

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List


@dataclass
class GapReport:
    total_results: int = 0
    by_hq_country: Dict[str, int] = field(default_factory=dict)
    by_source_provider: Dict[str, int] = field(default_factory=dict)
    by_page_type: Dict[str, int] = field(default_factory=dict)
    by_domain_keyword: Dict[str, int] = field(default_factory=dict)
    missing_contact_count: int = 0
    missing_linkedin_count: int = 0
    missing_geo_evidence_count: int = 0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_result_gaps(records: Iterable, task_spec=None) -> GapReport:
    rows = list(records or [])
    report = GapReport(total_results=len(rows))

    hq = Counter()
    providers = Counter()
    page_types = Counter()
    domain_keywords = Counter()

    include_countries = list(getattr(getattr(task_spec, "geography", None), "include_countries", []) or []) if task_spec else []

    for rec in rows:
        if getattr(rec, "hq_country", ""):
            hq[str(getattr(rec, "hq_country", "")).strip().lower()] += 1
        if getattr(rec, "source_provider", ""):
            providers[str(getattr(rec, "source_provider", "")).strip().lower()] += 1
        if getattr(rec, "page_type", ""):
            page_types[str(getattr(rec, "page_type", "")).strip().lower()] += 1
        for kw in (getattr(rec, "matched_keywords", []) or []):
            domain_keywords[str(kw).strip().lower()] += 1
        if not getattr(rec, "email", "") and not getattr(rec, "phone", ""):
            report.missing_contact_count += 1
        if not getattr(rec, "linkedin_url", "") and not getattr(rec, "linkedin_profile", ""):
            report.missing_linkedin_count += 1
        if include_countries:
            text = " ".join([
                str(getattr(rec, "hq_country", "") or ""),
                " ".join(getattr(rec, "presence_countries", []) or []),
                str(getattr(rec, "description", "") or ""),
                str(getattr(rec, "notes", "") or ""),
            ]).lower()
            if not any(c.lower() in text for c in include_countries):
                report.missing_geo_evidence_count += 1

    report.by_hq_country = dict(hq.most_common())
    report.by_source_provider = dict(providers.most_common())
    report.by_page_type = dict(page_types.most_common())
    report.by_domain_keyword = dict(domain_keywords.most_common())

    if report.total_results == 0:
        report.recommendations.append("No accepted results yet — widen queries, lower minimum score, or add more providers.")
        return report

    if report.missing_contact_count > report.total_results * 0.5:
        report.recommendations.append("More than half the results lack email/phone — enrich with contact pages and structured data extraction.")
    if report.missing_geo_evidence_count > report.total_results * 0.5:
        report.recommendations.append("Most results lack strong geography evidence — require contact/location evidence before acceptance.")
    if len(report.by_hq_country) <= 1 and report.total_results >= 10:
        report.recommendations.append("Results are geographically concentrated — run broader country/city variants to improve diversity.")
    if len(report.by_source_provider) <= 1:
        report.recommendations.append("Results came from very few providers — enable more providers for better recall.")
    if task_spec and getattr(task_spec, "domain_keywords", None):
        missing_requested = [kw for kw in task_spec.domain_keywords if str(kw).strip().lower() not in report.by_domain_keyword]
        if missing_requested:
            report.recommendations.append(
                f"Requested niche terms are underrepresented: {', '.join(missing_requested[:6])}. Add dedicated domain query families."
            )
    return report
