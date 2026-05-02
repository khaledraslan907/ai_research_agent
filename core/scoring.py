from __future__ import annotations

from typing import List

from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city

_SOFTWARE_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital", "mes",
    "quality management", "production planning", "ai", "artificial intelligence",
    "machine learning", "data platform", "cloud", "iot", "scada", "optimization",
    "optimisation", "monitoring", "digital twin",
]
_FOOD_HINTS = ["food", "beverage", "food processing", "food manufacturing", "bakery", "dairy", "meat processing"]
_WIRELINE_HINTS = ["wireline", "slickline", "well logging", "open hole", "cased hole", "perforation"]
_EVENT_HINTS = ["egyps", "exhibitor", "booth", "conference", "event", "exhibit"]
_TENDER_HINTS = ["tender", "rfq", "rfp", "procurement", "invitation to tender", "bid notice"]
_OFFICIAL_HINTS = ["ministry", "authority", "operator", "government", "official"]
_BAD_NEWS_HINTS = [
    "bbc", "news", "article", "blog", "wikipedia", "job", "jobs", "vacancy",
    "courier", "shipping", "stock photo", "directory", "list of companies", "ranking",
]
_OIL_GAS_HINTS = [
    "oil and gas", "oil & gas", "petroleum", "upstream", "midstream", "downstream",
    "energy services", "offshore", "subsurface", "reservoir", "drilling", "production",
]
_NORWAY_HINTS = ["norway", "norwegian", "oslo", "stavanger", "bergen", "trondheim"]
_GENERIC_ENTERPRISE_HINTS = ["sap", "oracle", "microsoft", "erp", "crm"]
_ACADEMIC_HINTS = [
    "abstract", "doi", "journal", "conference", "paper", "research", "study",
    "authors", "published", "publication", "proceedings", "manuscript", "scholar",
]
_NON_ACADEMIC_DOC_HINTS = [
    "case study", "services", "product page", "solutions", "our capabilities",
    "halliburton", "slb", "weatherford", "baker hughes", "request a demo", "contact us",
]


def _text_blob(record: CompanyRecord) -> str:
    return " ".join(
        [
            record.company_name or "",
            record.description or "",
            record.notes or "",
            record.website or "",
            record.source_url or "",
            getattr(record, "summary", "") or "",
            getattr(record, "contact_page", "") or "",
            getattr(record, "linkedin_url", "") or "",
            getattr(record, "authors", "") or "",
            getattr(record, "doi", "") or "",
            getattr(record, "job_title", "") or "",
            getattr(record, "employer_name", "") or "",
        ]
    ).lower()


def _count_hits(hay: str, hints: list[str]) -> int:
    return sum(1 for h in hints if h in hay)


def _has_any(hay: str, hints: list[str]) -> bool:
    return any(h in hay for h in hints)


def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0
    hay = _text_blob(record)

    if record.company_name:
        score += 8
    if record.website:
        score += 10
    if record.email:
        score += 8
    if record.phone:
        score += 5
    if getattr(record, "page_type", "") == "company":
        score += 8
    if getattr(record, "page_type", "") == "document":
        score += 8
    if record.is_directory_or_media:
        score -= 25
    if _has_any(hay, _BAD_NEWS_HINTS):
        score -= 35

    category = (getattr(spec, "target_category", "general") or "general").lower()
    sector = (getattr(spec, "sector", "") or "").lower()
    entity_type = (getattr(spec, "entity_type", "company") or "company").lower()
    dks = [str(x).lower() for x in (getattr(spec, "domain_keywords", []) or [])]
    sks = [str(x).lower() for x in (getattr(spec, "solution_keywords", []) or [])]
    include_countries = [str(c).lower() for c in (getattr(spec, "include_countries", []) or [])]

    # Digital / software scoring
    if category == "software_company":
        soft_hits = _count_hits(hay, _SOFTWARE_HINTS)
        score += min(soft_hits * 4, 20)
        if soft_hits == 0:
            score -= 25

        # penalize generic company/service pages that don't look digital
        if soft_hits < 2 and _has_any(hay, ["drilling services", "ship services", "courier", "recruitment"]):
            score -= 35

    # Oil and gas context
    if "oil and gas" in sector or _has_any(hay, _OIL_GAS_HINTS):
        oil_hits = _count_hits(hay, _OIL_GAS_HINTS)
        score += min(oil_hits * 2, 12)

    # Norway-specific digital oil and gas companies
    if category == "software_company" and sector == "oil and gas" and "norway" in include_countries:
        norway_hits = _count_hits(hay, _NORWAY_HINTS)
        if norway_hits >= 1 or contains_country_or_city(hay, "norway"):
            score += 18
        else:
            score -= 30

        soft_hits = _count_hits(hay, _SOFTWARE_HINTS)
        oil_hits = _count_hits(hay, _OIL_GAS_HINTS)
        if soft_hits >= 2 and oil_hits >= 1:
            score += 18
        else:
            score -= 28

        # strong penalty for generic ERP / enterprise pages with no O&G/norway evidence
        if _has_any(hay, _GENERIC_ENTERPRISE_HINTS) and (oil_hits == 0 or norway_hits == 0):
            score -= 30

    # Food manufacturing software
    if category == "software_company" and sector == "food manufacturing":
        food_hits = _count_hits(hay, _FOOD_HINTS)
        if food_hits >= 1:
            score += 25
        else:
            score -= 45
        if _has_any(hay, _GENERIC_ENTERPRISE_HINTS) and food_hits == 0:
            score -= 30

    # Egypt wireline / logging
    if any(x in dks for x in ["wireline", "slickline", "well logging"]):
        wire_hits = _count_hits(hay, _WIRELINE_HINTS)
        if wire_hits >= 1:
            score += 18
        else:
            score -= 35
        if "egypt" in include_countries and not contains_country_or_city(hay, "egypt"):
            score -= 45

    # Tenders
    if entity_type == "tender" or getattr(spec, "intent_type", "") == "market_research":
        tender_hits = _count_hits(hay, _TENDER_HINTS)
        if tender_hits >= 1:
            score += 18
        else:
            score -= 30
        official_hits = _count_hits(hay, _OFFICIAL_HINTS)
        score += min(official_hits * 5, 15)

    # EGYPS exhibitor style searches
    if "egyps" in sector or (_has_any(" ".join(dks), ["wireline", "well logging"]) and entity_type == "company"):
        event_hits = _count_hits(hay, _EVENT_HINTS)
        if event_hits >= 1:
            score += 15

    # Academic document scoring
    if entity_type == "paper" or getattr(spec, "intent_type", "") == "document_research":
        acad_hits = _count_hits(hay, _ACADEMIC_HINTS)
        score += min(acad_hits * 5, 28)
        if getattr(record, "doi", ""):
            score += 14
        if getattr(record, "authors", ""):
            score += 12
        if getattr(record, "publication_year", ""):
            score += 5
        if _has_any(hay, _NON_ACADEMIC_DOC_HINTS):
            score -= 40
        if acad_hits == 0:
            score -= 35

    # Geography bonus/penalty
    if include_countries:
        if any(contains_country_or_city(hay, c) for c in include_countries):
            score += 10
        elif len(include_countries) <= 2:
            score -= 18

    return max(0.0, min(100.0, score))


def score_records(records: List[CompanyRecord], spec: SearchSpec) -> List[CompanyRecord]:
    for rec in records:
        rec.confidence_score = score_company_record(rec, spec)
    return records
