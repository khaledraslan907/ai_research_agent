from __future__ import annotations

from typing import List

from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city

_SOFTWARE_HINTS = ["software", "platform", "saas", "analytics", "automation", "digital", "mes", "quality management", "production planning"]
_FOOD_HINTS = ["food", "beverage", "food processing", "food manufacturing", "bakery", "dairy", "meat processing"]
_WIRELINE_HINTS = ["wireline", "slickline", "well logging", "open hole", "cased hole", "perforation"]
_EVENT_HINTS = ["egyps", "exhibitor", "booth", "conference", "event", "exhibit"]
_TENDER_HINTS = ["tender", "rfq", "rfp", "procurement", "invitation to tender", "bid notice"]
_OFFICIAL_HINTS = ["ministry", "authority", "operator", "government", "official"]
_BAD_NEWS_HINTS = ["bbc", "news", "article", "blog", "wikipedia"]

def _text_blob(record: CompanyRecord) -> str:
    return " ".join([record.company_name or "", record.description or "", record.notes or "", record.website or "", record.source_url or "", getattr(record, "summary", "") or "", getattr(record, "contact_page", "") or "", getattr(record, "linkedin_url", "") or ""]).lower()

def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0
    hay = _text_blob(record)
    if record.company_name: score += 8
    if record.website: score += 10
    if record.email: score += 8
    if record.phone: score += 5
    if record.page_type == "company": score += 8
    if record.is_directory_or_media: score -= 25
    if any(x in hay for x in _BAD_NEWS_HINTS): score -= 35

    category = (getattr(spec, "target_category", "general") or "general").lower()
    sector = (getattr(spec, "sector", "") or "").lower()
    entity_type = (getattr(spec, "entity_type", "company") or "company").lower()

    if category == "software_company":
        soft_hits = sum(1 for h in _SOFTWARE_HINTS if h in hay)
        score += min(soft_hits * 3, 12)
        if soft_hits == 0: score -= 15

    if sector == "food manufacturing":
        food_hits = sum(1 for h in _FOOD_HINTS if h in hay)
        if food_hits >= 1: score += 25
        else: score -= 45
        if any(x in hay for x in ["sap", "oracle", "microsoft", "enterprise software", "erp"]) and food_hits == 0:
            score -= 30

    dks = [str(x).lower() for x in (getattr(spec, "domain_keywords", []) or [])]
    if any(x in dks for x in ["wireline", "slickline", "well logging"]):
        wire_hits = sum(1 for h in _WIRELINE_HINTS if h in hay)
        if wire_hits >= 1: score += 18
        else: score -= 35
        include_countries = [str(c).lower() for c in (getattr(spec, "include_countries", []) or [])]
        if "egypt" in include_countries and not contains_country_or_city(hay, "egypt"):
            score -= 45

    if entity_type == "tender" or getattr(spec, "intent_type", "") == "market_research":
        tender_hits = sum(1 for h in _TENDER_HINTS if h in hay)
        if tender_hits >= 1: score += 18
        else: score -= 30
        official_hits = sum(1 for h in _OFFICIAL_HINTS if h in hay)
        score += min(official_hits * 5, 15)

    if "egyps" in sector or (any(x in dks for x in ["wireline", "well logging"]) and entity_type == "company"):
        event_hits = sum(1 for h in _EVENT_HINTS if h in hay)
        if event_hits >= 1: score += 15

    include_countries = [str(c).lower() for c in (getattr(spec, "include_countries", []) or [])]
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
