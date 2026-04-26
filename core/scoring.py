from __future__ import annotations
from typing import Iterable, List
from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city, normalize_country_name

_SOFTWARE_HINTS = ["software","platform","saas","analytics","automation","digital","ai","iot","scada","cloud","data"]
_FOOD_HINTS = ["food","food processing","food manufacturing","beverage","fmcg","bakery","dairy","meat","snack","confectionery"]
_EVENT_HINTS = ["exhibitor","egyps","conference","expo","trade show"]
_WIRELINE_HINTS = ["wireline","well logging","logging","slickline","cased hole","open hole"]

def _norm_country(v): 
    return normalize_country_name((v or "").strip()) or (v or "").strip().lower()

def _text_blob(record: CompanyRecord) -> str:
    return " ".join([
        record.company_name or "", record.description or "", record.notes or "",
        record.website or "", record.source_url or "", getattr(record, "summary", "") or "",
        getattr(record, "contact_page", "") or "", getattr(record, "linkedin_url", "") or "",
    ]).lower()

def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0
    hay = _text_blob(record)
    if record.company_name: score += 8
    if record.website: score += 10
    if record.email: score += 8
    if record.phone: score += 5
    if record.page_type == "company": score += 8
    if record.is_directory_or_media: score -= 25

    category = (getattr(spec, "target_category", "general") or "general").lower()
    sector = (getattr(spec, "sector", "") or "").lower()

    if category == "software_company":
        soft_hits = sum(1 for h in _SOFTWARE_HINTS if h in hay)
        score += min(soft_hits * 3, 12)
        if soft_hits == 0: score -= 15

    # strict vertical relevance for food manufacturing
    if sector == "food manufacturing":
        food_hits = sum(1 for h in _FOOD_HINTS if h in hay)
        if food_hits >= 1:
            score += 18
        else:
            score -= 35
        # penalize broad generic enterprise pages
        if any(x in hay for x in ["erp", "enterprise software", "digital transformation"]) and food_hits == 0:
            score -= 20

    # strict event relevance for EGYPS exhibitor searches
    dks = [str(x).lower() for x in (getattr(spec, "domain_keywords", []) or [])]
    if "wireline" in dks or "well logging" in dks:
        wire_hits = sum(1 for h in _WIRELINE_HINTS if h in hay)
        if wire_hits >= 1:
            score += 14
        else:
            score -= 25
        if getattr(spec, "intent_type", "") == "market_research" or "egyps" in sector:
            event_hits = sum(1 for h in _EVENT_HINTS if h in hay)
            if event_hits >= 1:
                score += 12
            else:
                score -= 15

    # geography
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
