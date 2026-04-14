from __future__ import annotations

from typing import List

from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city, normalize_country_name


def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0

    # --- base structure ---
    if record.company_name:
        score += 12
    if record.website:
        score += 16
    if record.domain:
        score += 6
    if record.description:
        desc_len = len(record.description)
        score += 10 if desc_len > 200 else (6 if desc_len > 50 else 3)

    # --- contact completeness (bonus only, not a gate) ---
    if record.email:
        score += 12
    if record.phone:
        score += 8
    if record.contact_page:
        score += 6
    if record.linkedin_url:
        score += 5

    # --- page quality ---
    if not record.is_directory_or_media:
        score += 10
    if record.page_type == "company":
        score += 8
    elif record.page_type in {"directory", "media", "blog"}:
        score -= 15
    elif record.page_type == "document":
        score += 4

    # --- relevance: sector keyword match ---
    haystack = " ".join([
        record.company_name or "",
        record.description  or "",
        record.notes        or "",
        record.website      or "",
        record.source_url   or "",
    ]).lower()

    sector_words = [w for w in (spec.sector or "").lower().split() if len(w) >= 2]
    sector_hits  = sum(1 for w in sector_words if w in haystack)
    if sector_words:
        sector_ratio = sector_hits / len(sector_words)
        score += min(sector_ratio * 16, 16)
        if sector_hits == 0:
            score -= 35  # completely off-topic

    # include/exclude term scoring
    for t in spec.include_terms or []:
        if t.strip().lower() in haystack:
            score += 4
    for t in spec.exclude_terms or []:
        if t.strip().lower() in haystack:
            score -= 6

    # --- geography scoring ---
    rec_country       = normalize_country_name(record.country)
    rec_hq            = normalize_country_name(record.hq_country)
    presence          = [normalize_country_name(c) for c in (record.presence_countries or [])]
    include_countries = [normalize_country_name(c) for c in (spec.include_countries or [])]
    exclude_countries = [normalize_country_name(c) for c in (spec.exclude_countries or [])]
    exclude_presence  = [normalize_country_name(c) for c in (spec.exclude_presence_countries or [])]

    if include_countries:
        matched = False
        if rec_hq and rec_hq in include_countries:
            score += 18; matched = True
        elif rec_country and rec_country in include_countries:
            score += 14; matched = True
        elif any(c in include_countries for c in presence):
            score += 10; matched = True
        elif any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 6; matched = True
        if not matched:
            score -= 2  # tiny penalty for unknown geo (NOT a hard reject)

    if exclude_countries:
        if rec_hq and rec_hq in exclude_countries:
            score -= 35
        elif rec_country and rec_country in exclude_countries:
            score -= 28

    if exclude_presence:
        if any(c in exclude_presence for c in presence):
            score -= 20
        if "usa" in exclude_presence and record.has_usa_presence:
            score -= 20
        if "egypt" in exclude_presence and record.has_egypt_presence:
            score -= 20

    # requested fields bonus
    for field_name in (spec.required_fields or []):
        f = field_name.lower()
        if   f == "website"  and record.website:        score += 3
        elif f == "email"    and record.email:          score += 5
        elif f == "phone"    and record.phone:          score += 4
        elif f == "linkedin" and record.linkedin_url:   score += 2

    return round(max(0.0, min(100.0, score)), 2)


def score_records(records: List[CompanyRecord], spec: SearchSpec) -> List[CompanyRecord]:
    for rec in records:
        rec.confidence_score = score_company_record(rec, spec)
    return records
