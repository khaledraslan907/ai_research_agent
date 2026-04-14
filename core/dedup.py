from __future__ import annotations

from typing import List, Dict

from core.models import CompanyRecord
from core.normalizer import company_key, are_company_names_similar, prefer_best_company_name
from core.utils import extract_domain, clean_text, unique_list


def _merge(a: CompanyRecord, b: CompanyRecord) -> CompanyRecord:
    merged = CompanyRecord()
    merged.company_name     = prefer_best_company_name(a.company_name, b.company_name)
    merged.website          = a.website or b.website
    merged.domain           = a.domain or b.domain or extract_domain(merged.website)
    merged.description      = a.description if len(a.description) >= len(b.description) else b.description
    merged.country          = a.country or b.country
    merged.city             = a.city or b.city
    merged.hq_country       = a.hq_country or b.hq_country
    merged.presence_countries = unique_list((a.presence_countries or []) + (b.presence_countries or []))
    merged.has_usa_presence = a.has_usa_presence or b.has_usa_presence
    merged.has_egypt_presence = a.has_egypt_presence or b.has_egypt_presence
    merged.email            = a.email or b.email
    merged.phone            = a.phone or b.phone
    merged.linkedin_url     = a.linkedin_url or b.linkedin_url
    merged.contact_page     = a.contact_page or b.contact_page
    merged.source_url       = a.source_url or b.source_url
    merged.source_provider  = a.source_provider or b.source_provider
    merged.confidence_score = max(a.confidence_score, b.confidence_score)
    merged.page_type        = a.page_type or b.page_type
    merged.is_directory_or_media = a.is_directory_or_media and b.is_directory_or_media
    merged.matched_keywords = unique_list((a.matched_keywords or []) + (b.matched_keywords or []))
    merged.notes            = clean_text(f"{a.notes} | {b.notes}".strip(" |"))
    merged.raw_sources      = (a.raw_sources or []) + (b.raw_sources or [])
    return merged


def deduplicate_people(records: List[CompanyRecord]) -> List[CompanyRecord]:
    """
    Deduplicate LinkedIn person profiles.
    Use linkedin_url as primary key (multiple people share linkedin.com domain).
    Secondary: same name + same employer = duplicate.
    """
    if not records:
        return []
    seen_urls:  set = set()
    seen_names: dict = {}  # "name|employer" → index
    final: List[CompanyRecord] = []
    for rec in records:
        url = (rec.linkedin_url or rec.linkedin_profile or rec.website or "").lower().rstrip("/")
        if url and url in seen_urls:
            continue
        # Name+employer dedup (catches same person at slightly different URL)
        name_key = f"{(rec.company_name or '').lower()}|{(rec.employer_name or '').lower()}"
        if name_key and name_key != "|" and name_key in seen_names:
            continue
        if url:
            seen_urls.add(url)
        if name_key and name_key != "|":
            seen_names[name_key] = len(final)
        final.append(rec)
    return final


def deduplicate_companies(records: List[CompanyRecord]) -> List[CompanyRecord]:
    if not records:
        return []

    # For people search records, use URL-based dedup
    if records and records[0].page_type == "person":
        return deduplicate_people(records)

    # --- pass 1: merge by exact domain ---
    by_domain: Dict[str, CompanyRecord] = {}
    no_domain: List[CompanyRecord] = []

    for rec in records:
        rec.domain = rec.domain or extract_domain(rec.website)
        if rec.domain:
            if rec.domain in by_domain:
                by_domain[rec.domain] = _merge(by_domain[rec.domain], rec)
            else:
                by_domain[rec.domain] = rec
        else:
            no_domain.append(rec)

    candidates = list(by_domain.values()) + no_domain

    # --- pass 2: fuzzy name merge ---
    final: List[CompanyRecord] = []
    for rec in candidates:
        matched = False
        for i, existing in enumerate(final):
            k1 = company_key(rec.company_name, rec.website)
            k2 = company_key(existing.company_name, existing.website)
            same = (k1 and k2 and k1 == k2) or are_company_names_similar(
                rec.company_name, existing.company_name
            )
            if same:
                final[i] = _merge(existing, rec)
                matched = True
                break
        if not matched:
            final.append(rec)

    return final
