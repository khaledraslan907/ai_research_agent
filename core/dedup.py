from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord
from core.normalizer import are_company_names_similar, company_key, prefer_best_company_name
from core.utils import clean_text, extract_domain, unique_list


def _canonical_domain(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    host = extract_domain(raw) or raw
    host = host.strip().lower()
    host = re.sub(r"^https?://", "", host)
    host = re.sub(r"^www\d*\.", "", host)
    host = host.split("/")[0].split("?")[0].split("#")[0].strip().rstrip(".")
    return host


def _canonical_url(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"^http://", "https://", raw)
    raw = re.sub(r"^https://www\d*\.", "https://", raw)
    return raw.rstrip("/")


def _norm_name(value: str) -> str:
    s = (value or "").lower().strip()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    noise = {
        "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "group", "holding",
        "holdings", "international", "global", "solutions", "solution", "technologies", "technology",
        "systems", "system", "services", "service", "official", "homepage", "home", "version",
    }
    words = [w for w in s.split() if w not in noise]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _record_quality_score(rec: CompanyRecord) -> int:
    score = 0
    if rec.company_name:
        score += 8
    if rec.website:
        score += 10
    if rec.domain:
        score += 8
    if rec.description:
        score += min(len(rec.description) // 40, 12)
    if rec.email:
        score += 6
    if rec.phone:
        score += 5
    if rec.linkedin_url:
        score += 4
    if rec.contact_page:
        score += 3
    if rec.hq_country:
        score += 4
    if rec.country:
        score += 2
    if rec.presence_countries:
        score += 3
    if rec.page_type == "company":
        score += 5
    elif rec.page_type in {"directory", "media", "blog"}:
        score -= 8
    score += int(rec.confidence_score or 0) // 5
    return score


def _pick_better_record(a: CompanyRecord, b: CompanyRecord) -> tuple[CompanyRecord, CompanyRecord]:
    return (b, a) if _record_quality_score(b) > _record_quality_score(a) else (a, b)


def _best_url(a: str, b: str) -> str:
    ca, cb = _canonical_url(a), _canonical_url(b)
    if ca and cb:
        return a if len(ca) <= len(cb) else b
    return a or b


def _best_domain(a: str, b: str, fallback_url_a: str = "", fallback_url_b: str = "") -> str:
    da = _canonical_domain(a) or _canonical_domain(fallback_url_a)
    db = _canonical_domain(b) or _canonical_domain(fallback_url_b)
    if da and db:
        return da if len(da) <= len(db) else db
    return da or db


def _merge(a: CompanyRecord, b: CompanyRecord) -> CompanyRecord:
    primary, secondary = _pick_better_record(a, b)
    merged = CompanyRecord()
    merged.company_name = prefer_best_company_name(a.company_name, b.company_name)
    merged.website = _best_url(primary.website, secondary.website)
    merged.domain = _best_domain(primary.domain, secondary.domain, primary.website, secondary.website)
    merged.description = a.description if len(a.description or "") >= len(b.description or "") else b.description
    merged.country = primary.country or secondary.country
    merged.city = primary.city or secondary.city
    merged.hq_country = primary.hq_country or secondary.hq_country
    merged.presence_countries = unique_list((a.presence_countries or []) + (b.presence_countries or []))
    merged.has_usa_presence = a.has_usa_presence or b.has_usa_presence
    merged.has_egypt_presence = a.has_egypt_presence or b.has_egypt_presence
    merged.email = primary.email or secondary.email
    merged.phone = primary.phone or secondary.phone
    merged.linkedin_url = primary.linkedin_url or secondary.linkedin_url
    merged.contact_page = primary.contact_page or secondary.contact_page
    merged.source_url = primary.source_url or secondary.source_url
    merged.source_provider = primary.source_provider or secondary.source_provider
    merged.confidence_score = max(a.confidence_score, b.confidence_score)
    merged.page_type = "company" if "company" in {a.page_type, b.page_type} else (primary.page_type or secondary.page_type)
    merged.is_directory_or_media = a.is_directory_or_media and b.is_directory_or_media
    merged.matched_keywords = unique_list((a.matched_keywords or []) + (b.matched_keywords or []))
    merged.notes = clean_text(f"{a.notes} | {b.notes}".strip(" |"))
    merged.raw_sources = (a.raw_sources or []) + (b.raw_sources or [])
    merged.authors = primary.authors or secondary.authors
    merged.doi = primary.doi or secondary.doi
    merged.publication_year = primary.publication_year or secondary.publication_year
    merged.job_title = primary.job_title or secondary.job_title
    merged.seniority = primary.seniority or secondary.seniority
    merged.department = primary.department or secondary.department
    merged.employer_name = primary.employer_name or secondary.employer_name
    merged.linkedin_profile = primary.linkedin_profile or secondary.linkedin_profile
    return merged


def deduplicate_people(records: List[CompanyRecord]) -> List[CompanyRecord]:
    if not records:
        return []
    seen_urls: set = set()
    seen_names: dict = {}
    final: List[CompanyRecord] = []
    for rec in records:
        url = _canonical_url(rec.linkedin_url or rec.linkedin_profile or rec.website or "")
        if url and url in seen_urls:
            continue
        name_key = f"{_norm_name(rec.company_name)}|{_norm_name(rec.employer_name)}"
        if name_key and name_key != "|" and name_key in seen_names:
            idx = seen_names[name_key]
            final[idx] = _merge(final[idx], rec)
            if url:
                seen_urls.add(url)
            continue
        if url:
            seen_urls.add(url)
        if name_key and name_key != "|":
            seen_names[name_key] = len(final)
        final.append(rec)
    return final


def _same_company_by_name_and_domain(rec: CompanyRecord, existing: CompanyRecord) -> bool:
    d1 = _canonical_domain(rec.domain or rec.website)
    d2 = _canonical_domain(existing.domain or existing.website)
    if d1 and d2 and d1 == d2:
        return True
    k1 = company_key(rec.company_name, rec.website)
    k2 = company_key(existing.company_name, existing.website)
    if k1 and k2 and k1 == k2:
        if d1 and d2 and d1 != d2:
            return False
        return True
    if are_company_names_similar(rec.company_name, existing.company_name):
        if d1 and d2 and d1 != d2:
            return False
        c1 = (rec.hq_country or rec.country or "").strip().lower()
        c2 = (existing.hq_country or existing.country or "").strip().lower()
        if c1 and c2 and c1 != c2:
            return False
        return True
    return False


def deduplicate_companies(records: List[CompanyRecord]) -> List[CompanyRecord]:
    if not records:
        return []
    if records and all((r.page_type == "person" or r.linkedin_profile or r.linkedin_url) for r in records):
        return deduplicate_people(records)

    by_domain: Dict[str, CompanyRecord] = {}
    no_domain: List[CompanyRecord] = []
    for rec in records:
        rec.domain = _canonical_domain(rec.domain or rec.website)
        if rec.domain:
            if rec.domain in by_domain:
                by_domain[rec.domain] = _merge(by_domain[rec.domain], rec)
            else:
                by_domain[rec.domain] = rec
        else:
            no_domain.append(rec)
    candidates = list(by_domain.values()) + no_domain

    by_name_key: Dict[str, CompanyRecord] = {}
    leftovers: List[CompanyRecord] = []
    for rec in candidates:
        nk = _norm_name(rec.company_name)
        if not nk:
            leftovers.append(rec)
            continue
        if nk in by_name_key:
            existing = by_name_key[nk]
            if _same_company_by_name_and_domain(rec, existing):
                by_name_key[nk] = _merge(existing, rec)
            else:
                leftovers.append(rec)
        else:
            by_name_key[nk] = rec

    merged = list(by_name_key.values()) + leftovers
    final: List[CompanyRecord] = []
    for rec in merged:
        matched = False
        for i, existing in enumerate(final):
            if _same_company_by_name_and_domain(rec, existing):
                final[i] = _merge(existing, rec)
                matched = True
                break
        if not matched:
            final.append(rec)
    return final
