from __future__ import annotations

from typing import Iterable, List

from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city, normalize_country_name


_SOFTWARE_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital",
    "ai", "artificial intelligence", "machine learning", "iot", "scada",
    "monitoring", "optimization", "optimisation", "cloud", "data platform",
]

_PARTNER_HINTS = [
    "partner", "partners", "partner program", "channel partner", "channel partners",
    "reseller", "resellers", "distributor", "distributors", "representative",
    "representation", "local representative", "regional representative",
    "sales partner", "authorized partner", "authorised partner",
]

_SOLUTION_SYNONYM_MAP = {
    "machine learning": ["machine learning", " ml "],
    "artificial intelligence": ["artificial intelligence"],
    "ai": [" ai "],
    "analytics": ["analytics", "analytic"],
    "monitoring": ["monitoring", "remote monitoring", "surveillance"],
    "optimization": ["optimization", "optimisation", "optimizer", "optimiser"],
    "automation": ["automation", "automated", "autonomous"],
    "iot": ["iot", "internet of things"],
    "scada": ["scada"],
    "digital twin": ["digital twin"],
    "predictive maintenance": ["predictive maintenance"],
}

_DOMAIN_SYNONYM_MAP = {
    "esp": [" esp ", "electrical submersible pump", "electric submersible pump"],
    "virtual flow metering": ["virtual flow metering", "virtual flow meter", "virtual meter"],
    "well performance": ["well performance"],
    "artificial lift": ["artificial lift"],
    "production optimization": ["production optimization", "production optimisation"],
    "well surveillance": ["well surveillance"],
    "multiphase metering": ["multiphase metering", "multiphase meter"],
    "flow assurance": ["flow assurance"],
    "production monitoring": ["production monitoring"],
    "reservoir simulation": ["reservoir simulation"],
    "reservoir modeling": ["reservoir modeling", "reservoir modelling"],
    "drilling optimization": ["drilling optimization", "drilling optimisation"],
    "production engineering": ["production engineering"],
}


def _norm_country(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    norm = normalize_country_name(raw)
    if norm:
        return norm
    raw_l = raw.lower().strip()
    aliases = {
        "usa": "united states",
        "u.s.a.": "united states",
        "u.s.": "united states",
        "us": "united states",
        "uk": "united kingdom",
        "uae": "united arab emirates",
    }
    return aliases.get(raw_l, raw_l)


def _presence_matches(presence_list: Iterable[str], candidates: Iterable[str]) -> bool:
    cand_norms = {_norm_country(c) for c in candidates if c}
    for item in presence_list or []:
        if _norm_country(item) in cand_norms:
            return True
    return False


def _text_blob(record: CompanyRecord) -> str:
    return " ".join([
        record.company_name or "",
        record.description or "",
        record.notes or "",
        record.website or "",
        record.source_url or "",
        getattr(record, "summary", "") or "",
        getattr(record, "contact_page", "") or "",
        getattr(record, "linkedin_url", "") or "",
    ]).lower()


def _keyword_hit_count(haystack: str, keywords: List[str], synonym_map: dict) -> int:
    if not keywords:
        return 0
    hits = 0
    seen = set()
    padded = f" {haystack} "
    for kw in keywords:
        key = str(kw).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        terms = synonym_map.get(key, [key])
        if any(term in padded or term in haystack for term in terms):
            hits += 1
    return hits


def _commercial_intent_bonus(haystack: str, commercial_intent: str) -> float:
    ci = (commercial_intent or "general").strip().lower()
    if ci == "general":
        return 0.0

    bonus = 0.0
    if any(term in haystack for term in _PARTNER_HINTS):
        bonus += 10.0

    if ci == "agent_or_distributor":
        if any(term in haystack for term in ["distribution", "distributor", "reseller", "channel", "partner"]):
            bonus += 6.0
    elif ci == "reseller":
        if any(term in haystack for term in ["reseller", "resellers", "channel", "partner program"]):
            bonus += 5.0
    elif ci == "partner":
        if any(term in haystack for term in ["partner", "partners", "partner program", "alliance"]):
            bonus += 5.0

    return bonus


def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0

    target_category = (getattr(spec, "target_category", "general") or "general").strip().lower()
    solution_keywords = [str(x).strip().lower() for x in (getattr(spec, "solution_keywords", []) or []) if str(x).strip()]
    domain_keywords = [str(x).strip().lower() for x in (getattr(spec, "domain_keywords", []) or []) if str(x).strip()]
    commercial_intent = (getattr(spec, "commercial_intent", "general") or "general").strip().lower()

    if record.company_name:
        score += 12
    if record.website:
        score += 16
    if record.domain:
        score += 6
    if record.description:
        desc_len = len(record.description)
        score += 10 if desc_len > 200 else (6 if desc_len > 50 else 3)

    if record.email:
        score += 12
    if record.phone:
        score += 8
    if record.contact_page:
        score += 6
    if record.linkedin_url:
        score += 5

    if not record.is_directory_or_media:
        score += 10
    if record.page_type == "company":
        score += 8
    elif record.page_type in {"directory", "media", "blog"}:
        score -= 15
    elif record.page_type == "document":
        score += 4

    haystack = _text_blob(record)

    sector_words = [w for w in (getattr(spec, "sector", "") or "").lower().split() if len(w) >= 2]
    sector_hits = sum(1 for w in sector_words if w in haystack)
    if sector_words:
        sector_ratio = sector_hits / len(sector_words)
        score += min(sector_ratio * 16, 16)
        if sector_hits == 0:
            score -= 35

    if target_category == "software_company":
        software_hits = sum(1 for h in _SOFTWARE_HINTS if h in haystack)
        if software_hits >= 3:
            score += 12
        elif software_hits >= 1:
            score += 6
        else:
            score -= 14

    if solution_keywords:
        solution_hits = _keyword_hit_count(haystack, solution_keywords, _SOLUTION_SYNONYM_MAP)
        solution_ratio = solution_hits / max(1, len(set(solution_keywords)))
        score += min(solution_ratio * 16, 16)
        if target_category == "software_company" and solution_hits == 0:
            score -= 10

    if domain_keywords:
        domain_hits = _keyword_hit_count(haystack, domain_keywords, _DOMAIN_SYNONYM_MAP)
        domain_ratio = domain_hits / max(1, len(set(domain_keywords)))
        score += min(domain_ratio * 18, 18)
        if domain_hits == 0:
            score -= 8

    for t in getattr(spec, "include_terms", []) or []:
        if t.strip().lower() in haystack:
            score += 4
    for t in getattr(spec, "exclude_terms", []) or []:
        if t.strip().lower() in haystack:
            score -= 6

    score += _commercial_intent_bonus(haystack, commercial_intent)

    rec_country = _norm_country(getattr(record, "country", None))
    rec_hq = _norm_country(getattr(record, "hq_country", None))
    presence = [_norm_country(c) for c in (getattr(record, "presence_countries", None) or []) if c]

    include_countries = [_norm_country(c) for c in (getattr(spec, "include_countries", None) or []) if c]
    exclude_countries = [_norm_country(c) for c in (getattr(spec, "exclude_countries", None) or []) if c]
    exclude_presence = [_norm_country(c) for c in (getattr(spec, "exclude_presence_countries", None) or []) if c]

    if include_countries:
        matched = False
        if rec_hq and rec_hq in include_countries:
            score += 18
            matched = True
        elif rec_country and rec_country in include_countries:
            score += 14
            matched = True
        elif any(c in include_countries for c in presence):
            score += 10
            matched = True
        elif any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 6
            matched = True
        if not matched:
            score -= 2

    if exclude_countries:
        if rec_hq and rec_hq in exclude_countries:
            score -= 35
        elif rec_country and rec_country in exclude_countries:
            score -= 28

    if exclude_presence:
        if _presence_matches(presence, exclude_presence):
            score -= 24
        if "united states" in exclude_presence and getattr(record, "has_usa_presence", False):
            score -= 24
        if "egypt" in exclude_presence and getattr(record, "has_egypt_presence", False):
            score -= 24
        for c in exclude_presence:
            if contains_country_or_city(haystack, c):
                score -= 4

    for field_name in (getattr(spec, "required_fields", None) or []):
        f = field_name.lower()
        if f == "website" and record.website:
            score += 3
        elif f == "email" and record.email:
            score += 5
        elif f == "phone" and record.phone:
            score += 4
        elif f == "linkedin" and record.linkedin_url:
            score += 2

    return round(max(0.0, min(100.0, score)), 2)


def score_records(records: List[CompanyRecord], spec: SearchSpec) -> List[CompanyRecord]:
    for rec in records:
        rec.confidence_score = score_company_record(rec, spec)
    return records
