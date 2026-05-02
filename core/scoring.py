from __future__ import annotations

from typing import Iterable, List

from core.models import CompanyRecord, SearchSpec
from core.geography import contains_country_or_city, normalize_country_name


_SOFTWARE_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital",
    "ai", "artificial intelligence", "machine learning", "iot", "scada",
    "monitoring", "optimization", "optimisation", "cloud", "data platform",
    "mes", "quality management", "production planning",
]

_SERVICE_HINTS = [
    "service", "services", "contractor", "engineering", "intervention",
    "wireline", "slickline", "logging", "drilling", "completion", "oilfield",
    "well services", "open hole", "cased hole", "perforation",
]

_FOOD_HINTS = [
    "food", "food processing", "food manufacturing", "food industry",
    "beverage", "bakery", "dairy", "meat processing", "factory",
]

_WIRELINE_HINTS = [
    "wireline", "slickline", "well logging", "open hole", "cased hole",
    "perforation", "e-line", "electric line",
]

_EVENT_HINTS = [
    "egyps", "exhibitor", "booth", "conference", "event", "exhibit",
]

_TENDER_HINTS = [
    "tender", "rfq", "rfp", "procurement", "invitation to tender", "bid notice",
    "deadline", "closing date", "buyer", "tender title",
]

_ACADEMIC_HINTS = [
    "abstract", "doi", "journal", "conference", "paper", "publication",
    "published", "peer-reviewed", "authors", "author", "research", "study",
    "onepetro", "spe", "ieee", "sciencedirect", "springer", "elsevier",
    "conference proceedings", "preprint", "manuscript",
]

_COMPANY_CASE_STUDY_HINTS = [
    "case study", "brochure", "product page", "services", "solutions",
    "halliburton", "schlumberger", "baker hughes", "weatherford",
]

_JUNK_HINTS = [
    "job opportunities", "apply today", "wuzzuf", "courier", "shipping services",
    "ship services", "amazon", "marketplace", "news", "article", "blog",
    "directory", "list of", "top companies", "business directory",
]

_PARTNER_HINTS = [
    "partner", "partners", "partner program", "channel partner", "channel partners",
    "reseller", "resellers", "distributor", "distributors", "representative",
    "representation", "local representative", "regional representative",
    "sales partner", "authorized partner", "authorised partner",
]

_SOLUTION_SYNONYM_MAP = {
    "machine learning": ["machine learning", " ml "],
    "artificial intelligence": ["artificial intelligence", "agentic ai", "agentic"],
    "ai": [" ai ", "artificial intelligence", "agentic ai", "agentic"],
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
    "electrical submersible pump": ["electrical submersible pump", "electric submersible pump", " esp "],
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
    "wireline": ["wireline", "e-line", "electric line"],
    "slickline": ["slickline"],
    "well logging": ["well logging", "open hole", "cased hole", "logging"],
}

_ACADEMIC_DOMAINS = [
    "doi.org", "onepetro.org", "ieeexplore.ieee.org", "sciencedirect.com",
    "springer.com", "link.springer.com", "researchgate.net", "mdpi.com",
    "frontiersin.org", "arxiv.org", "osti.gov", "scholar.google",
]


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
        getattr(record, "authors", "") or "",
        getattr(record, "doi", "") or "",
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


def _count_hits(haystack: str, hints: List[str]) -> int:
    return sum(1 for h in hints if h in haystack)


def _sector_words(spec: SearchSpec) -> List[str]:
    text = (getattr(spec, "sector", "") or "").lower().replace("-", " ")
    return [w for w in text.split() if len(w) >= 3 and w not in {"and", "for", "the"}]


def _score_academic_record(record: CompanyRecord, spec: SearchSpec, haystack: str) -> float:
    score = 0.0
    score += 14 if record.website else 0
    score += 8 if record.description else 0
    score += 10 if getattr(record, "authors", "") else 0
    score += 12 if getattr(record, "doi", "") else 0
    score += 6 if getattr(record, "publication_year", "") else 0
    if record.page_type == "document":
        score += 16
    elif record.page_type == "company":
        score -= 12

    academic_hits = _count_hits(haystack, _ACADEMIC_HINTS)
    score += min(academic_hits * 5, 30)

    if any(d in haystack for d in _ACADEMIC_DOMAINS):
        score += 18

    sector_hits = sum(1 for w in _sector_words(spec) if w in haystack)
    score += min(sector_hits * 6, 18)

    solution_keywords = [str(x).strip().lower() for x in (getattr(spec, "solution_keywords", []) or []) if str(x).strip()]
    domain_keywords = [str(x).strip().lower() for x in (getattr(spec, "domain_keywords", []) or []) if str(x).strip()]
    solution_hits = _keyword_hit_count(haystack, solution_keywords, _SOLUTION_SYNONYM_MAP)
    domain_hits = _keyword_hit_count(haystack, domain_keywords, _DOMAIN_SYNONYM_MAP)
    score += min(solution_hits * 8, 16)
    score += min(domain_hits * 9, 18)

    if academic_hits == 0 and not getattr(record, "doi", "") and not getattr(record, "authors", ""):
        score -= 35

    if any(h in haystack for h in _COMPANY_CASE_STUDY_HINTS):
        score -= 24
    if any(h in haystack for h in _JUNK_HINTS):
        score -= 28

    return score


def _score_service_company(record: CompanyRecord, spec: SearchSpec, haystack: str) -> float:
    score = 0.0
    score += 12 if record.company_name else 0
    score += 14 if record.website else 0
    score += 10 if record.email else 0
    score += 7 if record.phone else 0
    score += 6 if record.contact_page else 0
    if record.page_type == "company":
        score += 10
    if record.is_directory_or_media or record.page_type in {"directory", "media", "blog"}:
        score -= 25

    service_hits = _count_hits(haystack, _SERVICE_HINTS)
    score += min(service_hits * 4, 20)

    solution_keywords = [str(x).strip().lower() for x in (getattr(spec, "solution_keywords", []) or []) if str(x).strip()]
    domain_keywords = [str(x).strip().lower() for x in (getattr(spec, "domain_keywords", []) or []) if str(x).strip()]
    domain_hits = _keyword_hit_count(haystack, domain_keywords, _DOMAIN_SYNONYM_MAP)
    solution_hits = _keyword_hit_count(haystack, solution_keywords, _SOLUTION_SYNONYM_MAP)
    score += min(domain_hits * 9, 22)
    score += min(solution_hits * 6, 12)

    if any(k in domain_keywords for k in ["wireline", "slickline", "well logging"]):
        wire_hits = _count_hits(haystack, _WIRELINE_HINTS)
        if wire_hits:
            score += min(wire_hits * 7, 18)
        else:
            score -= 38

    include_countries = [_norm_country(c) for c in (getattr(spec, "include_countries", None) or []) if c]
    rec_country = _norm_country(getattr(record, "country", None))
    rec_hq = _norm_country(getattr(record, "hq_country", None))
    presence = [_norm_country(c) for c in (getattr(record, "presence_countries", None) or []) if c]

    if include_countries:
        if rec_hq and rec_hq in include_countries:
            score += 18
        elif rec_country and rec_country in include_countries:
            score += 15
        elif any(c in include_countries for c in presence):
            score += 12
        elif any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 8
        else:
            score -= 22

    if "egypt" in include_countries and not (
        rec_hq == "egypt" or rec_country == "egypt" or "egypt" in presence or contains_country_or_city(haystack, "egypt")
    ):
        score -= 24

    if any(h in haystack for h in _JUNK_HINTS + ["ship services", "courier", "job vacancies", "recruitment"]):
        score -= 35

    return score


def _score_software_company(record: CompanyRecord, spec: SearchSpec, haystack: str) -> float:
    score = 0.0
    score += 12 if record.company_name else 0
    score += 15 if record.website else 0
    score += 8 if record.email else 0
    score += 4 if record.linkedin_url else 0
    if record.page_type == "company":
        score += 8
    if record.is_directory_or_media:
        score -= 20

    software_hits = _count_hits(haystack, _SOFTWARE_HINTS)
    if software_hits:
        score += min(software_hits * 4, 20)
    else:
        score -= 18

    sector = (getattr(spec, "sector", "") or "").lower().strip()
    if sector == "food manufacturing":
        food_hits = _count_hits(haystack, _FOOD_HINTS)
        if food_hits:
            score += min(food_hits * 8, 24)
        else:
            score -= 42
        if any(x in haystack for x in ["sap", "oracle", "microsoft", "erp"]) and food_hits == 0:
            score -= 24

    solution_keywords = [str(x).strip().lower() for x in (getattr(spec, "solution_keywords", []) or []) if str(x).strip()]
    domain_keywords = [str(x).strip().lower() for x in (getattr(spec, "domain_keywords", []) or []) if str(x).strip()]
    solution_hits = _keyword_hit_count(haystack, solution_keywords, _SOLUTION_SYNONYM_MAP)
    domain_hits = _keyword_hit_count(haystack, domain_keywords, _DOMAIN_SYNONYM_MAP)
    score += min(solution_hits * 7, 18)
    score += min(domain_hits * 7, 18)

    include_countries = [_norm_country(c) for c in (getattr(spec, "include_countries", None) or []) if c]
    rec_country = _norm_country(getattr(record, "country", None))
    rec_hq = _norm_country(getattr(record, "hq_country", None))
    presence = [_norm_country(c) for c in (getattr(record, "presence_countries", None) or []) if c]
    if include_countries:
        if rec_hq and rec_hq in include_countries:
            score += 14
        elif rec_country and rec_country in include_countries:
            score += 10
        elif any(c in include_countries for c in presence):
            score += 8
        elif any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 5
        else:
            score -= 10

    return score


def _score_market_research(record: CompanyRecord, spec: SearchSpec, haystack: str) -> float:
    score = 0.0
    score += 10 if record.company_name else 0
    score += 10 if record.website else 0
    if record.page_type == "company":
        score += 6
    if record.page_type == "document":
        score += 4

    tender_hits = _count_hits(haystack, _TENDER_HINTS)
    event_hits = _count_hits(haystack, _EVENT_HINTS)
    score += min(tender_hits * 6, 18)
    score += min(event_hits * 5, 15)

    if any(k in (getattr(spec, "domain_keywords", []) or []) for k in ["wireline", "well logging"]):
        wire_hits = _count_hits(haystack, _WIRELINE_HINTS)
        score += min(wire_hits * 7, 18) if wire_hits else -18

    include_countries = [_norm_country(c) for c in (getattr(spec, "include_countries", None) or []) if c]
    if include_countries:
        if any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 8
        else:
            score -= 10

    return score


def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    haystack = _text_blob(record)
    target_category = (getattr(spec, "target_category", "general") or "general").strip().lower()
    intent_type = (getattr(spec, "intent_type", "general") or "general").strip().lower()
    entity_type = (getattr(spec, "entity_type", "company") or "company").strip().lower()

    if intent_type == "document_research" or entity_type == "paper":
        score = _score_academic_record(record, spec, haystack)
    elif intent_type == "market_research" or entity_type == "tender":
        score = _score_market_research(record, spec, haystack)
    elif target_category == "software_company":
        score = _score_software_company(record, spec, haystack)
    elif target_category == "service_company":
        score = _score_service_company(record, spec, haystack)
    else:
        score = 0.0
        score += 10 if record.company_name else 0
        score += 12 if record.website else 0
        score += 8 if record.email else 0
        score += 4 if record.phone else 0
        if record.page_type == "company":
            score += 8
        if record.is_directory_or_media:
            score -= 18
        sector_hits = sum(1 for w in _sector_words(spec) if w in haystack)
        score += min(sector_hits * 5, 15)

    # Generic geography and exclusions
    rec_country = _norm_country(getattr(record, "country", None))
    rec_hq = _norm_country(getattr(record, "hq_country", None))
    presence = [_norm_country(c) for c in (getattr(record, "presence_countries", None) or []) if c]

    include_countries = [_norm_country(c) for c in (getattr(spec, "include_countries", None) or []) if c]
    exclude_countries = [_norm_country(c) for c in (getattr(spec, "exclude_countries", None) or []) if c]
    exclude_presence = [_norm_country(c) for c in (getattr(spec, "exclude_presence_countries", None) or []) if c]

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

    score += _commercial_intent_bonus(haystack, getattr(spec, "commercial_intent", "general"))
    return round(max(0.0, min(100.0, score)), 2)


def score_records(records: List[CompanyRecord], spec: SearchSpec) -> List[CompanyRecord]:
    for rec in records:
        rec.confidence_score = score_company_record(rec, spec)
    return records
