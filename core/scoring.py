from __future__ import annotations

from typing import Dict, Iterable, List

from core.geography import contains_country_or_city, normalize_country_name
from core.models import CompanyRecord, SearchSpec


_SOFTWARE_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital",
    "ai", "artificial intelligence", "machine learning", "iot", "scada",
    "monitoring", "optimization", "optimisation", "cloud", "data platform",
    "digital twin", "predictive maintenance",
]

_SERVICE_HINTS = [
    "service", "services", "contractor", "contractors", "engineering",
    "oilfield services", "petroleum services", "wireline", "slickline",
    "well logging", "well intervention", "coiled tubing", "stimulation",
    "cementing", "mud logging", "well testing", "inspection", "ndt",
    "pipeline", "integrity", "instrumentation", "automation",
    "maintenance", "offshore", "marine services", "diving services",
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
    "ai": [" ai ", "artificial intelligence"],
    "analytics": ["analytics", "analytic", "insights"],
    "monitoring": ["monitoring", "remote monitoring", "surveillance"],
    "optimization": ["optimization", "optimisation", "optimizer", "optimiser"],
    "automation": ["automation", "automated", "autonomous"],
    "iot": ["iot", "internet of things"],
    "scada": ["scada"],
    "digital twin": ["digital twin"],
    "predictive maintenance": ["predictive maintenance"],
}

_DOMAIN_SYNONYM_MAP = {
    "wireline": ["wireline", "wire line"],
    "slickline": ["slickline", "slick line"],
    "e-line": ["e-line", "eline", "electric line"],
    "well logging": ["well logging", "well log"],
    "open hole logging": ["open hole logging", "open-hole logging"],
    "cased hole logging": ["cased hole logging", "cased-hole logging"],
    "perforation": ["perforation", "perforating"],
    "memory gauge": ["memory gauge", "memory gauges"],
    "well intervention": ["well intervention"],
    "completion": ["completion services", "well completion"],
    "coiled tubing": ["coiled tubing", "coiled-tubing"],
    "well testing": ["well testing", "well test"],
    "mud logging": ["mud logging"],
    "drilling fluids": ["drilling fluids", "mud engineering"],
    "cementing": ["cementing", "cementation"],
    "fishing": ["fishing services", "downhole fishing"],
    "downhole tools": ["downhole tools", "downhole tool"],
    "well stimulation": ["well stimulation", "stimulation services"],
    "acidizing": ["acidizing", "acidising", "acid stimulation"],
    "fracturing": ["fracturing", "fracking", "hydraulic fracturing"],
    "pipeline inspection": ["pipeline inspection"],
    "ndt": ["ndt", "non-destructive testing", "non destructive testing"],
    "asset integrity": ["asset integrity", "integrity management"],
    "corrosion monitoring": ["corrosion monitoring"],
    "instrumentation": ["instrumentation"],
    "automation": ["industrial automation", "process automation", "automation"],
    "scada": ["scada"],
    "process control": ["process control"],
    "offshore": ["offshore"],
    "marine services": ["marine services", "marine service"],
    "diving services": ["diving services", "diving service"],
    "rov": ["rov", "remotely operated vehicle"],
    "oilfield chemicals": ["oilfield chemicals", "production chemicals", "drilling chemicals"],
    "water treatment": ["water treatment"],
    "hse": ["hse", "health safety", "process safety"],
    "fire and gas": ["fire and gas", "gas detection"],
    "esp": [" esp ", "electrical submersible pump", "electric submersible pump"],
    "rod pump": ["rod pump", "sucker rod pump"],
    "gas lift": ["gas lift"],
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
    "drilling": ["drilling"],
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
        getattr(record, "entity_type", "") or "",
        record.company_name or "",
        getattr(record, "canonical_name", "") or "",
        record.description or "",
        getattr(record, "industry", "") or "",
        getattr(record, "notes", "") or "",
        record.website or "",
        record.source_url or "",
        getattr(record, "abstract", "") or "",
        getattr(record, "summary", "") or "",
        getattr(record, "contact_page", "") or "",
        getattr(record, "linkedin_url", "") or "",
        getattr(record, "location", "") or "",
        getattr(record, "city", "") or "",
        getattr(record, "country", "") or "",
        getattr(record, "hq_country", "") or "",
        " ".join(getattr(record, "presence_countries", []) or []),
    ]).lower()


def _keyword_hit_count(haystack: str, keywords: List[str], synonym_map: Dict[str, List[str]]) -> int:
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
        if any(f" {term} " in padded or term in haystack for term in terms):
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


def _geo_score(record: CompanyRecord, spec: SearchSpec, haystack: str) -> tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    score = 0.0

    include_countries = [_norm_country(x) for x in (getattr(spec, "include_countries", None) or []) if x]
    exclude_countries = [_norm_country(x) for x in (getattr(spec, "exclude_countries", None) or []) if x]
    exclude_presence = [_norm_country(x) for x in (getattr(spec, "exclude_presence_countries", None) or []) if x]

    rec_country = _norm_country(getattr(record, "country", None))
    rec_hq = _norm_country(getattr(record, "hq_country", None))
    presence = [_norm_country(c) for c in (getattr(record, "presence_countries", None) or []) if c]

    if include_countries:
        matched = False
        if rec_hq and rec_hq in include_countries:
            score += 18
            breakdown["geo_hq_include"] = 18
            matched = True
        elif rec_country and rec_country in include_countries:
            score += 14
            breakdown["geo_country_include"] = 14
            matched = True
        elif any(c in include_countries for c in presence):
            score += 12
            breakdown["geo_presence_include"] = 12
            matched = True
        elif any(contains_country_or_city(haystack, c) for c in include_countries):
            score += 6
            breakdown["geo_text_include"] = 6
            matched = True
        if not matched:
            strict = bool(getattr(spec, "strict_mode", False))
            penalty = -18 if strict else -4
            score += penalty
            breakdown["geo_missing_include"] = penalty

    if exclude_countries:
        if rec_hq and rec_hq in exclude_countries:
            score -= 35
            breakdown["geo_excluded_hq"] = -35
        elif rec_country and rec_country in exclude_countries:
            score -= 28
            breakdown["geo_excluded_country"] = -28
        elif any(contains_country_or_city(haystack, c) for c in exclude_countries):
            score -= 6
            breakdown["geo_excluded_text"] = -6

    if exclude_presence:
        if _presence_matches(presence, exclude_presence):
            score -= 24
            breakdown["geo_excluded_presence"] = -24
        if "united states" in exclude_presence and getattr(record, "has_usa_presence", False):
            score -= 24
            breakdown["geo_excluded_presence_usa"] = -24
        if "egypt" in exclude_presence and getattr(record, "has_egypt_presence", False):
            score -= 24
            breakdown["geo_excluded_presence_egypt"] = -24
        for c in exclude_presence:
            if contains_country_or_city(haystack, c):
                score -= 4
                breakdown[f"geo_excluded_presence_text:{c}"] = breakdown.get(f"geo_excluded_presence_text:{c}", 0.0) - 4

    return score, breakdown


def score_company_record(record: CompanyRecord, spec: SearchSpec) -> float:
    score = 0.0
    breakdown: Dict[str, float] = {}

    entity_type = (getattr(spec, "entity_type", "company") or "company").strip().lower()
    target_category = (getattr(spec, "target_category", "general") or "general").strip().lower()
    solution_keywords = [str(x).strip().lower() for x in (getattr(spec, "solution_keywords", []) or []) if str(x).strip()]
    domain_keywords = [str(x).strip().lower() for x in (getattr(spec, "domain_keywords", []) or []) if str(x).strip()]
    commercial_intent = (getattr(spec, "commercial_intent", "general") or "general").strip().lower()

    haystack = _text_blob(record)

    if entity_type == "paper" or getattr(record, "entity_type", "") == "paper":
        if record.company_name:
            score += 18; breakdown["title"] = 18
        if getattr(record, "authors", ""):
            score += 12; breakdown["authors"] = 12
        if getattr(record, "doi", ""):
            score += 12; breakdown["doi"] = 12
        if getattr(record, "publication_year", ""):
            score += 4; breakdown["publication_year"] = 4
        if getattr(record, "abstract", "") or record.description:
            score += 16; breakdown["abstract"] = 16
    elif entity_type == "person" or getattr(record, "entity_type", "") == "person":
        if record.company_name:
            score += 14; breakdown["name"] = 14
        if getattr(record, "job_title", ""):
            score += 16; breakdown["job_title"] = 16
        if getattr(record, "employer_name", ""):
            score += 12; breakdown["employer_name"] = 12
        if getattr(record, "linkedin_profile", "") or record.linkedin_url:
            score += 18; breakdown["profile"] = 18
        if getattr(record, "city", "") or getattr(record, "country", ""):
            score += 6; breakdown["location"] = 6
    else:
        if record.company_name:
            score += 12; breakdown["company_name"] = 12
        if record.website:
            score += 16; breakdown["website"] = 16
        if record.domain:
            score += 6; breakdown["domain"] = 6
        if record.description:
            desc_len = len(record.description)
            val = 10 if desc_len > 200 else (6 if desc_len > 50 else 3)
            score += val; breakdown["description"] = val

        if record.email:
            score += 12; breakdown["email"] = 12
        if record.phone:
            score += 8; breakdown["phone"] = 8
        if record.contact_page:
            score += 6; breakdown["contact_page"] = 6
        if record.linkedin_url:
            score += 5; breakdown["linkedin_url"] = 5

        if not record.is_directory_or_media:
            score += 10; breakdown["not_directory"] = 10
        if record.page_type == "company":
            score += 8; breakdown["page_type_company"] = 8
        elif record.page_type in {"directory", "media", "blog"}:
            score -= 15; breakdown["page_type_bad"] = -15
        elif record.page_type == "document":
            score += 4; breakdown["page_type_document"] = 4

    sector_words = [w for w in (getattr(spec, "sector", "") or "").lower().split() if len(w) >= 2]
    sector_hits = sum(1 for w in sector_words if w in haystack)
    if sector_words:
        sector_ratio = sector_hits / len(sector_words)
        val = min(sector_ratio * 16, 16)
        score += val; breakdown["sector_match"] = val
        if sector_hits == 0:
            score -= 20; breakdown["sector_miss"] = -20

    if target_category == "software_company":
        software_hits = sum(1 for h in _SOFTWARE_HINTS if h in haystack)
        if software_hits >= 3:
            score += 12; breakdown["software_hits"] = 12
        elif software_hits >= 1:
            score += 6; breakdown["software_hits"] = 6
        else:
            score -= 14; breakdown["software_missing"] = -14

    if target_category == "service_company":
        service_hits = sum(1 for h in _SERVICE_HINTS if h in haystack)
        if service_hits >= 3:
            score += 12; breakdown["service_hits"] = 12
        elif service_hits >= 1:
            score += 6; breakdown["service_hits"] = 6
        else:
            score -= 12; breakdown["service_missing"] = -12

    if solution_keywords:
        solution_hits = _keyword_hit_count(haystack, solution_keywords, _SOLUTION_SYNONYM_MAP)
        solution_ratio = solution_hits / max(1, len(set(solution_keywords)))
        val = min(solution_ratio * 16, 16)
        score += val; breakdown["solution_match"] = val
        if target_category == "software_company" and solution_hits == 0:
            score -= 10; breakdown["solution_missing"] = -10

    if domain_keywords:
        domain_hits = _keyword_hit_count(haystack, domain_keywords, _DOMAIN_SYNONYM_MAP)
        domain_ratio = domain_hits / max(1, len(set(domain_keywords)))
        val = min(domain_ratio * 18, 18)
        score += val; breakdown["domain_match"] = val
        if domain_hits == 0:
            score -= 8; breakdown["domain_missing"] = -8

    for t in getattr(spec, "include_terms", []) or []:
        if t.strip().lower() in haystack:
            score += 4
            breakdown[f"include_term:{t.strip().lower()}"] = 4
    for t in getattr(spec, "exclude_terms", []) or []:
        if t.strip().lower() in haystack:
            score -= 6
            breakdown[f"exclude_term:{t.strip().lower()}"] = -6

    bonus = _commercial_intent_bonus(haystack, commercial_intent)
    if bonus:
        score += bonus
        breakdown["commercial_intent"] = bonus

    geo_score, geo_breakdown = _geo_score(record, spec, haystack)
    score += geo_score
    breakdown.update(geo_breakdown)

    if getattr(record, "is_verified", False):
        score += 6; breakdown["verified"] = 6
    if getattr(record, "validation_status", "") == "accepted":
        score += 4; breakdown["validation_accepted"] = 4

    final_score = round(max(0.0, min(100.0, score)), 2)
    if hasattr(record, "scoring_breakdown"):
        record.scoring_breakdown = breakdown
    return final_score


def score_records(records: List[CompanyRecord], spec: SearchSpec) -> List[CompanyRecord]:
    for rec in records:
        rec.confidence_score = score_company_record(rec, spec)
    return records
