"""LLM-first task parser with regex fallback and conservative merge."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import INTENT_PARSE_PROMPT
from core.task_models import CredentialMode, GeographyRules, OutputSpec, TaskSpec
from core.geography import normalize_country_name, expand_region_name

GEO_ALIAS_MAP = {
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "usa": "usa",
    "u.s.a.": "usa",
    "u.s.": "usa",
    "us": "usa",
}

REGION_EXPANSIONS = {
    "europe": [
        "united kingdom", "france", "germany", "italy", "spain", "portugal", "netherlands", "belgium",
        "switzerland", "austria", "norway", "sweden", "denmark", "finland", "poland", "czech republic",
        "romania", "greece", "turkey", "hungary", "ukraine", "ireland", "serbia", "croatia",
        "bulgaria", "slovakia", "slovenia", "estonia", "latvia", "lithuania", "luxembourg", "iceland",
    ],
    "middle east": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain", "iraq", "jordan", "lebanon", "iran"],
    "north africa": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "mena": [
        "egypt", "libya", "algeria", "tunisia", "morocco", "saudi arabia", "united arab emirates",
        "qatar", "oman", "kuwait", "bahrain", "iraq", "jordan", "lebanon", "iran",
    ],
    "gcc": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
}

_GENERIC_DOC_TOPICS = {
    "petroleum", "petroleum engineering", "oil", "gas", "oil and gas", "oil & gas", "energy",
    "engineering", "research", "papers", "paper", "study", "studies", "article", "articles", "journal",
    "journals", "literature",
}

_OUTPUT_NOISE_PATTERNS = [
    r"\bwith\s+authors\b.*$",
    r"\bwith\s+abstracts?\b.*$",
    r"\bwith\s+doi\b.*$",
    r"\bexport\s+as\s+\w+\b.*$",
    r"\bexport\b.*$",
    r"\bas\s+pdf\b.*$",
    r"\bas\s+csv\b.*$",
    r"\bas\s+xlsx\b.*$",
    r"\bas\s+excel\b.*$",
    r"\bto\s+pdf\b.*$",
    r"\bto\s+csv\b.*$",
    r"\bdownload\b.*$",
]

_CATEGORY_NOISE_TERMS = {
    "digital", "software", "technology", "tech", "analytics", "automation", "platform", "platforms", "saas",
    "cloud", "ai", "artificial intelligence", "machine learning", "data", "iot", "scada", "monitoring",
    "optimization", "optimisation", "company", "companies", "vendor", "vendors", "provider", "providers",
    "service", "services", "contractor", "contractors",
}

_PRESENCE_PATTERNS = [
    r"\boperate(?:s|d|ing)?\s+outside\b",
    r"\bwork(?:s|ed|ing)?\s+outside\b",
    r"\bserv(?:e|es|ed|ing)?\s+outside\b",
    r"\bactive\s+outside\b",
    r"\bpresent\s+outside\b",
    r"\bno\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bwithout\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bhas\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bwith\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bdo(?:es)?\s+not\s+have\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\b(?:do(?:es)?\s+not|don't|doesn't)\s+(?:operate|work|serve|have|be\s+active|be\s+present)\b.*?\b(?:in|inside|within)\b",
    r"\bnot\s+(?:operating|working|serving|active|present)\b.*?\b(?:in|inside|within)\b",
]


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _merge_keyword_lists(primary: list[str], secondary: list[str]) -> list[str]:
    return _dedupe_keep_order(list(primary or []) + list(secondary or []))


def _norm_country_value(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    if s in GEO_ALIAS_MAP:
        return GEO_ALIAS_MAP[s]
    norm = normalize_country_name(s)
    return norm or s


def _norm_country_list(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for v in values or []:
        n = _norm_country_value(v)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _expand_countries(country_list: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in country_list:
        item_lower = str(item).lower().strip()
        if not item_lower:
            continue
        if item_lower in GEO_ALIAS_MAP:
            expanded.append(GEO_ALIAS_MAP[item_lower])
            continue
        if item_lower in REGION_EXPANSIONS:
            expanded.extend(REGION_EXPANSIONS[item_lower])
            continue
        region_exp = expand_region_name(item_lower)
        if region_exp:
            expanded.extend(region_exp)
            continue
        norm = normalize_country_name(item_lower)
        if norm:
            expanded.append(norm)
    return _norm_country_list(expanded)


def _strip_output_noise(text: str) -> str:
    cleaned = _norm_spaces(text)
    for pattern in _OUTPUT_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _norm_spaces(cleaned.strip(" -,:;|"))


def _normalize_entity_topic(text: str) -> str:
    text = _norm_spaces(text.lower())
    text = re.sub(r"\boil\s*(?:&|and)?\s*gas\b", "oil and gas", text)
    text = re.sub(r"\bpetroleum\b", "oil and gas", text)
    for term in sorted(_CATEGORY_NOISE_TERMS, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(term) + r"\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-./")
    if "oil and gas" in text:
        lead = re.sub(r"\boil and gas\b", " ", text).strip(" ,-./")
        return f"{lead} oil and gas".strip() if lead else "oil and gas"
    return text


def _is_generic_document_topic(text: str) -> bool:
    t = _norm_spaces(text.lower())
    if not t:
        return True
    if t in _GENERIC_DOC_TOPICS:
        return True
    words = [w for w in re.findall(r"[a-z0-9&+/.-]+", t) if w not in {"the", "a", "an", "of", "in", "for", "on", "and", "to", "with", "about"}]
    if not words:
        return True
    return all(w in _GENERIC_DOC_TOPICS for w in words)


def _looks_like_category_polluted_topic(text: str) -> bool:
    t = _norm_spaces(text.lower())
    return bool(t and any(term in t for term in _CATEGORY_NOISE_TERMS))


def _looks_like_presence_exclusion(prompt_lower: str) -> bool:
    return any(re.search(pat, prompt_lower) for pat in _PRESENCE_PATTERNS)


def _clean_geography_with_regex(llm_geo: GeographyRules, regex_geo: GeographyRules) -> GeographyRules:
    llm_include = _norm_country_list(list(llm_geo.include_countries or []))
    llm_exclude = _norm_country_list(list(llm_geo.exclude_countries or []))
    llm_exclude_presence = _norm_country_list(list(llm_geo.exclude_presence_countries or []))

    rx_include = _norm_country_list(list(regex_geo.include_countries or []))
    rx_exclude = _norm_country_list(list(regex_geo.exclude_countries or []))
    rx_exclude_presence = _norm_country_list(list(regex_geo.exclude_presence_countries or []))

    if rx_include:
        llm_exclude = [c for c in llm_exclude if c not in rx_include]
        llm_exclude_presence = [c for c in llm_exclude_presence if c not in rx_include]

    include = _dedupe_keep_order(llm_include + rx_include)
    exclude = _dedupe_keep_order(llm_exclude + rx_exclude)
    exclude_presence = _dedupe_keep_order(llm_exclude_presence + rx_exclude_presence)

    exclude_presence = [c for c in exclude_presence if c not in include]
    exclude = [c for c in exclude if c not in include and c not in exclude_presence]

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=bool(include or exclude or exclude_presence),
    )


def _repair_topic(raw_prompt: str, topic: str, task_type: str, regex_topic: str = "") -> str:
    topic = _norm_spaces(topic)
    if task_type == "document_research":
        topic = _strip_output_noise(topic)
        if _is_generic_document_topic(topic) and regex_topic:
            return regex_topic
        return topic or regex_topic or raw_prompt
    if regex_topic:
        regex_topic = _norm_spaces(regex_topic)
        if not topic:
            return regex_topic
        if _looks_like_category_polluted_topic(topic):
            normalized = _normalize_entity_topic(topic)
            if normalized and regex_topic.lower() in normalized.lower():
                return regex_topic
            if normalized:
                return normalized
    return _normalize_entity_topic(topic) or regex_topic or topic


def _finalize_topic(topic: str, regex_spec: TaskSpec, prompt_lower: str) -> str:
    topic = _normalize_entity_topic(topic)
    regex_topic = _norm_spaces(getattr(regex_spec, "industry", ""))
    if regex_topic and topic and regex_topic.lower() in topic.lower():
        topic = regex_topic
    if regex_topic and not topic:
        topic = regex_topic

    domain_keywords = list(getattr(regex_spec, "domain_keywords", []) or [])
    if (not topic or topic == "oil and gas") and domain_keywords:
        if re.search(r"\boil and gas\b|\bpetroleum\b|نفط|غاز|بترول", prompt_lower):
            return f"{' / '.join(domain_keywords[:4])} in oil and gas"
        return " / ".join(domain_keywords[:4])
    return topic or regex_topic or "general"


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any], regex_spec: Optional[TaskSpec] = None) -> TaskSpec:
    task_type = data.get("task_type", "entity_discovery")
    if task_type not in {"entity_discovery", "entity_enrichment", "similar_entity_expansion", "market_research", "document_research", "people_search"}:
        task_type = "entity_discovery"

    entity_type = data.get("entity_type", (regex_spec.target_entity_types[0] if regex_spec and regex_spec.target_entity_types else "company"))
    entity_category = data.get("entity_category", getattr(regex_spec, "target_category", "general") if regex_spec else "general")
    if entity_category not in {"service_company", "software_company", "general"}:
        entity_category = "general"

    raw_topic = (data.get("topic") or "").strip()
    regex_topic = getattr(regex_spec, "industry", "") if regex_spec else ""
    topic = _repair_topic(raw_prompt, raw_topic, task_type, regex_topic=regex_topic)

    include_countries = _expand_countries([c for c in (data.get("include_countries", []) or []) if c])
    exclude_countries = _expand_countries([c for c in (data.get("exclude_countries", []) or []) if c])
    excpres_countries = _expand_countries([c for c in (data.get("exclude_presence_countries", []) or []) if c])
    include_set = set(include_countries)
    exclude_countries = [c for c in exclude_countries if c not in include_set]
    excpres_countries = [c for c in excpres_countries if c not in include_set]

    prompt_lower = _norm_spaces(raw_prompt).lower()
    if task_type in {"entity_discovery", "entity_enrichment", "similar_entity_expansion", "market_research"}:
        if any(h in prompt_lower for h in ["digital", "software", "analytics", "ai", "cloud", "platform"]):
            entity_category = "software_company"
        if any(h in prompt_lower for h in ["service", "contractor", "wireline", "slickline", "well logging", "خدمات", "مقاول"]):
            entity_category = "service_company"

    if _looks_like_presence_exclusion(prompt_lower) and exclude_countries and not excpres_countries:
        excpres_countries = _dedupe_keep_order(excpres_countries + exclude_countries)
        exclude_countries = []

    llm_geo = GeographyRules(
        include_countries=include_countries,
        exclude_countries=exclude_countries,
        exclude_presence_countries=excpres_countries,
        strict_mode=bool(include_countries or exclude_countries or excpres_countries),
    )
    if regex_spec:
        llm_geo = _clean_geography_with_regex(llm_geo, regex_spec.geography)

    attrs_raw = data.get("attributes_wanted", []) or []
    valid_attrs = {"website", "email", "phone", "linkedin", "summary", "hq_country", "presence_countries", "pdf", "author", "authors", "doi", "abstract"}
    attributes = [a for a in attrs_raw if a in valid_attrs]
    if not attributes and regex_spec:
        attributes = list(regex_spec.target_attributes or [])
    if "website" not in attributes:
        attributes = ["website"] + [a for a in attributes if a != "website"]
    if llm_geo.strict_mode:
        attributes = _merge_keyword_lists(attributes, ["presence_countries", "hq_country"])

    llm_solution_keywords = list(data.get("solution_keywords", []) or [])
    llm_domain_keywords = list(data.get("domain_keywords", []) or [])

    solution_keywords = _merge_keyword_lists(llm_solution_keywords, list(getattr(regex_spec, "solution_keywords", []) or []))
    domain_keywords = _merge_keyword_lists(llm_domain_keywords, list(getattr(regex_spec, "domain_keywords", []) or []))
    commercial_intent = data.get("commercial_intent") or (getattr(regex_spec, "commercial_intent", "general") if regex_spec else "general")

    fmt = data.get("output_format", "xlsx")
    if fmt not in {"xlsx", "csv", "json", "pdf", "ui_table"}:
        fmt = "xlsx"

    max_results = int(data.get("max_results", getattr(regex_spec, "max_results", 25) if regex_spec else 25))
    max_results = max(1, min(max_results, 500))
    topic = _finalize_topic(topic, regex_spec or TaskSpec(), raw_prompt.lower())

    return TaskSpec(
        raw_prompt=raw_prompt,
        task_type=task_type,
        target_entity_types=[entity_type],
        target_category=entity_category,
        industry=topic,
        solution_keywords=solution_keywords,
        domain_keywords=domain_keywords,
        commercial_intent=commercial_intent,
        target_attributes=attributes,
        geography=llm_geo,
        output=OutputSpec(format=fmt, filename=f"results.{fmt if fmt != 'ui_table' else 'xlsx'}"),
        credential_mode=CredentialMode(mode="free"),
        use_local_llm=False,
        use_cloud_llm=False,
        max_results=max_results,
        mode=getattr(regex_spec, "mode", "Balanced") if regex_spec else "Balanced",
    )


def _merge_llm_with_regex(prompt: str, llm_spec: TaskSpec, regex_spec: TaskSpec) -> TaskSpec:
    if not getattr(llm_spec, "industry", "").strip():
        llm_spec.industry = regex_spec.industry
    elif llm_spec.task_type == "document_research" and _is_generic_document_topic(llm_spec.industry):
        llm_spec.industry = regex_spec.industry or llm_spec.industry
    elif regex_spec.industry and _looks_like_category_polluted_topic(llm_spec.industry):
        repaired = _normalize_entity_topic(llm_spec.industry)
        llm_spec.industry = regex_spec.industry if repaired == regex_spec.industry else repaired

    llm_spec.industry = _finalize_topic(llm_spec.industry, regex_spec, prompt.lower())

    if getattr(llm_spec, "target_category", "general") == "general" and getattr(regex_spec, "target_category", "general") != "general":
        llm_spec.target_category = regex_spec.target_category

    llm_spec.geography = _clean_geography_with_regex(llm_spec.geography, regex_spec.geography)

    if llm_spec.task_type == "entity_discovery" and regex_spec.task_type in {"document_research", "people_search", "market_research"}:
        llm_spec.task_type = regex_spec.task_type

    if not list(llm_spec.target_attributes or []):
        llm_spec.target_attributes = list(regex_spec.target_attributes or [])
    if llm_spec.geography.strict_mode:
        llm_spec.target_attributes = _merge_keyword_lists(list(llm_spec.target_attributes or []), ["presence_countries", "hq_country"])

    llm_spec.solution_keywords = _merge_keyword_lists(list(getattr(llm_spec, "solution_keywords", []) or []), list(getattr(regex_spec, "solution_keywords", []) or []))
    llm_spec.domain_keywords = _merge_keyword_lists(list(getattr(llm_spec, "domain_keywords", []) or []), list(getattr(regex_spec, "domain_keywords", []) or []))
    if getattr(llm_spec, "commercial_intent", "general") == "general":
        llm_spec.commercial_intent = getattr(regex_spec, "commercial_intent", "general")

    rx_inc = _norm_country_list(list(getattr(regex_spec.geography, "include_countries", []) or []))
    if rx_inc:
        llm_spec.geography.include_countries = _norm_country_list(list(getattr(llm_spec.geography, "include_countries", []) or []) + rx_inc)
        llm_spec.geography.exclude_countries = [c for c in _norm_country_list(list(getattr(llm_spec.geography, "exclude_countries", []) or [])) if c not in rx_inc]
        llm_spec.geography.exclude_presence_countries = [c for c in _norm_country_list(list(getattr(llm_spec.geography, "exclude_presence_countries", []) or [])) if c not in rx_inc]
        llm_spec.geography.strict_mode = bool(llm_spec.geography.include_countries or llm_spec.geography.exclude_countries or llm_spec.geography.exclude_presence_countries)

    if not getattr(llm_spec, "max_results", 0):
        llm_spec.max_results = regex_spec.max_results
    return llm_spec


def parse_with_llm(prompt: str, llm: FreeLLMClient) -> Optional[TaskSpec]:
    if not llm or not llm.is_available():
        return None
    llm_prompt = INTENT_PARSE_PROMPT.format(prompt=prompt)
    result = llm.generate_json(llm_prompt, timeout=30)
    if not result or not isinstance(result, dict):
        return None

    from core.task_parser import parse_task_prompt as _regex_parse
    regex_spec = _regex_parse(prompt)
    return _dict_to_task_spec(prompt, result, regex_spec=regex_spec)


def parse_task_prompt_llm_first(prompt: str, llm: Optional[FreeLLMClient] = None) -> TaskSpec:
    from core.task_parser import parse_task_prompt as _regex_parse

    regex_spec = _regex_parse(prompt)
    if llm and llm.is_available():
        try:
            result = parse_with_llm(prompt, llm)
            if result:
                return _merge_llm_with_regex(prompt, result, regex_spec)
        except Exception:
            pass
    return regex_spec
