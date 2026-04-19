from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import INTENT_PARSE_PROMPT
from core.task_models import CredentialMode, GeographyRules, OutputSpec, TaskSpec
from core.geography import normalize_country_name, expand_region_name

REGION_EXPANSIONS = {
    "europe": ["france", "germany", "united kingdom", "italy", "spain",
               "netherlands", "belgium", "switzerland", "norway", "sweden",
               "denmark", "finland", "poland", "austria", "czech republic",
               "portugal", "romania", "greece", "ireland", "hungary", "ukraine", "turkey"],
    "middle east": ["saudi arabia", "united arab emirates", "qatar", "oman",
                    "kuwait", "bahrain", "iraq", "jordan", "lebanon", "iran"],
    "north africa": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "mena": ["egypt", "libya", "algeria", "tunisia", "morocco",
             "saudi arabia", "united arab emirates", "qatar", "oman",
             "kuwait", "bahrain", "iraq", "jordan", "lebanon", "iran"],
    "asia": ["india", "china", "japan", "south korea", "singapore",
             "malaysia", "indonesia", "thailand", "vietnam", "philippines"],
    "africa": ["south africa", "nigeria", "angola", "kenya", "ghana",
               "ethiopia", "egypt", "morocco", "algeria"],
    "north america": ["usa", "canada", "mexico"],
    "south america": ["brazil", "argentina", "chile", "colombia", "peru"],
    "cis": ["russia", "kazakhstan", "azerbaijan", "ukraine"],
    "gcc": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
}

_GENERIC_DOC_TOPICS = {
    "petroleum", "petroleum engineering", "oil", "gas", "oil and gas", "oil & gas",
    "energy", "engineering", "research", "papers", "paper", "study", "studies",
    "article", "articles", "journal", "journals", "literature",
}

_DOC_TOPIC_PATTERNS = [
    r"(?:find|search|show|get|give me|list)\s+(?:research\s+)?(?:papers?|articles?|studies?|publications?)\s+(?:about|on|regarding|for)\s+(.+)$",
    r"(?:research\s+)?(?:papers?|articles?|studies?|publications?)\s+(?:about|on|regarding|for)\s+(.+)$",
    r"(?:find|search|show|get|give me|list)\s+literature\s+(?:about|on|regarding|for)\s+(.+)$",
]

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

_BROAD_DOMAIN_TAIL_RE = re.compile(
    r"""
    \s+
    (?:in|within|for|related\ to|used\ in|applied\ to)
    \s+
    (?:the\s+)?
    (?:
        petroleum(?:\s+engineering)? |
        oil(?:\s+and\s+gas|\s*&\s*gas)? |
        gas |
        energy |
        upstream |
        downstream
    )
    \b.*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DIGITAL_HINTS = (
    "digital", "software", "platform", "platforms", "analytics", "automation",
    "saas", "cloud", "ai", "artificial intelligence", "machine learning",
    "data", "iot", "scada", "tech company", "technology company", "technology vendor",
    "monitoring", "optimization", "optimisation",
)

_PRESENCE_PATTERNS = [
    r"\boperate(?:s|d|ing)?\s+outside\b",
    r"\bwork(?:s|ed|ing)?\s+outside\b",
    r"\bserv(?:e|es|ed|ing)\s+outside\b",
    r"\bactive\s+outside\b",
    r"\bpresent\s+outside\b",
    r"\bno\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bwithout\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bhas\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bwith\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bdo(?:es)?\s+not\s+have\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\b(?:exclude|excluding|reject|remove|avoid)\b[^.\n;]{0,80}\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|operations?|legal entities?)\b",
]

_CATEGORY_NOISE_TERMS = {
    "digital", "software", "technology", "tech", "analytics", "automation", "platform",
    "platforms", "saas", "cloud", "ai", "artificial intelligence", "machine learning",
    "data", "iot", "scada", "monitoring", "optimization", "optimisation",
    "company", "companies", "vendor", "vendors", "provider", "providers",
}

# Exact-only allowed user-facing labels.
_ALLOWED_SOLUTION_KEYWORDS = {
    "machine learning",
    "artificial intelligence",
    "ai",
    "analytics",
    "monitoring",
    "optimization",
    "automation",
    "iot",
    "scada",
    "digital twin",
    "predictive maintenance",
}

_SOLUTION_KEYWORD_PATTERNS = {
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "artificial intelligence": [r"\bartificial intelligence\b"],
    "ai": [r"\bai\b"],
    "analytics": [r"\banalytics\b", r"\banalytic\b"],
    "monitoring": [r"\bmonitoring\b", r"\bremote monitoring\b", r"\bsurveillance\b"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b", r"\boptimizer\b", r"\boptimiser\b"],
    "automation": [r"\bautomation\b", r"\bautomated\b", r"\bautonomous\b"],
    "iot": [r"\biot\b", r"\binternet of things\b"],
    "scada": [r"\bscada\b"],
    "digital twin": [r"\bdigital twin\b", r"\bdigital twins\b"],
    "predictive maintenance": [r"\bpredictive maintenance\b"],
}


def parse_with_llm(prompt: str, llm: FreeLLMClient) -> Optional[TaskSpec]:
    if not llm.is_available():
        return None

    llm_prompt = INTENT_PARSE_PROMPT.format(prompt=prompt)
    result = llm.generate_json(llm_prompt, timeout=30)
    if not result or not isinstance(result, dict):
        return None

    return _dict_to_task_spec(prompt, result)


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _expand_countries(country_list: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in country_list:
        item_lower = item.lower().strip()
        if item_lower in REGION_EXPANSIONS:
            expanded.extend(REGION_EXPANSIONS[item_lower])
        else:
            region_exp = expand_region_name(item_lower)
            if region_exp:
                expanded.extend(region_exp)
            else:
                norm = normalize_country_name(item_lower)
                if norm:
                    expanded.append(norm)
    return _dedupe_keep_order(expanded)


def _strip_output_noise(text: str) -> str:
    cleaned = _norm_spaces(text)
    for pattern in _OUTPUT_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _norm_spaces(cleaned.strip(" -,:;|"))


def _strip_broad_domain_tail(text: str) -> str:
    cleaned = _norm_spaces(text)
    cleaned = _BROAD_DOMAIN_TAIL_RE.sub("", cleaned)
    cleaned = _norm_spaces(cleaned.strip(" -,:;|"))

    words = cleaned.split()
    while len(words) > 1 and " ".join(words[-2:]).lower() in {"oil gas", "oil & gas"}:
        words = words[:-2]
    while len(words) > 1 and words[-1].lower() in {
        "petroleum", "engineering", "energy", "research", "studies", "journal", "journals"
    }:
        words = words[:-1]
    return _norm_spaces(" ".join(words))


def _is_generic_document_topic(text: str) -> bool:
    t = _norm_spaces(text.lower())
    if not t:
        return True
    if t in _GENERIC_DOC_TOPICS:
        return True

    words = [
        w for w in re.findall(r"[a-z0-9&+/.-]+", t)
        if w not in {"the", "a", "an", "of", "in", "for", "on", "and", "to", "with", "about"}
    ]
    if not words:
        return True
    return all(w in _GENERIC_DOC_TOPICS for w in words)


def _extract_document_topic_from_prompt(raw_prompt: str) -> str:
    prompt = _norm_spaces(raw_prompt)

    for pattern in _DOC_TOPIC_PATTERNS:
        m = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not m:
            continue

        candidate = m.group(1)
        candidate = _strip_output_noise(candidate)
        candidate = _strip_broad_domain_tail(candidate)
        candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = _norm_spaces(candidate.strip(" .,:;|-"))

        if candidate:
            return candidate

    quoted = re.findall(r"[\"“](.+?)[\"”]", prompt)
    for q in quoted:
        candidate = _strip_broad_domain_tail(_strip_output_noise(q))
        if candidate and not _is_generic_document_topic(candidate):
            return candidate

    return ""


def _normalize_entity_topic(text: str) -> str:
    text = _norm_spaces(text.lower())
    text = re.sub(r"\boil\s*(?:&|and)?\s*gas\b", "oil and gas", text)
    text = re.sub(r"\boil\s+gas\b", "oil and gas", text)

    removable = sorted(_CATEGORY_NOISE_TERMS, key=len, reverse=True)
    for term in removable:
        text = re.sub(r"\b" + re.escape(term) + r"\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip(" ,-")
    if re.search(r"\boil\b", text) and re.search(r"\bgas\b", text):
        return "oil and gas"
    return text


def _looks_like_category_polluted_topic(text: str) -> bool:
    t = _norm_spaces(text.lower())
    if not t:
        return False
    return any(term in t for term in _CATEGORY_NOISE_TERMS)


def _extract_solution_keywords_from_prompt(prompt_lower: str) -> list[str]:
    found = []
    for label, pats in _SOLUTION_KEYWORD_PATTERNS.items():
        if any(re.search(p, prompt_lower) for p in pats):
            found.append(label)
    return _dedupe_keep_order(found)


def _sanitize_solution_keywords(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for v in values or []:
        s = str(v).strip().lower()
        if s in _ALLOWED_SOLUTION_KEYWORDS and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_commercial_intent_from_prompt(prompt_lower: str) -> str:
    if re.search(r"\b(agent|agency|distributor|distribution|local representation|representative|representation)\b", prompt_lower):
        return "agent_or_distributor"
    if re.search(r"\b(reseller|resellers)\b", prompt_lower):
        return "reseller"
    if re.search(r"\b(partner|partners|channel partner|channel partners)\b", prompt_lower):
        return "partner"
    return "general"


def _repair_topic(raw_prompt: str, topic: str, task_type: str, regex_topic: str = "") -> str:
    topic = _norm_spaces(topic)

    if task_type == "document_research":
        extracted = _extract_document_topic_from_prompt(raw_prompt)
        if extracted:
            if _is_generic_document_topic(topic):
                return extracted
            if topic and topic.lower() in extracted.lower() and len(extracted.split()) >= len(topic.split()):
                return extracted
        topic = _strip_broad_domain_tail(_strip_output_noise(topic))
        return topic or extracted

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


def _looks_like_presence_exclusion(prompt_lower: str) -> bool:
    return any(re.search(pat, prompt_lower) for pat in _PRESENCE_PATTERNS)


def _clean_geography_with_regex(llm_geo: GeographyRules, regex_geo: GeographyRules) -> GeographyRules:
    include = list(llm_geo.include_countries or [])
    exclude = list(llm_geo.exclude_countries or [])
    exclude_presence = list(llm_geo.exclude_presence_countries or [])

    include.extend(list(regex_geo.include_countries or []))
    exclude.extend(list(regex_geo.exclude_countries or []))
    exclude_presence.extend(list(regex_geo.exclude_presence_countries or []))

    include = _dedupe_keep_order(include)
    exclude = _dedupe_keep_order(exclude)
    exclude_presence = _dedupe_keep_order(exclude_presence)

    include = [c for c in include if c not in exclude and c not in exclude_presence]
    exclude = [c for c in exclude if c not in exclude_presence]

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=bool(include or exclude or exclude_presence),
    )


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any], regex_spec: Optional[TaskSpec] = None) -> TaskSpec:
    task_type = data.get("task_type", "entity_discovery")
    if task_type not in {
        "entity_discovery", "entity_enrichment", "similar_entity_expansion",
        "market_research", "document_research", "people_search",
    }:
        task_type = "entity_discovery"

    entity_type = data.get("entity_type", "company")
    entity_category = data.get("entity_category", "general")
    if entity_category not in {"service_company", "software_company", "general"}:
        entity_category = "general"

    raw_topic = (data.get("topic") or "").strip()
    regex_topic = getattr(regex_spec, "industry", "") if regex_spec else ""
    topic = _repair_topic(raw_prompt, raw_topic, task_type, regex_topic=regex_topic)

    include_raw = data.get("include_countries", []) or []
    exclude_raw = data.get("exclude_countries", []) or []
    excpres_raw = data.get("exclude_presence_countries", []) or []

    include_countries = _expand_countries([c for c in include_raw if c])
    exclude_countries = _expand_countries([c for c in exclude_raw if c])
    excpres_countries = _expand_countries([c for c in excpres_raw if c])

    prompt_lower = _norm_spaces(raw_prompt).lower()
    if task_type in {"entity_discovery", "entity_enrichment", "similar_entity_expansion", "market_research"}:
        if any(hint in prompt_lower for hint in _DIGITAL_HINTS):
            entity_category = "software_company"

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
    valid_attrs = {
        "website", "email", "phone", "linkedin", "summary", "hq_country",
        "presence_countries", "pdf", "author", "authors", "doi", "abstract",
    }
    attributes = [a for a in attrs_raw if a in valid_attrs]
    if not attributes and regex_spec:
        attributes = list(regex_spec.target_attributes or [])
    if "website" not in attributes:
        attributes = ["website"] + [a for a in attributes if a != "website"]

    # IMPORTANT:
    # 1) Trust explicit prompt keywords first.
    # 2) Use LLM solution_keywords only if user did not explicitly type any.
    prompt_solution_keywords = _extract_solution_keywords_from_prompt(prompt_lower)
    llm_solution_keywords = _sanitize_solution_keywords(data.get("solution_keywords", []) or [])

    if prompt_solution_keywords:
        solution_keywords = prompt_solution_keywords
    else:
        solution_keywords = llm_solution_keywords
        if not solution_keywords and regex_spec:
            solution_keywords = list(getattr(regex_spec, "solution_keywords", []) or [])

    commercial_intent = str(data.get("commercial_intent", "") or "").strip().lower()
    if commercial_intent not in {"general", "agent_or_distributor", "reseller", "partner"}:
        commercial_intent = "general"

    regex_commercial_intent = _extract_commercial_intent_from_prompt(prompt_lower)
    if commercial_intent == "general" and regex_spec and getattr(regex_spec, "commercial_intent", "general") != "general":
        commercial_intent = regex_spec.commercial_intent
    elif commercial_intent == "general" and regex_commercial_intent != "general":
        commercial_intent = regex_commercial_intent

    fmt = data.get("output_format", "xlsx")
    if fmt not in {"xlsx", "csv", "json", "pdf", "ui_table"}:
        fmt = "xlsx"

    max_results = int(data.get("max_results", getattr(regex_spec, "max_results", 25)))
    max_results = max(1, min(max_results, 500))

    return TaskSpec(
        raw_prompt=raw_prompt,
        task_type=task_type,
        target_entity_types=[entity_type],
        target_category=entity_category,
        industry=topic,
        solution_keywords=solution_keywords,
        commercial_intent=commercial_intent,
        target_attributes=attributes,
        geography=llm_geo,
        output=OutputSpec(
            format=fmt,
            filename=f"results.{fmt if fmt != 'ui_table' else 'xlsx'}",
        ),
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

    if getattr(llm_spec, "target_category", "general") == "general" and getattr(regex_spec, "target_category", "general") != "general":
        llm_spec.target_category = regex_spec.target_category

    llm_spec.geography = _clean_geography_with_regex(llm_spec.geography, regex_spec.geography)

    if llm_spec.task_type == "entity_discovery" and regex_spec.task_type in {"document_research", "people_search", "market_research"}:
        llm_spec.task_type = regex_spec.task_type

    if not list(llm_spec.target_attributes or []):
        llm_spec.target_attributes = list(regex_spec.target_attributes or [])

    # Exact prompt terms are source-of-truth. Do not inflate by unioning inferred keywords.
    prompt_lower = _norm_spaces(prompt).lower()
    prompt_solution_keywords = _extract_solution_keywords_from_prompt(prompt_lower)
    if prompt_solution_keywords:
        llm_spec.solution_keywords = prompt_solution_keywords
    elif not list(llm_spec.solution_keywords or []):
        llm_spec.solution_keywords = list(regex_spec.solution_keywords or [])

    if getattr(llm_spec, "commercial_intent", "general") == "general":
        llm_spec.commercial_intent = getattr(regex_spec, "commercial_intent", "general")

    if not getattr(llm_spec, "max_results", 0):
        llm_spec.max_results = regex_spec.max_results

    return llm_spec


def parse_task_prompt_llm_first(
    prompt: str,
    llm: Optional[FreeLLMClient] = None,
) -> TaskSpec:
    from core.task_parser import parse_task_prompt as _regex_parse

    regex_spec = _regex_parse(prompt)

    if llm and llm.is_available():
        llm_prompt = INTENT_PARSE_PROMPT.format(prompt=prompt)
        result = llm.generate_json(llm_prompt, timeout=30)
        if result and isinstance(result, dict):
            llm_spec = _dict_to_task_spec(prompt, result, regex_spec=regex_spec)
            return _merge_llm_with_regex(prompt, llm_spec, regex_spec)

    return regex_spec
