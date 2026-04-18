"""
llm_task_parser.py
==================
LLM-powered intent parser with regex repair / merge.

Main improvements in this version:
- Preserves digital/software-company intent even when the LLM collapses the
  topic to just "oil and gas"
- Repairs overly broad document topics from the raw prompt
- Treats phrasing like "operate outside Egypt and USA" as presence exclusion
- Merges strong regex signals when the LLM output is incomplete or too generic
"""

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
)

_PRESENCE_OUTSIDE_PATTERNS = [
    r"\boperate(?:s|d|ing)?\s+outside\b",
    r"\bwork(?:s|ed|ing)?\s+outside\b",
    r"\bserv(?:e|es|ed|ing)\s+outside\b",
    r"\bactive\s+outside\b",
    r"\bpresent\s+outside\b",
]


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
    return sorted(set(expanded))


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


def _repair_topic(raw_prompt: str, topic: str, task_type: str) -> str:
    topic = _norm_spaces(topic)

    if task_type != "document_research":
        return topic

    extracted = _extract_document_topic_from_prompt(raw_prompt)
    if extracted:
        if _is_generic_document_topic(topic):
            return extracted
        if topic and topic.lower() in extracted.lower() and len(extracted.split()) >= len(topic.split()):
            return extracted

    topic = _strip_broad_domain_tail(_strip_output_noise(topic))
    return topic or extracted


def _looks_like_presence_exclusion(prompt_lower: str) -> bool:
    return any(re.search(pat, prompt_lower) for pat in _PRESENCE_OUTSIDE_PATTERNS)


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any]) -> TaskSpec:
    task_type = data.get("task_type", "entity_discovery")
    if task_type not in {
        "entity_discovery",
        "entity_enrichment",
        "similar_entity_expansion",
        "market_research",
        "document_research",
        "people_search",
    }:
        task_type = "entity_discovery"

    entity_type = data.get("entity_type", "company")
    entity_category = data.get("entity_category", "general")
    if entity_category not in {"service_company", "software_company", "general"}:
        entity_category = "general"

    raw_topic = (data.get("topic") or "").strip()
    topic = _repair_topic(raw_prompt, raw_topic, task_type)

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
        excpres_countries = sorted(set(exclude_countries))

    include_countries = [c for c in include_countries if c not in exclude_countries and c not in excpres_countries]
    strict_mode = bool(include_countries or exclude_countries or excpres_countries)

    attrs_raw = data.get("attributes_wanted", []) or []
    valid_attrs = {
        "website", "email", "phone", "linkedin", "summary", "hq_country",
        "presence_countries", "pdf", "author", "authors", "doi", "abstract"
    }
    attributes = [a for a in attrs_raw if a in valid_attrs]
    if "website" not in attributes:
        attributes = ["website"] + attributes

    fmt = data.get("output_format", "xlsx")
    if fmt not in {"xlsx", "csv", "json", "pdf", "ui_table"}:
        fmt = "xlsx"

    max_results = int(data.get("max_results", 25))
    max_results = max(1, min(max_results, 500))

    return TaskSpec(
        raw_prompt=raw_prompt,
        task_type=task_type,
        target_entity_types=[entity_type],
        target_category=entity_category,
        industry=topic,
        target_attributes=attributes,
        geography=GeographyRules(
            include_countries=include_countries,
            exclude_countries=exclude_countries,
            exclude_presence_countries=excpres_countries,
            strict_mode=strict_mode,
        ),
        output=OutputSpec(
            format=fmt,
            filename=f"results.{fmt if fmt != 'ui_table' else 'xlsx'}",
        ),
        credential_mode=CredentialMode(mode="free"),
        use_local_llm=False,
        use_cloud_llm=False,
        max_results=max_results,
        mode="Balanced",
    )


def _merge_llm_with_regex(prompt: str, llm_spec: TaskSpec, regex_spec: TaskSpec) -> TaskSpec:
    """
    Fill gaps in the LLM parse with strong regex signals.
    """
    # Prefer regex if the LLM left the topic blank or too generic.
    if not getattr(llm_spec, "industry", "").strip():
        llm_spec.industry = regex_spec.industry
    elif llm_spec.task_type == "document_research" and _is_generic_document_topic(llm_spec.industry):
        llm_spec.industry = regex_spec.industry or llm_spec.industry

    # Strengthen category when regex confidently detects software/service company
    if getattr(llm_spec, "target_category", "general") == "general" and getattr(regex_spec, "target_category", "general") != "general":
        llm_spec.target_category = regex_spec.target_category

    # Geography merge
    llm_geo = llm_spec.geography
    rx_geo = regex_spec.geography

    if not list(llm_geo.include_countries or []) and list(rx_geo.include_countries or []):
        llm_geo.include_countries = list(rx_geo.include_countries or [])
    if not list(llm_geo.exclude_countries or []) and list(rx_geo.exclude_countries or []):
        llm_geo.exclude_countries = list(rx_geo.exclude_countries or [])
    if not list(llm_geo.exclude_presence_countries or []) and list(rx_geo.exclude_presence_countries or []):
        llm_geo.exclude_presence_countries = list(rx_geo.exclude_presence_countries or [])

    llm_geo.strict_mode = bool(
        list(llm_geo.include_countries or [])
        or list(llm_geo.exclude_countries or [])
        or list(llm_geo.exclude_presence_countries or [])
    )

    # If LLM missed a more specific task type, allow regex to upgrade it.
    if llm_spec.task_type == "entity_discovery" and regex_spec.task_type in {"document_research", "people_search", "market_research"}:
        llm_spec.task_type = regex_spec.task_type

    # Use regex attributes when LLM returned almost nothing.
    if not list(llm_spec.target_attributes or []):
        llm_spec.target_attributes = list(regex_spec.target_attributes or [])

    return llm_spec


def parse_task_prompt_llm_first(
    prompt: str,
    llm: Optional[FreeLLMClient] = None,
) -> TaskSpec:
    """
    Main entry point. Tries LLM first, then repairs/merges with the regex parser.
    """
    from core.task_parser import parse_task_prompt as _regex_parse

    regex_spec = _regex_parse(prompt)

    if llm and llm.is_available():
        result = parse_with_llm(prompt, llm)
        if result:
            return _merge_llm_with_regex(prompt, result, regex_spec)

    return regex_spec
