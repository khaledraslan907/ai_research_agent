"""
llm_task_parser.py
==================
LLM-powered intent parser. Converts free-form user prompts into
structured TaskSpec objects.

Fixes in this version:
- Repairs overly broad document topics from the raw prompt
- Preserves specific paper subjects like "sucker rod pump"
- Strips output noise like "with authors", "export as PDF", etc.
- Supports people_search in the task-type allow-list
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import INTENT_PARSE_PROMPT
from core.task_models import (
    CredentialMode,
    GeographyRules,
    OutputSpec,
    TaskSpec,
)
from core.geography import normalize_country_name, expand_region_name


# Regions that should be expanded to country lists
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
    "petroleum",
    "petroleum engineering",
    "oil",
    "gas",
    "oil and gas",
    "oil & gas",
    "energy",
    "engineering",
    "research",
    "papers",
    "paper",
    "study",
    "studies",
    "article",
    "articles",
    "journal",
    "journals",
    "literature",
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


def parse_with_llm(prompt: str, llm: FreeLLMClient) -> Optional[TaskSpec]:
    """
    Parse user prompt into TaskSpec using LLM.
    Returns None if LLM is unavailable or parsing fails.
    """
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
    """Expand region names and normalise all country names."""
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

    # Also remove a trailing generic domain word if the phrase already has a specific head
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
    """
    Recover the specific document topic from the raw user prompt.
    Example:
      'find papers about sucker rod pump in petroleum'
      -> 'sucker rod pump'
    """
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
    """
    Repair overly broad topics produced by the LLM for document research.
    """
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


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any]) -> TaskSpec:
    """Convert LLM JSON output → TaskSpec dataclass."""

    # --- task type ---
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

    # --- entity ---
    entity_type = data.get("entity_type", "company")
    entity_category = data.get("entity_category", "general")
    if entity_category not in {"service_company", "software_company", "general"}:
        entity_category = "general"

    # --- topic ---
    raw_topic = (data.get("topic") or "").strip()
    topic = _repair_topic(raw_prompt, raw_topic, task_type)

    # --- geography ---
    include_raw = data.get("include_countries", []) or []
    exclude_raw = data.get("exclude_countries", []) or []
    excpres_raw = data.get("exclude_presence_countries", []) or []

    include_countries = _expand_countries([c for c in include_raw if c])
    exclude_countries = _expand_countries([c for c in exclude_raw if c])
    excpres_countries = _expand_countries([c for c in excpres_raw if c])

    include_countries = [c for c in include_countries if c not in exclude_countries]
    strict_mode = bool(include_countries or exclude_countries or excpres_countries)

    # --- attributes ---
    attrs_raw = data.get("attributes_wanted", []) or []
    valid_attrs = {
        "website", "email", "phone", "linkedin", "summary",
        "hq_country", "presence_countries", "pdf", "author", "authors", "doi", "abstract"
    }
    attributes = [a for a in attrs_raw if a in valid_attrs]
    if "website" not in attributes:
        attributes = ["website"] + attributes

    # --- output ---
    fmt = data.get("output_format", "xlsx")
    if fmt not in {"xlsx", "csv", "json", "pdf", "ui_table"}:
        fmt = "xlsx"

    # --- max results ---
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


def parse_task_prompt_llm_first(
    prompt: str,
    llm: Optional[FreeLLMClient] = None,
) -> TaskSpec:
    """
    Main entry point. Tries LLM first; falls back to regex parser.
    """
    if llm and llm.is_available():
        result = parse_with_llm(prompt, llm)
        if result and result.industry:
            return result

    from core.task_parser import parse_task_prompt as _regex_parse
    return _regex_parse(prompt)
