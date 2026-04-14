"""
llm_task_parser.py
==================
LLM-powered intent parser. Converts free-form user prompts into
structured TaskSpec objects.

Why this replaces task_parser.py:
  - Regex needed 4 iterations to parse "service companies in Egypt working in oil and gas"
  - LLM understands natural language, synonyms, implied intent
  - Single point of failure → single point of improvement
  - Cost: ~0 with Groq free tier (14,400 req/day free)

The regex parser (task_parser.py) is kept as a fallback for when
no LLM backend is available (fully offline mode).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import INTENT_PARSE_PROMPT
from core.task_models import (
    CredentialMode, GeographyRules, OutputSpec, TaskSpec,
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


def parse_with_llm(prompt: str, llm: FreeLLMClient) -> Optional[TaskSpec]:
    """
    Parse user prompt into TaskSpec using LLM.
    Returns None if LLM is unavailable or parsing fails — caller falls back to regex.
    """
    if not llm.is_available():
        return None

    llm_prompt = INTENT_PARSE_PROMPT.format(prompt=prompt)
    result = llm.generate_json(llm_prompt, timeout=30)

    if not result or not isinstance(result, dict):
        return None

    return _dict_to_task_spec(prompt, result)


def _expand_countries(country_list: list[str]) -> list[str]:
    """Expand region names and normalise all country names."""
    expanded = []
    for item in country_list:
        item_lower = item.lower().strip()
        if item_lower in REGION_EXPANSIONS:
            expanded.extend(REGION_EXPANSIONS[item_lower])
        else:
            # try geography module expansion
            region_exp = expand_region_name(item_lower)
            if region_exp:
                expanded.extend(region_exp)
            else:
                norm = normalize_country_name(item_lower)
                if norm:
                    expanded.append(norm)
    return sorted(set(expanded))


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any]) -> TaskSpec:
    """Convert LLM JSON output → TaskSpec dataclass."""

    # --- task type ---
    task_type = data.get("task_type", "entity_discovery")
    if task_type not in {
        "entity_discovery", "entity_enrichment", "similar_entity_expansion",
        "market_research", "document_research",
    }:
        task_type = "entity_discovery"

    # --- entity ---
    entity_type = data.get("entity_type", "company")
    entity_category = data.get("entity_category", "general")
    if entity_category not in {"service_company", "software_company", "general"}:
        entity_category = "general"

    # --- topic (most critical field) ---
    topic = (data.get("topic") or "").strip()
    # Sanitise — remove country names that may have leaked in
    if topic:
        words = topic.lower().split()
        country_names = set(normalize_country_name(w) for w in words)
        # If the entire topic is just a country name, clear it
        if all(w in REGION_EXPANSIONS or normalize_country_name(w) != w for w in words):
            topic = ""

    # --- geography ---
    include_raw  = data.get("include_countries", []) or []
    exclude_raw  = data.get("exclude_countries", []) or []
    excpres_raw  = data.get("exclude_presence_countries", []) or []

    include_countries  = _expand_countries([c for c in include_raw if c])
    exclude_countries  = _expand_countries([c for c in exclude_raw if c])
    excpres_countries  = _expand_countries([c for c in excpres_raw if c])

    # Remove any include country that is also in exclude
    include_countries = [c for c in include_countries if c not in exclude_countries]

    strict_mode = bool(include_countries or exclude_countries or excpres_countries)

    # --- attributes ---
    attrs_raw = data.get("attributes_wanted", []) or []
    valid_attrs = {"website", "email", "phone", "linkedin", "summary",
                   "hq_country", "presence_countries", "pdf"}
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
    # Try LLM
    if llm and llm.is_available():
        result = parse_with_llm(prompt, llm)
        if result and result.industry:   # valid if topic was extracted
            return result

    # Fallback to regex parser
    from core.task_parser import parse_task_prompt as _regex_parse
    return _regex_parse(prompt)
