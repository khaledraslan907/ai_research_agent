from __future__ import annotations

"""
query_expander.py
=================
Extra query expansion layer that sits between task parsing and provider-specific
query building.

Goals:
- broaden recall without losing constraints
- support bilingual English/Arabic variants when useful
- reuse domain packs and ontology hints
- stay backward-compatible with the current task/query model
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.domain_registry import detect_domain_packs, merge_domain_hints
from core.language_utils import choose_query_languages, expand_terms_multilingual
from core.ontology import collect_topic_terms, merge_unique
from core.models import SearchQuery
from core.task_models import TaskSpec


@dataclass
class ExpansionBundle:
    synonyms: List[str] = field(default_factory=list)
    broader_terms: List[str] = field(default_factory=list)
    narrower_terms: List[str] = field(default_factory=list)
    company_types: List[str] = field(default_factory=list)
    geo_variants: List[str] = field(default_factory=list)
    source_angles: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "synonyms": list(self.synonyms),
            "broader_terms": list(self.broader_terms),
            "narrower_terms": list(self.narrower_terms),
            "company_types": list(self.company_types),
            "geo_variants": list(self.geo_variants),
            "source_angles": list(self.source_angles),
            "languages": list(self.languages),
        }


_GENERIC_COMPANY_TYPES = {
    "service_company": ["service company", "contractor", "engineering services", "solutions provider"],
    "software_company": ["software company", "platform vendor", "technology provider", "saas company"],
    "manufacturer": ["manufacturer", "fabricator", "equipment maker", "industrial supplier"],
    "consultancy": ["consultancy", "consulting firm", "advisory company"],
    "general": ["company", "vendor", "supplier", "provider"],
}

_SOURCE_ANGLES = {
    "company": ["official website", "contact page", "about page", "brochure pdf"],
    "paper": ["publisher page", "pdf", "doi", "conference proceedings"],
    "person": ["linkedin profile", "speaker page", "team page", "author profile"],
    "event": ["exhibitor list", "speaker list", "conference page"],
    "tender": ["tender portal", "procurement notice", "vendor list"],
}


def expand_task_keywords(task_spec: TaskSpec, max_per_bucket: int = 8) -> ExpansionBundle:
    text = f"{getattr(task_spec, 'raw_prompt', '')} {getattr(task_spec, 'industry', '')} {' '.join(getattr(task_spec, 'solution_keywords', []) or [])} {' '.join(getattr(task_spec, 'domain_keywords', []) or [])}"
    packs = detect_domain_packs(text, getattr(task_spec, "industry", ""))
    merged = merge_domain_hints(packs)

    languages = choose_query_languages(getattr(task_spec, "raw_prompt", ""), getattr(getattr(task_spec, "geography", None), "include_countries", []) or [])

    bundle = ExpansionBundle(languages=languages)

    base_terms = merge_unique(
        getattr(task_spec, "solution_keywords", []) or [],
        getattr(task_spec, "domain_keywords", []) or [],
        collect_topic_terms(getattr(task_spec, "industry", ""), max_terms=6),
    )
    bundle.synonyms = expand_terms_multilingual(base_terms)[: max_per_bucket * 2]

    bundle.broader_terms = expand_terms_multilingual((merged.get("industries", []) or [])[:max_per_bucket])[:max_per_bucket]
    bundle.narrower_terms = expand_terms_multilingual((merged.get("domain_keywords", []) or [])[: max_per_bucket * 2])[: max_per_bucket * 2]

    category = getattr(task_spec, "target_category", "general") or "general"
    bundle.company_types = _GENERIC_COMPANY_TYPES.get(category, _GENERIC_COMPANY_TYPES["general"])[:max_per_bucket]

    geo_terms = list(getattr(getattr(task_spec, "geography", None), "include_countries", []) or [])
    if not geo_terms and getattr(getattr(task_spec, "geography", None), "exclude_presence_countries", None):
        geo_terms = list(getattr(task_spec.geography, "exclude_presence_countries", []) or [])
    bundle.geo_variants = expand_terms_multilingual(geo_terms)[: max_per_bucket * 2]

    primary_entity = (getattr(task_spec, "target_entity_types", []) or ["company"])[0]
    bundle.source_angles = list(_SOURCE_ANGLES.get(primary_entity, ["official website", "pdf", "directory"]))
    if merged.get("source_hints"):
        bundle.source_angles = merge_unique(bundle.source_angles, merged["source_hints"])[:max_per_bucket]

    return bundle


def build_expanded_queries(task_spec: TaskSpec, max_queries: int = 12) -> Dict[str, List[SearchQuery]]:
    """
    Produce provider-agnostic expanded query ideas.
    These are not meant to replace query_builder/llm_query_planner; they provide
    extra recall-oriented families that can be appended downstream.
    """
    bundle = expand_task_keywords(task_spec)
    topic = (getattr(task_spec, "industry", "") or "").strip() or "industry"
    primary_entity = (getattr(task_spec, "target_entity_types", []) or ["company"])[0]
    category = getattr(task_spec, "target_category", "general") or "general"
    include_countries = list(getattr(getattr(task_spec, "geography", None), "include_countries", []) or [])
    main_geo = include_countries[0] if include_countries else ""

    entity_word = {
        "company": "company",
        "paper": "paper",
        "person": "profile",
        "event": "event",
        "tender": "tender",
        "product": "product",
    }.get(primary_entity, "entity")
    if primary_entity == "company" and category == "service_company":
        entity_word = "service company"
    elif primary_entity == "company" and category == "software_company":
        entity_word = "software company"

    ddg: List[SearchQuery] = []
    exa: List[SearchQuery] = []
    tavily: List[SearchQuery] = []
    serpapi: List[SearchQuery] = []

    priority = 1
    seeds = merge_unique(getattr(task_spec, "domain_keywords", []) or [], getattr(task_spec, "solution_keywords", []) or [], [topic])
    for seed in seeds[:4]:
        q = f"{seed} {entity_word} {main_geo}".strip()
        ddg.append(SearchQuery(text=q, priority=priority, family="expanded_core", provider_hint="ddg"))
        priority += 1

    for term in bundle.narrower_terms[:4]:
        q = f"{term} {entity_word} {main_geo}".strip()
        ddg.append(SearchQuery(text=q, priority=priority, family="expanded_niche", provider_hint="ddg"))
        priority += 1

    for angle in bundle.source_angles[:4]:
        q = f"{topic} {entity_word} {angle} {main_geo}".strip()
        exa.append(SearchQuery(text=q, priority=priority, family="expanded_source", provider_hint="exa"))
        priority += 1

    for term in bundle.synonyms[:4]:
        q = f"{term} {topic} {entity_word} {main_geo}".strip()
        exa.append(SearchQuery(text=q, priority=priority, family="expanded_synonym", provider_hint="exa"))
        priority += 1

    for geo in bundle.geo_variants[:3] or ([main_geo] if main_geo else []):
        q = f"What are relevant {topic} {entity_word}s in {geo}?".strip()
        tavily.append(SearchQuery(text=q, priority=priority, family="expanded_question", provider_hint="tavily"))
        priority += 1

    if primary_entity == "person":
        for geo in include_countries[:2] or [main_geo]:
            if geo:
                serpapi.append(SearchQuery(
                    text=f'site:linkedin.com/in "{topic}" "{geo}"',
                    priority=priority,
                    family="expanded_people",
                    provider_hint="serpapi",
                ))
                priority += 1

    def _trim(items: List[SearchQuery], n: int) -> List[SearchQuery]:
        seen = set()
        out: List[SearchQuery] = []
        for item in items:
            key = item.text.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= n:
                break
        return out

    per_provider = max(2, max_queries // 4)
    return {
        "ddg": _trim(ddg, per_provider + 2),
        "exa": _trim(exa, per_provider + 2),
        "tavily": _trim(tavily, per_provider),
        "serpapi": _trim(serpapi, per_provider),
    }
