from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import INTENT_PARSE_PROMPT
from core.task_models import CredentialMode, GeographyRules, OutputSpec, TaskSpec


_VALID_TASK_TYPES = {
    "entity_discovery", "document_research", "entity_enrichment",
    "similar_entity_expansion", "market_research", "people_search",
}
_VALID_ENTITY_TYPES = {"company", "paper", "person", "organization", "event", "product", "tender"}
_VALID_ENTITY_CATEGORIES = {"service_company", "software_company", "general"}
_VALID_ATTRS = {
    "website", "email", "phone", "linkedin", "summary", "hq_country",
    "presence_countries", "pdf", "author", "authors", "doi", "abstract",
    "company_name", "deadline", "buyer", "exhibitors",
}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out=[]
    seen=set()
    for value in values or []:
        s=str(value or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _clean_topic(topic: str, raw_prompt: str = "") -> str:
    t = _norm_spaces(topic)
    if not t:
        return ""
    t = re.sub(r"\b(with|including|include|and extract)\b.*$", "", t, flags=re.I).strip(" ,.-")
    # remove output/contact noise
    t = re.sub(r"\b(website|websites|email|emails|phone|phones|contact|contacts|deadline|buyer|company names?|names?)\b", " ", t, flags=re.I)
    t = re.sub(r"\bsoftware companies?\b", " ", t, flags=re.I)
    t = re.sub(r"\bexhibitors?\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")
    # domain-specific corrections
    low = (raw_prompt or "").lower()
    if "food manufacturing" in low:
        return "food manufacturing"
    if "egyps" in low and ("wireline" in low or "well logging" in low):
        bits=[]
        if "wireline" in low: bits.append("wireline")
        if "well logging" in low: bits.append("well logging")
        return " ".join(bits) or "oil and gas"
    if any(x in low for x in ["oilfield", "petroleum", "oil and gas"]) and not t:
        return "oil and gas"
    return t


def _merge_geography(llm_geo: GeographyRules, regex_geo: GeographyRules) -> GeographyRules:
    include = _dedupe_keep_order(list(llm_geo.include_countries or []) + list(regex_geo.include_countries or []))
    exclude = [c for c in _dedupe_keep_order(list(llm_geo.exclude_countries or []) + list(regex_geo.exclude_countries or [])) if c not in include]
    excpres = [c for c in _dedupe_keep_order(list(llm_geo.exclude_presence_countries or []) + list(regex_geo.exclude_presence_countries or [])) if c not in include]
    exclude = [c for c in exclude if c not in excpres]
    return GeographyRules(include_countries=include, exclude_countries=exclude, exclude_presence_countries=excpres, strict_mode=bool(include or exclude or excpres))


def _dict_to_task_spec(raw_prompt: str, data: Dict[str, Any], regex_spec: Optional[TaskSpec] = None) -> TaskSpec:
    task_type = str(data.get("task_type") or "entity_discovery").strip()
    if task_type not in _VALID_TASK_TYPES:
        task_type = getattr(regex_spec, "task_type", "entity_discovery") if regex_spec else "entity_discovery"
        if task_type not in _VALID_TASK_TYPES:
            task_type = "entity_discovery"

    entity_type = str(data.get("entity_type") or "company").strip()
    if entity_type not in _VALID_ENTITY_TYPES:
        entity_type = (getattr(regex_spec, "target_entity_types", ["company"]) or ["company"])[0] if regex_spec else "company"
        if entity_type not in _VALID_ENTITY_TYPES:
            entity_type = "company"

    entity_category = str(data.get("entity_category") or "general").strip()
    if entity_category not in _VALID_ENTITY_CATEGORIES:
        entity_category = getattr(regex_spec, "target_category", "general") if regex_spec else "general"
        if entity_category not in _VALID_ENTITY_CATEGORIES:
            entity_category = "general"

    raw_topic = str(data.get("topic") or "").strip()
    regex_topic = getattr(regex_spec, "industry", "") if regex_spec else ""
    topic = _clean_topic(raw_topic, raw_prompt) or regex_topic

    prompt_lower = _norm_spaces(raw_prompt).lower()
    if "food manufacturing" in prompt_lower:
        topic = "food manufacturing"
        entity_category = "software_company"
    if ("oilfield service" in prompt_lower or "petroleum service" in prompt_lower or "خدمات البترول" in prompt_lower) and not topic:
        topic = "oil and gas"
        entity_category = "service_company"
    if "egyps" in prompt_lower:
        task_type = "market_research"
        entity_type = "company"
        entity_category = "service_company" if entity_category == "general" else entity_category
        if not topic:
            parts=[]
            if "wireline" in prompt_lower: parts.append("wireline")
            if "well logging" in prompt_lower: parts.append("well logging")
            topic = " ".join(parts) or "oil and gas"

    include = list(data.get("include_countries") or [])
    exclude = list(data.get("exclude_countries") or [])
    excpres = list(data.get("exclude_presence_countries") or [])
    llm_geo = GeographyRules(include_countries=include, exclude_countries=exclude, exclude_presence_countries=excpres, strict_mode=bool(include or exclude or excpres))
    if regex_spec:
        llm_geo = _merge_geography(llm_geo, regex_spec.geography)
    if "egyps" in prompt_lower and "egypt" not in [c.lower() for c in llm_geo.include_countries]:
        llm_geo.include_countries = _dedupe_keep_order(list(llm_geo.include_countries) + ["egypt"])
        llm_geo.strict_mode = True

    attrs = [a for a in list(data.get("attributes_wanted") or []) if a in _VALID_ATTRS]
    if regex_spec:
        attrs = _dedupe_keep_order(list(regex_spec.target_attributes or []) + attrs)
    if "website" not in attrs:
        attrs = ["website"] + [a for a in attrs if a != "website"]
    if "egyps" in prompt_lower:
        attrs = _dedupe_keep_order(attrs + ["company_name", "website"])
    if entity_type == "tender":
        attrs = _dedupe_keep_order(attrs + ["deadline", "buyer"])

    solution_keywords = list(getattr(regex_spec, "solution_keywords", []) or []) if regex_spec else []
    domain_keywords = list(getattr(regex_spec, "domain_keywords", []) or []) if regex_spec else []
    commercial_intent = getattr(regex_spec, "commercial_intent", "general") if regex_spec else "general"

    fmt = str(data.get("output_format") or "xlsx").strip()
    if fmt not in {"xlsx", "csv", "json", "pdf", "ui_table"}:
        fmt = "xlsx"
    try:
        max_results = int(data.get("max_results", getattr(regex_spec, "max_results", 25) if regex_spec else 25))
    except Exception:
        max_results = getattr(regex_spec, "max_results", 25) if regex_spec else 25
    max_results = max(1, min(max_results, 500))

    return TaskSpec(
        raw_prompt=raw_prompt,
        task_type=task_type,
        target_entity_types=[entity_type],
        target_category=entity_category,
        industry=topic,
        solution_keywords=solution_keywords,
        domain_keywords=domain_keywords,
        commercial_intent=commercial_intent,
        target_attributes=attrs,
        geography=llm_geo,
        output=OutputSpec(format=fmt, filename=f"results.{fmt if fmt != 'ui_table' else 'xlsx'}"),
        credential_mode=CredentialMode(mode="free"),
        use_local_llm=False,
        use_cloud_llm=False,
        max_results=max_results,
        mode=getattr(regex_spec, "mode", "Balanced") if regex_spec else "Balanced",
    )


def parse_with_llm(prompt: str, llm: FreeLLMClient, regex_spec: Optional[TaskSpec] = None) -> Optional[TaskSpec]:
    if not llm or not llm.is_available():
        return None
    try:
        llm_prompt = INTENT_PARSE_PROMPT.format(prompt=prompt)
        result = llm.generate_json(llm_prompt, timeout=30)
        if not result or not isinstance(result, dict):
            return None
        return _dict_to_task_spec(prompt, result, regex_spec=regex_spec)
    except Exception:
        return None


def _merge_llm_with_regex(prompt: str, llm_spec: TaskSpec, regex_spec: TaskSpec) -> TaskSpec:
    # Prefer regex for niche/domain/category when LLM is generic or polluted
    if regex_spec.target_category == "service_company" and llm_spec.target_category == "software_company" and "software companies in" not in prompt.lower():
        llm_spec.target_category = "service_company"
    if not llm_spec.industry or any(x in llm_spec.industry.lower() for x in ["website", "email", "extract", "buyer", "deadline", "company names"]):
        llm_spec.industry = regex_spec.industry or llm_spec.industry
    llm_spec.solution_keywords = _dedupe_keep_order(list(regex_spec.solution_keywords or []) + list(llm_spec.solution_keywords or []))
    llm_spec.domain_keywords = _dedupe_keep_order(list(regex_spec.domain_keywords or []) + list(llm_spec.domain_keywords or []))
    llm_spec.target_attributes = _dedupe_keep_order(list(regex_spec.target_attributes or []) + list(llm_spec.target_attributes or []))
    llm_spec.geography = _merge_geography(llm_spec.geography, regex_spec.geography)
    if not llm_spec.target_entity_types or llm_spec.target_entity_types == ["company"]:
        if getattr(regex_spec, "target_entity_types", None):
            llm_spec.target_entity_types = list(regex_spec.target_entity_types)
    if regex_spec.task_type in {"market_research", "document_research", "people_search"} and llm_spec.task_type == "entity_discovery":
        llm_spec.task_type = regex_spec.task_type
    return llm_spec


def parse_task_prompt_llm_first(prompt: str, llm: Optional[FreeLLMClient] = None) -> TaskSpec:
    from core.task_parser import parse_task_prompt as _regex_parse
    regex_spec = _regex_parse(prompt)
    if llm and llm.is_available():
        llm_spec = parse_with_llm(prompt, llm, regex_spec=regex_spec)
        if llm_spec:
            return _merge_llm_with_regex(prompt, llm_spec, regex_spec)
    return regex_spec
