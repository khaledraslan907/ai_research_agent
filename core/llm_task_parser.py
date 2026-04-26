from __future__ import annotations
from typing import Optional
from core.free_llm_client import FreeLLMClient
from core.task_models import TaskSpec
from core.task_parser import parse_task_prompt



# ---- release safety override ----
def parse_task_prompt_llm_first(raw_prompt: str, llm=None):
    from core.task_parser import parse_task_prompt
    regex_spec = parse_task_prompt(raw_prompt)
    if llm is None or not getattr(llm, "is_available", lambda: False)():
        return regex_spec
    try:
        llm_spec = parse_with_llm(raw_prompt, llm)
    except Exception:
        return regex_spec
    if not llm_spec:
        return regex_spec
    # strong merge: regex wins for category/geo/keywords when specific
    if getattr(regex_spec, "target_category", "general") != "general":
        llm_spec.target_category = regex_spec.target_category
    if getattr(regex_spec, "industry", "") and len(regex_spec.industry) >= 3:
        llm_spec.industry = regex_spec.industry
    if getattr(regex_spec, "target_entity_types", None):
        llm_spec.target_entity_types = regex_spec.target_entity_types
    if getattr(regex_spec, "task_type", None):
        llm_spec.task_type = regex_spec.task_type
    rg = getattr(regex_spec, "geography", None)
    lg = getattr(llm_spec, "geography", None)
    if rg and lg:
        if rg.include_countries: lg.include_countries = rg.include_countries
        if rg.exclude_countries: lg.exclude_countries = rg.exclude_countries
        if rg.exclude_presence_countries: lg.exclude_presence_countries = rg.exclude_presence_countries
        lg.strict_mode = bool(lg.include_countries or lg.exclude_countries or lg.exclude_presence_countries)
    if getattr(regex_spec, "solution_keywords", None):
        llm_spec.solution_keywords = regex_spec.solution_keywords
    if getattr(regex_spec, "domain_keywords", None):
        llm_spec.domain_keywords = regex_spec.domain_keywords
    # ensure requested attrs preserved
    llm_spec.target_attributes = list(dict.fromkeys(list(getattr(regex_spec,'target_attributes',[]) or []) + list(getattr(llm_spec,'target_attributes',[]) or [])))
    return llm_spec
