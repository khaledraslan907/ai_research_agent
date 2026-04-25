from __future__ import annotations

from typing import Dict, List

from core.models import SearchQuery
from core.task_models import TaskSpec
from core.prompt_templates import QUERY_PLAN_PROMPT
from core.query_builder import QueryBuilder


def plan_queries(task_spec: TaskSpec, llm=None) -> Dict[str, List[SearchQuery]]:
    builder = QueryBuilder(max_core_queries=8, max_fallback_queries=4)
    templates = {
        provider: builder.build_for_provider(task_spec, provider)
        for provider in ("ddg", "exa", "tavily", "serpapi")
    }

    # For papers/people, template planning is usually strong enough.
    if task_spec.task_type in {"people_search", "document_research"}:
        return templates

    if llm and getattr(llm, "is_available", lambda: False)():
        try:
            llm_result = _plan_with_llm(task_spec, llm)
            if llm_result and any(llm_result.values()):
                merged = {}
                for provider in ("ddg", "exa", "tavily", "serpapi"):
                    llm_q = llm_result.get(provider, [])
                    tmpl_q = templates.get(provider, [])
                    seen = {q.text.lower().strip() for q in llm_q}
                    merged[provider] = llm_q + [q for q in tmpl_q if q.text.lower().strip() not in seen]
                return merged
        except Exception:
            pass

    return templates


def _plan_with_llm(task_spec: TaskSpec, llm) -> Dict[str, List[SearchQuery]]:
    topic = (task_spec.industry or task_spec.raw_prompt or "general").strip()
    ent_type = (task_spec.target_entity_types or ["company"])[0]
    inc_c = task_spec.geography.include_countries or []
    exc_c = task_spec.geography.exclude_countries or []
    exc_presence = task_spec.geography.exclude_presence_countries or []
    solution_terms = list(getattr(task_spec, "solution_keywords", []) or [])
    domain_terms = list(getattr(task_spec, "domain_keywords", []) or [])
    commercial_terms = []
    ci = (getattr(task_spec, "commercial_intent", "general") or "general").lower()
    if ci == "agent_or_distributor":
        commercial_terms = ["distributor", "local representative", "reseller", "channel partner"]
    elif ci == "reseller":
        commercial_terms = ["reseller", "partner program"]
    elif ci == "partner":
        commercial_terms = ["partner", "alliance", "channel partner"]

    prompt = QUERY_PLAN_PROMPT.format(
        topic_description=topic,
        task_type=task_spec.task_type,
        entity_category=task_spec.target_category or "general",
        entity_type=ent_type,
        include_countries=", ".join(inc_c) if inc_c else "any",
        exclude_countries=", ".join(exc_c) if exc_c else "none",
        exclude_presence_countries=", ".join(exc_presence) if exc_presence else "none",
        solution_keywords=", ".join(solution_terms) if solution_terms else "none",
        domain_keywords=", ".join(domain_terms) if domain_terms else "none",
        commercial_intent=(getattr(task_spec, "commercial_intent", "general") or "general"),
        topic_words=" ".join((topic.split() + solution_terms + domain_terms + commercial_terms)[:12]),
    )
    raw = llm.generate_json(prompt, timeout=40)
    if not raw or not isinstance(raw, dict):
        return {}

    result: Dict[str, List[SearchQuery]] = {}
    for provider in ("ddg", "exa", "tavily", "serpapi"):
        items = raw.get(provider, [])
        if not isinstance(items, list):
            continue
        queries = []
        for i, item in enumerate(items):
            text = item if isinstance(item, str) else (item.get("text") or item.get("query") or "")
            text = str(text).strip()
            if text and len(text) > 4:
                queries.append(SearchQuery(text=text, priority=i + 1, family="llm_generated", provider_hint=provider))
        result[provider] = queries
    return result
