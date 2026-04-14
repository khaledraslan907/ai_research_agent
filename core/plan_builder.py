"""
plan_builder.py
================
Builds the ExecutionPlan from TaskSpec + ProviderSettings.

Auto-scales mode based on max_results:
  > 60 results requested  → upgrades Balanced to Deep
  > 30 results requested  → upgrades Fast to Balanced

People search gets separate query budgets (EXA-heavy, no Tavily).
"""
from __future__ import annotations

from core.task_models import TaskSpec, ExecutionPlan


def build_execution_plan(task: TaskSpec, provider_settings=None) -> ExecutionPlan:
    ps   = provider_settings
    mode = (task.mode or "Balanced").lower()
    max_r = getattr(task, "max_results", 25) or 25

    # ── Auto-upgrade mode ──────────────────────────────────────────────────
    if max_r > 60 and mode == "balanced":
        mode = "deep"
    elif max_r > 30 and mode == "fast":
        mode = "balanced"

    # ── Strategy label ─────────────────────────────────────────────────────
    strategy_name = "general_search"
    if task.task_type == "entity_enrichment":
        strategy_name = "enrichment_first"
    elif task.task_type == "similar_entity_expansion":
        strategy_name = "similar_entity_search"
    elif task.task_type == "market_research":
        strategy_name = "market_research"
    elif task.task_type == "document_research":
        strategy_name = "document_research"
    elif task.task_type == "people_search":
        strategy_name = "linkedin_people_search"
    elif task.target_category == "service_company":
        strategy_name = "service_vendor_search"
    elif task.target_category == "software_company":
        strategy_name = "software_vendor_search"

    # ── Provider activation ────────────────────────────────────────────────
    use_ddg  = ps.use_ddg  if ps else True
    use_exa  = (ps.use_exa if ps else True) and task.task_type in {
        "entity_discovery", "similar_entity_expansion",
        "market_research", "document_research", "people_search",
    }
    use_tavily    = (ps.use_tavily   if ps else False) and mode in {"balanced", "deep"}
    use_serpapi   = (ps.use_serpapi  if ps else False) and mode == "deep"
    use_firecrawl = ps.use_firecrawl if ps else False
    # People search: SerpApi useful in any mode (site:linkedin.com)
    if task.task_type == "people_search":
        use_serpapi = (ps.use_serpapi if ps else False)

    use_exa_find_similar       = use_exa and task.task_type == "similar_entity_expansion"
    use_local_llm_parser       = getattr(task, "use_local_llm", False)
    use_local_llm_classifier   = use_local_llm_parser and task.task_type in {
        "entity_discovery", "entity_enrichment", "market_research",
    }
    use_cloud_llm_batch_verify = (
        getattr(task, "use_cloud_llm", False) and mode in {"balanced", "deep"}
    )

    # ── Query counts and candidate caps ────────────────────────────────────
    if task.task_type == "people_search":
        # EXA-heavy: domain-scoped to linkedin.com/in — no Tavily
        if mode == "fast":
            max_q = {"ddg": 4, "exa": 10, "tavily": 0, "serpapi": 4}
        elif mode == "deep":
            max_q = {"ddg": 6, "exa": 20, "tavily": 0, "serpapi": 8}
        else:
            max_q = {"ddg": 4, "exa": 15, "tavily": 0, "serpapi": 5}
        max_candidates = max(200, max_r * 5)
        use_tavily = False

    elif mode == "fast":
        max_q = {"ddg": 4, "exa": 2, "tavily": 0, "serpapi": 0}
        max_candidates = max(30, max_r * 3)

    elif mode == "deep":
        max_q = {"ddg": 12, "exa": 8, "tavily": 6, "serpapi": 4}
        max_candidates = max(200, max_r * 4)

    else:  # balanced
        max_q = {"ddg": 7, "exa": 5, "tavily": 4, "serpapi": 0}
        max_candidates = max(100, max_r * 3)

    # Zero out inactive providers
    if not use_exa:     max_q["exa"]     = 0
    if not use_tavily:  max_q["tavily"]  = 0
    if not use_serpapi: max_q["serpapi"] = 0

    return ExecutionPlan(
        strategy_name=strategy_name,
        use_ddg=use_ddg,
        use_exa=use_exa,
        use_tavily=use_tavily,
        use_serpapi=use_serpapi,
        use_firecrawl=use_firecrawl,
        use_exa_find_similar=use_exa_find_similar,
        use_local_llm_parser=use_local_llm_parser,
        use_local_llm_classifier=use_local_llm_classifier,
        use_cloud_llm_batch_verify=use_cloud_llm_batch_verify,
        stop_when_enough_valid=False,
        max_queries_per_provider=max_q,
        max_candidates_to_process=max_candidates,
    )
