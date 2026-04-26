from __future__ import annotations
from typing import List
from core.task_models import TaskSpec
from core.models import SearchQuery

class QueryBuilder:
    def __init__(self, max_core_queries: int = 8, max_fallback_queries: int = 4):
        self.max_core_queries = max_core_queries
        self.max_fallback_queries = max_fallback_queries

    def build_for_provider(self, task: TaskSpec, provider_name: str) -> List[SearchQuery]:
        return self._build_standard_queries(task)

    def _deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        out=[]; seen=set()
        for q in queries:
            t=q.text.strip().lower()
            if t and t not in seen:
                seen.add(t); out.append(q)
        return out

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []
        p=1
        entity = (task.target_entity_types[0] if task.target_entity_types else "company").lower()
        category = (task.target_category or "general").lower()
        inc = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        geo = inc[0] if inc else ""

        # food manufacturing software -> niche queries, not generic ERP
        if category == "software_company" and (task.industry or "").lower() == "food manufacturing":
            geo_part = f" {geo}" if geo else ""
            for q in [
                f'food manufacturing software company{geo_part}',
                f'food processing software vendor{geo_part}',
                f'MES food manufacturing software{geo_part}',
                f'factory software for food manufacturing{geo_part}',
                f'food production software company{geo_part}',
            ]:
                queries.append(SearchQuery(text=q.strip(), priority=p, family="vertical")); p += 1
            return self._deduplicate(queries)

        # EGYPS exhibitors -> event-aware company discovery
        if task.task_type == "market_research" and "wireline" in " ".join(task.domain_keywords).lower():
            for q in [
                'EGYPS exhibitor wireline',
                'EGYPS exhibitor well logging',
                'EGYPS exhibitors wireline well logging',
                'site:egyps.com wireline exhibitor',
                'site:egyps.com well logging exhibitor',
            ]:
                queries.append(SearchQuery(text=q, priority=p, family="event")); p += 1
            return self._deduplicate(queries)

        focus = (task.industry or "").strip() or "industry"
        base_entity = "software company" if category == "software_company" else ("service company" if category == "service_company" else "company")
        q = f"{focus} {base_entity} {geo}".strip()
        queries.append(SearchQuery(text=q, priority=1, family="core"))
        return self._deduplicate(queries)
