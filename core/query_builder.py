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
        out = []
        seen = set()
        for q in queries:
            t = q.text.strip().lower()
            if t and t not in seen:
                seen.add(t)
                out.append(q)
        return out

    def _add_queries(self, queries: List[SearchQuery], texts: List[str], family: str, priority_start: int) -> int:
        p = priority_start
        for q in texts:
            queries.append(SearchQuery(text=q.strip(), priority=p, family=family))
            p += 1
        return p

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []
        p = 1
        entity = (task.target_entity_types[0] if task.target_entity_types else "company").lower()
        category = (task.target_category or "general").lower()
        inc = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        geo = inc[0] if inc else ""
        focus = (task.industry or "").strip() or "industry"
        dks = [str(x).strip().lower() for x in (getattr(task, "domain_keywords", []) or []) if str(x).strip()]
        sks = [str(x).strip().lower() for x in (getattr(task, "solution_keywords", []) or []) if str(x).strip()]
        raw_prompt = (getattr(task, "raw_prompt", "") or "").lower()

        # Norway digital oil and gas companies
        if category == "software_company" and focus.lower() == "oil and gas" and geo.lower() == "norway":
            texts = [
                "digital oil and gas software companies Norway",
                "Norway oil and gas software company",
                "Norwegian digital oil and gas technology company",
                "Norway energy software company oil and gas",
                "Norway oil and gas analytics platform company",
                "Norway oil and gas automation software company",
                "Norway offshore software company oil and gas",
                "Stavanger oil and gas software company",
            ]
            p = self._add_queries(queries, texts, "vertical", p)
            return self._deduplicate(queries)

        # Food manufacturing software
        if category == "software_company" and focus.lower() == "food manufacturing":
            geo_part = f" {geo}" if geo else ""
            texts = [
                f"food manufacturing software company{geo_part}",
                f"food processing software vendor{geo_part}",
                f"MES food manufacturing software{geo_part}",
                f"factory software for food manufacturing{geo_part}",
                f"food production software company{geo_part}",
            ]
            p = self._add_queries(queries, texts, "vertical", p)
            return self._deduplicate(queries)

        # Egypt wireline / logging companies
        if geo.lower() == "egypt" and any(x in dks for x in ["wireline", "slickline", "well logging"]):
            texts = [
                "wireline company Egypt",
                "slickline company Egypt",
                "well logging company Egypt",
                "wireline slickline services Egypt",
                "cased hole logging Egypt company",
                "open hole logging Egypt company",
                "oilfield wireline service Egypt",
            ]
            p = self._add_queries(queries, texts, "niche_geo", p)
            return self._deduplicate(queries)

        # EGYPS exhibitors
        if task.task_type == "market_research" and ("egyps" in raw_prompt or "exhibitor" in raw_prompt):
            texts = [
                "EGYPS exhibitor wireline",
                "EGYPS exhibitor well logging",
                "EGYPS exhibitors wireline well logging",
                "site:egyps.com wireline exhibitor",
                "site:egyps.com well logging exhibitor",
            ]
            p = self._add_queries(queries, texts, "event", p)
            return self._deduplicate(queries)

        # Academic papers
        if task.task_type == "document_research" or entity == "paper":
            topic = focus
            texts = [
                f'research paper "{topic}"',
                f'journal paper "{topic}"',
                f'conference paper "{topic}"',
                f'"{topic}" doi',
                f'"{topic}" abstract authors',
                f'site:sciencedirect.com "{topic}"',
                f'site:springer.com "{topic}"',
                f'site:onepetro.org "{topic}"',
                f'site:scholar.google.com "{topic}"',
            ]
            p = self._add_queries(queries, texts, "document", p)
            return self._deduplicate(queries)

        # Tenders
        if entity == "tender" or task.task_type == "market_research":
            geo_part = f" {geo}" if geo else ""
            texts = [
                f"{focus} tender{geo_part}",
                f"{focus} procurement{geo_part}",
                f"{focus} RFQ{geo_part}",
                f"{focus} RFP{geo_part}",
                f"{focus} invitation to tender{geo_part}",
            ]
            p = self._add_queries(queries, texts, "tender", p)
            return self._deduplicate(queries)

        base_entity = (
            "software company" if category == "software_company"
            else "service company" if category == "service_company"
            else "product" if entity == "product"
            else "company"
        )
        geo_part = f" {geo}" if geo else ""
        texts = [f"{focus} {base_entity}{geo_part}".strip()]
        p = self._add_queries(queries, texts, "core", p)
        return self._deduplicate(queries)
