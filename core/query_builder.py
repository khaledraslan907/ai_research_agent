from __future__ import annotations

from typing import List

from core.task_models import TaskSpec
from core.models import SearchQuery


class QueryBuilder:
    def __init__(self, max_core_queries: int = 10, max_fallback_queries: int = 4):
        self.max_core_queries = max_core_queries
        self.max_fallback_queries = max_fallback_queries

    def build_for_provider(self, task: TaskSpec, provider_name: str) -> List[SearchQuery]:
        provider_name = (provider_name or "").lower().strip()
        if provider_name == "exa":
            return self._build_exa_queries(task)
        return self._build_standard_queries(task)

    def _deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        out, seen = [], set()
        for q in queries:
            t = q.text.strip().lower()
            if t and t not in seen:
                seen.add(t)
                out.append(q)
        out.sort(key=lambda x: x.priority)
        return out

    def _geo(self, task: TaskSpec) -> str:
        inc = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        return inc[0] if inc else ""

    def _domain_terms(self, task: TaskSpec) -> List[str]:
        return [str(x).strip() for x in (task.domain_keywords or []) if str(x).strip()]

    def _solution_terms(self, task: TaskSpec) -> List[str]:
        return [str(x).strip() for x in (task.solution_keywords or []) if str(x).strip()]

    def _build_document_queries(self, task: TaskSpec) -> List[SearchQuery]:
        geo = self._geo(task)
        focus = (task.industry or "research topic").strip()
        domain_terms = self._domain_terms(task)
        solution_terms = self._solution_terms(task)
        q: List[SearchQuery] = []
        p = 1
        base_terms = " ".join([focus] + domain_terms[:2] + solution_terms[:2]).strip()
        geo_part = f" {geo}" if geo else ""
        patterns = [
            f"{base_terms} research paper{geo_part}",
            f"{base_terms} journal paper{geo_part}",
            f"{base_terms} conference paper{geo_part}",
            f"{base_terms} abstract DOI",
            f"site:onepetro.org {base_terms}",
            f"site:doi.org {base_terms}",
            f"site:sciencedirect.com {base_terms}",
            f"site:ieeexplore.ieee.org {base_terms}",
            f"site:researchgate.net {base_terms}",
        ]
        for text in patterns[: self.max_core_queries]:
            q.append(SearchQuery(text=text.strip(), priority=p, family="academic"))
            p += 1
        return self._deduplicate(q)

    def _build_food_software_queries(self, task: TaskSpec) -> List[SearchQuery]:
        geo = self._geo(task)
        geo_part = f" {geo}" if geo else ""
        texts = [
            f"food manufacturing software company{geo_part}",
            f"food processing software vendor{geo_part}",
            f"MES food manufacturing software{geo_part}",
            f"quality management food manufacturing software{geo_part}",
            f"production planning food manufacturing software{geo_part}",
            f"factory automation food industry software{geo_part}",
            f"food manufacturing software company{geo_part} -sap -oracle -microsoft",
        ]
        return self._deduplicate([SearchQuery(text=t, priority=i + 1, family="vertical") for i, t in enumerate(texts)])

    def _build_wireline_egypt_queries(self, task: TaskSpec) -> List[SearchQuery]:
        geo = self._geo(task) or "Egypt"
        texts = [
            f"wireline service company {geo} oil and gas",
            f"slickline service company {geo} oil and gas",
            f"well logging company {geo} oil and gas",
            f"wireline slickline well logging {geo}",
            f"Egypt oilfield service company wireline",
            f"Egypt well logging services company",
            f'"wireline" "Egypt" oil gas company',
            f'"slickline" "Egypt" oil gas company',
            f'"well logging" "Egypt" oil gas company',
            "شركات وايرلاين في مصر بترول",
        ]
        return self._deduplicate([SearchQuery(text=t, priority=i + 1, family="service") for i, t in enumerate(texts)])

    def _build_egyps_queries(self, task: TaskSpec) -> List[SearchQuery]:
        texts = [
            "EGYPS exhibitor wireline",
            "EGYPS exhibitor well logging",
            "EGYPS exhibitors wireline well logging",
            "site:egyps.com wireline exhibitor",
            "site:egyps.com well logging exhibitor",
            "site:egyps.com slickline exhibitor",
        ]
        return self._deduplicate([SearchQuery(text=t, priority=i + 1, family="event") for i, t in enumerate(texts)])

    def _build_tender_queries(self, task: TaskSpec) -> List[SearchQuery]:
        geo = self._geo(task)
        geo_part = f" {geo}" if geo else ""
        focus = (task.industry or "procurement").strip()
        texts = [
            f"{focus} tender{geo_part}",
            f"{focus} procurement{geo_part}",
            f"{focus} RFQ{geo_part}",
            f"{focus} invitation to tender{geo_part}",
            f"{focus} bid notice{geo_part}",
        ]
        return self._deduplicate([SearchQuery(text=t, priority=i + 1, family="tender") for i, t in enumerate(texts)])

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        if task.task_type == "document_research" or (task.target_entity_types and task.target_entity_types[0].lower() == "paper"):
            return self._build_document_queries(task)

        if (task.target_category or "").lower() == "software_company" and (task.industry or "").lower().strip() == "food manufacturing":
            return self._build_food_software_queries(task)

        dks = " ".join([x.lower() for x in self._domain_terms(task)])
        if (task.target_category or "").lower() == "service_company" and "egypt" in [c.lower() for c in (task.geography.include_countries or [])] and any(x in dks for x in ["wireline", "slickline", "well logging"]):
            return self._build_wireline_egypt_queries(task)

        if task.task_type == "market_research" and "egyps" in (task.raw_prompt or "").lower():
            return self._build_egyps_queries(task)

        if task.task_type == "market_research" and ((task.target_entity_types and task.target_entity_types[0].lower() == "tender") or "tender" in (task.raw_prompt or "").lower()):
            return self._build_tender_queries(task)

        queries: List[SearchQuery] = []
        focus = (task.industry or "industry").strip()
        category = (task.target_category or "general").strip().lower()
        geo = self._geo(task)
        domain_terms = self._domain_terms(task)
        solution_terms = self._solution_terms(task)
        entity_type = (task.target_entity_types[0] if task.target_entity_types else "company").lower()
        base_entity = "software company" if category == "software_company" else ("service company" if category == "service_company" else ("tender" if entity_type == "tender" else "company"))
        geo_part = f" {geo}" if geo else ""
        p = 1
        templates = [f"{focus} {base_entity}{geo_part}"]
        if domain_terms:
            templates.append(f"{domain_terms[0]} {base_entity}{geo_part}")
        if solution_terms:
            templates.append(f"{solution_terms[0]} {base_entity}{geo_part}")
        if domain_terms and solution_terms:
            templates.append(f"{focus} {solution_terms[0]} {domain_terms[0]} {base_entity}{geo_part}")
        for text in templates[: self.max_core_queries]:
            queries.append(SearchQuery(text=text.strip(), priority=p, family="core"))
            p += 1
        return self._deduplicate(queries)

    def _build_exa_queries(self, task: TaskSpec) -> List[SearchQuery]:
        if task.task_type == "document_research" or (task.target_entity_types and task.target_entity_types[0].lower() == "paper"):
            focus = (task.industry or "research").strip()
            qs = [
                SearchQuery(text=f"academic papers and publications about {focus}", priority=1, family="semantic", provider_hint="exa"),
                SearchQuery(text=f"research papers, authors, abstracts, and DOI for {focus}", priority=2, family="semantic", provider_hint="exa"),
            ]
            return self._deduplicate(qs)

        if task.task_type == "market_research" and "egyps" in (task.raw_prompt or "").lower():
            qs = [
                SearchQuery(text="EGYPS exhibitors related to wireline and well logging", priority=1, family="semantic", provider_hint="exa"),
                SearchQuery(text="official EGYPS exhibitor pages for wireline and well logging companies", priority=2, family="semantic", provider_hint="exa"),
            ]
            return self._deduplicate(qs)

        focus = (task.industry or "industry").strip()
        category = (task.target_category or "general").strip().lower()
        geo = self._geo(task) or "global markets"
        entity_phrase = "software and digital companies" if category == "software_company" else ("service companies" if category == "service_company" else "companies")
        qs = [SearchQuery(text=f"{entity_phrase} serving the {focus} industry in {geo}", priority=1, family="semantic", provider_hint="exa")]
        return self._deduplicate(qs)
