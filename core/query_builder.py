from __future__ import annotations

import re
from typing import List

from core.task_models import TaskSpec
from core.models import SearchQuery

COUNTRY_EXCLUSION_HINTS = {
    "usa": ['-"usa"', '-"united states"', '-"texas"', '-"houston"', '-"new york"'],
    "egypt": ['-"egypt"', '-"cairo"', '-"alexandria"', '-"egyptian"'],
}

SOLUTION_SYNONYMS = {
    "machine learning": ["machine learning", "ML"],
    "artificial intelligence": ["artificial intelligence", "AI"],
    "ai": ["AI"],
    "analytics": ["analytics", "data analytics"],
    "monitoring": ["monitoring", "remote monitoring"],
    "optimization": ["optimization", "optimisation"],
    "automation": ["automation"],
    "iot": ["IoT", "internet of things"],
    "scada": ["SCADA"],
    "digital twin": ["digital twin"],
    "predictive maintenance": ["predictive maintenance"],
}

DOMAIN_SYNONYMS = {
    "wireline": ["wireline", "wire line"],
    "slickline": ["slickline", "slick line"],
    "e-line": ["e-line", "electric line"],
    "well logging": ["well logging", "well log"],
    "coiled tubing": ["coiled tubing"],
    "well testing": ["well testing", "well test"],
    "stimulation": ["well stimulation", "stimulation"],
    "cementing": ["cementing"],
    "mud logging": ["mud logging", "mud logger"],
    "drilling fluids": ["drilling fluids", "mud chemicals"],
    "directional drilling": ["directional drilling"],
    "drilling optimization": ["drilling optimization", "drilling optimisation"],
    "artificial lift": ["artificial lift"],
    "production monitoring": ["production monitoring"],
    "production optimization": ["production optimization", "production optimisation"],
    "well surveillance": ["well surveillance"],
    "inspection": ["inspection", "NDT"],
    "asset integrity": ["asset integrity"],
    "pipeline monitoring": ["pipeline monitoring"],
    "event exhibitors": ["exhibitor list", "event exhibitors", "conference exhibitors"],
    "tender": ["tender", "procurement", "vendor list"],
}

COMMERCIAL_TERMS = {
    "agent_or_distributor": ["distributor", "local representative", "channel partner", "reseller"],
    "reseller": ["reseller", "channel partner", "partner program"],
    "partner": ["partner", "channel partner", "alliance"],
}

AR_COUNTRY_MAP = {
    "egypt": "مصر",
    "saudi arabia": "السعودية",
    "united arab emirates": "الإمارات",
    "united kingdom": "بريطانيا",
    "norway": "النرويج",
}

AR_DOMAIN_MAP = {
    "wireline": "وايرلاين",
    "slickline": "سليك لاين",
    "well logging": "تسجيل الآبار",
    "coiled tubing": "كويلد تيوبنج",
    "well testing": "اختبار الآبار",
    "stimulation": "تحفيز الآبار",
    "cementing": "إسمنت آبار",
    "mud logging": "ماد لوجنج",
    "inspection": "تفتيش",
}


class QueryBuilder:
    def __init__(self, max_core_queries: int = 8, max_fallback_queries: int = 4):
        self.max_core_queries = max_core_queries
        self.max_fallback_queries = max_fallback_queries

    def build_for_provider(self, task: TaskSpec, provider_name: str) -> List[SearchQuery]:
        provider_name = (provider_name or "").lower().strip()
        if provider_name == "exa":
            return self._build_exa_queries(task)
        if provider_name == "tavily":
            return self._build_tavily_queries(task)
        if provider_name == "serpapi":
            return self._build_serpapi_queries(task)
        return self._build_standard_queries(task)

    def _has_arabic(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text or ""))

    def _solution_terms(self, task: TaskSpec) -> List[str]:
        out, seen = [], set()
        for kw in list(getattr(task, "solution_keywords", []) or []):
            key = str(kw).strip().lower()
            for s in SOLUTION_SYNONYMS.get(key, [key]):
                if s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
        return out[:6]

    def _domain_terms(self, task: TaskSpec) -> List[str]:
        out, seen = [], set()
        for kw in list(getattr(task, "domain_keywords", []) or []):
            key = str(kw).strip().lower()
            for s in DOMAIN_SYNONYMS.get(key, [key]):
                if s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
        return out[:6]

    def _commercial_terms(self, task: TaskSpec) -> List[str]:
        return list(COMMERCIAL_TERMS.get((getattr(task, "commercial_intent", "general") or "general").strip().lower(), []))

    def _combined_excluded_geo(self, a: List[str], b: List[str]) -> List[str]:
        out, seen = [], set()
        for item in (a or []) + (b or []):
            s = str(item or "").strip().lower()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _country_exclusion_tokens(self, country: str) -> List[str]:
        return list(COUNTRY_EXCLUSION_HINTS.get((country or "").strip().lower(), [f'-"{country}"']))

    def _entity_phrase(self, task: TaskSpec) -> str:
        if task.task_type == "document_research":
            return "research papers"
        if task.task_type == "people_search":
            return "LinkedIn profiles"
        cat = (task.target_category or "general").strip().lower()
        if cat == "software_company":
            return "software company"
        if cat == "service_company":
            return "service company"
        return "company"

    def _base_terms(self, task: TaskSpec) -> dict:
        focus = (task.industry or task.raw_prompt or "industry").strip()
        include_countries = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        exclude_countries = [c.strip() for c in (task.geography.exclude_countries or []) if c.strip()]
        exclude_presence_countries = [c.strip() for c in (task.geography.exclude_presence_countries or []) if c.strip()]
        return {
            "focus": focus,
            "entity_phrase": self._entity_phrase(task),
            "include_countries": include_countries,
            "exclude_countries": exclude_countries,
            "exclude_presence_countries": exclude_presence_countries,
            "solution_terms": self._solution_terms(task),
            "domain_terms": self._domain_terms(task),
            "commercial_terms": self._commercial_terms(task),
            "raw_prompt": (task.raw_prompt or "").strip(),
            "is_arabic": self._has_arabic(task.raw_prompt or "") or any(self._has_arabic(t) for t in [task.industry or ""]),
        }

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        t = self._base_terms(task)
        queries: List[SearchQuery] = []
        p = 1

        if t["raw_prompt"]:
            queries.append(SearchQuery(text=t["raw_prompt"], priority=p, family="exact")); p += 1

        focus = t["focus"]
        entity = t["entity_phrase"]
        include_countries = t["include_countries"]
        solution_terms = t["solution_terms"]
        domain_terms = t["domain_terms"]
        commercial_terms = t["commercial_terms"]

        if include_countries:
            for geo in include_countries[:3]:
                queries.append(SearchQuery(text=f"{focus} {entity} {geo}".strip(), priority=p, family="geo")); p += 1
                if domain_terms:
                    queries.append(SearchQuery(text=f"{domain_terms[0]} {entity} {geo} {focus}".strip(), priority=p, family="domain_geo")); p += 1
        else:
            queries.append(SearchQuery(text=f"{focus} {entity}".strip(), priority=p, family="core")); p += 1

        for dt in domain_terms[:3]:
            q = f"{dt} {entity}"
            if include_countries:
                q += f" {include_countries[0]}"
            if focus and dt.lower() not in focus.lower():
                q += f" {focus}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="domain")); p += 1

        for st in solution_terms[:2]:
            q = f"{st} {entity}"
            if include_countries:
                q += f" {include_countries[0]}"
            if focus and st.lower() not in focus.lower():
                q += f" {focus}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="solution")); p += 1

        if task.task_type == "document_research":
            queries.append(SearchQuery(text=f"{focus} paper doi abstract", priority=p, family="paper_meta")); p += 1
            queries.append(SearchQuery(text=f"{focus} journal article", priority=p, family="paper_journal")); p += 1
        elif task.task_type == "people_search":
            geo = include_countries[0] if include_countries else ""
            queries.append(SearchQuery(text=f'site:linkedin.com/in {focus} {geo}'.strip(), priority=p, family="linkedin_site")); p += 1
        else:
            for ct in commercial_terms[:2]:
                queries.append(SearchQuery(text=f"{focus} {entity} {ct}".strip(), priority=p, family="commercial")); p += 1

        # bilingual recall
        if t["is_arabic"] or (include_countries and include_countries[0] == "egypt"):
            ar_geo = AR_COUNTRY_MAP.get(include_countries[0], "") if include_countries else ""
            if domain_terms:
                for dt in domain_terms[:2]:
                    ar_dt = AR_DOMAIN_MAP.get(dt.lower(), dt)
                    q = f"{ar_dt} شركات خدمات البترول {ar_geo}".strip()
                    queries.append(SearchQuery(text=q, priority=p, family="arabic_domain")); p += 1
            elif include_countries:
                queries.append(SearchQuery(text=f"شركات خدمات البترول في {ar_geo}".strip(), priority=p, family="arabic_geo")); p += 1
            if task.task_type == "document_research":
                queries.append(SearchQuery(text=f"أبحاث {focus}".strip(), priority=p, family="arabic_papers")); p += 1

        # exclusions appended on one query, not all
        excluded_geo = self._combined_excluded_geo(t["exclude_countries"], t["exclude_presence_countries"])
        if task.task_type != "people_search" and excluded_geo:
            suffixes = []
            for country in excluded_geo[:3]:
                suffixes.extend(self._country_exclusion_tokens(country))
            suffix = " ".join(suffixes)
            queries.append(SearchQuery(text=f"{focus} {entity} {suffix}".strip(), priority=p, family="exclude")); p += 1

        return self._deduplicate(queries)[: max(self.max_core_queries + self.max_fallback_queries, 10)]

    def _build_exa_queries(self, task: TaskSpec) -> List[SearchQuery]:
        t = self._base_terms(task)
        focus = t["focus"]
        include = ", ".join(t["include_countries"][:3]) if t["include_countries"] else "global markets"
        entity_phrase = self._entity_phrase(task)

        templates = [f"{entity_phrase} relevant to {focus} in {include}"]
        if t["domain_terms"]:
            templates.append(f"{entity_phrase} focused on {', '.join(t['domain_terms'][:3])} in {include}")
        if t["solution_terms"]:
            templates.append(f"{entity_phrase} focused on {', '.join(t['solution_terms'][:3])} in {include}")
        if t["exclude_presence_countries"]:
            templates.append(f"{entity_phrase} without offices or branch presence in {', '.join(t['exclude_presence_countries'][:2])}")
        if t["commercial_terms"]:
            templates.append(f"{focus} vendors with distributors, resellers, partners, or representatives")
        if task.task_type == "document_research":
            templates.append(f"research papers about {focus} with abstract and doi")
        if task.task_type == "people_search":
            templates.append(f"LinkedIn professionals working on {focus} in {include}")

        queries = [SearchQuery(text=q, priority=i + 1, family="semantic", provider_hint="exa") for i, q in enumerate(templates)]
        return self._deduplicate(queries)

    def _build_tavily_queries(self, task: TaskSpec) -> List[SearchQuery]:
        t = self._base_terms(task)
        queries = []
        focus = t["focus"]
        geo = t["include_countries"][0] if t["include_countries"] else ""
        if task.task_type == "document_research":
            queries.append(SearchQuery(text=f"What are strong papers about {focus}?", priority=1, family="question", provider_hint="tavily"))
            queries.append(SearchQuery(text=f"Which journals or papers cover {focus}?", priority=2, family="question", provider_hint="tavily"))
        elif task.task_type == "people_search":
            queries.append(SearchQuery(text=f"Who are LinkedIn professionals working on {focus} {geo}".strip(), priority=1, family="question", provider_hint="tavily"))
        else:
            entity = self._entity_phrase(task)
            queries.append(SearchQuery(text=f"Which {entity}s work on {focus} in {geo}".strip(), priority=1, family="question", provider_hint="tavily"))
            if t["domain_terms"]:
                queries.append(SearchQuery(text=f"Which companies provide {', '.join(t['domain_terms'][:2])} in {geo}".strip(), priority=2, family="question", provider_hint="tavily"))
        return self._deduplicate(queries)

    def _build_serpapi_queries(self, task: TaskSpec) -> List[SearchQuery]:
        t = self._base_terms(task)
        geo = t["include_countries"][0] if t["include_countries"] else ""
        focus = t["focus"]
        queries = []
        neg = " ".join(sum((self._country_exclusion_tokens(c) for c in self._combined_excluded_geo(t["exclude_countries"], t["exclude_presence_countries"])[:2]), []))
        if task.task_type == "people_search":
            queries.append(SearchQuery(text=f'site:linkedin.com/in {focus} {geo} {neg}'.strip(), priority=1, family="linkedin_google", provider_hint="serpapi"))
            queries.append(SearchQuery(text=f'site:linkedin.com/in ({focus}) ({geo})'.strip(), priority=2, family="linkedin_google", provider_hint="serpapi"))
        elif task.task_type == "document_research":
            queries.append(SearchQuery(text=f'"{focus}" filetype:pdf {neg}'.strip(), priority=1, family="pdf_google", provider_hint="serpapi"))
            queries.append(SearchQuery(text=f'"{focus}" DOI OR journal OR paper {neg}'.strip(), priority=2, family="paper_google", provider_hint="serpapi"))
        else:
            entity = self._entity_phrase(task)
            queries.append(SearchQuery(text=f'"{focus}" "{entity}" {geo} {neg}'.strip(), priority=1, family="core_google", provider_hint="serpapi"))
            if t["domain_terms"]:
                queries.append(SearchQuery(text=f'"{t["domain_terms"][0]}" "{entity}" {geo} {neg}'.strip(), priority=2, family="domain_google", provider_hint="serpapi"))
        return self._deduplicate(queries)

    def _deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        seen = set()
        out = []
        for q in queries:
            key = q.text.lower().strip()
            if key in seen or len(key) < 4:
                continue
            seen.add(key)
            out.append(q)
        out.sort(key=lambda x: (x.priority, x.family, x.text))
        return out
