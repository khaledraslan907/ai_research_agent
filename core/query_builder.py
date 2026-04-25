from __future__ import annotations

from typing import List

from core.task_models import TaskSpec
from core.models import SearchQuery

COUNTRY_TLD_HINTS = {
    "egypt": "eg", "saudi arabia": "sa", "united arab emirates": "ae", "qatar": "qa",
    "oman": "om", "kuwait": "kw", "bahrain": "bh", "india": "in", "united kingdom": "uk",
    "france": "fr", "germany": "de", "italy": "it", "spain": "es", "canada": "ca",
    "australia": "au", "brazil": "br", "netherlands": "nl", "norway": "no", "sweden": "se",
    "denmark": "dk",
}

COUNTRY_EXCLUSION_HINTS = {
    "usa": ['-"usa"', '-"united states"', '-"texas"', '-"houston"', '-"new york"'],
    "united states": ['-"usa"', '-"united states"', '-"texas"', '-"houston"', '-"new york"'],
    "egypt": ['-"egypt"', '-"cairo"', '-"alexandria"', '-"egyptian"'],
}

SOLUTION_SYNONYMS = {
    "machine learning": ["machine learning", "ML"],
    "artificial intelligence": ["artificial intelligence"],
    "ai": ["ai"],
    "analytics": ["analytics"],
    "monitoring": ["monitoring"],
    "optimization": ["optimization"],
    "automation": ["automation"],
    "iot": ["iot"],
    "scada": ["scada"],
    "digital twin": ["digital twin"],
    "predictive maintenance": ["predictive maintenance"],
}

DOMAIN_SYNONYMS = {
    "esp": ["ESP", "electrical submersible pump"],
    "virtual flow metering": ["virtual flow metering", "virtual flow meter"],
    "well performance": ["well performance"],
    "artificial lift": ["artificial lift"],
    "production optimization": ["production optimization"],
    "well surveillance": ["well surveillance"],
    "multiphase metering": ["multiphase metering"],
    "flow assurance": ["flow assurance"],
    "production monitoring": ["production monitoring"],
    "reservoir simulation": ["reservoir simulation"],
    "reservoir modeling": ["reservoir modeling"],
    "drilling optimization": ["drilling optimization"],
    "production engineering": ["production engineering"],
}

COMMERCIAL_TERMS = {
    "agent_or_distributor": ["distributor", "local representative", "channel partner", "reseller"],
    "reseller": ["reseller", "channel partner", "partner program"],
    "partner": ["partner", "channel partner", "partner program", "alliance"],
}


class QueryBuilder:
    def __init__(self, max_core_queries: int = 6, max_fallback_queries: int = 3):
        self.max_core_queries = max_core_queries
        self.max_fallback_queries = max_fallback_queries

    def build_for_provider(self, task: TaskSpec, provider_name: str) -> List[SearchQuery]:
        if (provider_name or "").lower().strip() == "exa":
            return self._build_exa_queries(task)
        return self._build_standard_queries(task)

    def _solution_terms(self, task: TaskSpec) -> List[str]:
        out = []
        seen = set()
        for kw in (getattr(task, "solution_keywords", []) or []):
            key = str(kw).strip().lower()
            for s in SOLUTION_SYNONYMS.get(key, [key]):
                if s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
        return out[:6]

    def _domain_terms(self, task: TaskSpec) -> List[str]:
        out = []
        seen = set()
        for kw in (getattr(task, "domain_keywords", []) or []):
            key = str(kw).strip().lower()
            for s in DOMAIN_SYNONYMS.get(key, [key]):
                if s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
        return out[:6]

    def _commercial_terms(self, task: TaskSpec) -> List[str]:
        return list(COMMERCIAL_TERMS.get((getattr(task, "commercial_intent", "general") or "general").strip().lower(), []))

    def _combined_excluded_geo(self, a: List[str], b: List[str]) -> List[str]:
        out = []
        seen = set()
        for item in (a or []) + (b or []):
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    def _country_exclusion_tokens(self, country: str) -> List[str]:
        return list(COUNTRY_EXCLUSION_HINTS.get((country or "").strip().lower(), [f'-"{country}"']))

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []

        focus = (task.industry or "").strip() or "industry"
        category = (task.target_category or "general").strip().lower()
        entity_type = (task.target_entity_types[0] if task.target_entity_types else "company").lower()
        prompt_lower = str(getattr(task, "raw_prompt", "") or "").lower()

        include_countries = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        exclude_countries = [c.strip() for c in (task.geography.exclude_countries or []) if c.strip()]
        exclude_presence_countries = [c.strip() for c in (task.geography.exclude_presence_countries or []) if c.strip()]

        solution_terms = self._solution_terms(task)
        domain_terms = self._domain_terms(task)
        commercial_terms = self._commercial_terms(task)

        # Tender / procurement queries
        if entity_type == "tender":
            p = 1
            for geo in include_countries[:6] or [""]:
                gp = f" {geo}" if geo else ""
                for phrase in ["tender", "procurement", "rfq", "rfp", "invitation to tender"]:
                    queries.append(SearchQuery(text=f"{focus} {phrase}{gp}".strip(), priority=p, family="tender")); p += 1
            return self._deduplicate(queries)

        # Exhibitor/event-style company discovery
        if "egyps" in prompt_lower or "exhibitor" in prompt_lower:
            p = 1
            topic = " ".join(domain_terms[:2]) or focus or "oil and gas"
            queries.extend([
                SearchQuery(text=f"EGYPS exhibitors {topic}", priority=p, family="event"),
                SearchQuery(text=f"EGYPS exhibitor list {topic}", priority=p+1, family="event"),
                SearchQuery(text=f"Egypt Energy Show exhibitors {topic}", priority=p+2, family="event"),
                SearchQuery(text=f"EGYPS {topic} exhibitors website", priority=p+3, family="event"),
            ])
            return self._deduplicate(queries)


        base_entity = "software company" if category == "software_company" else ("service company" if category == "service_company" else "company")

        p = 1
        for geo in (include_countries[:4] if include_countries else []):
            q = f"{focus} {base_entity}"
            if solution_terms:
                q += f" {solution_terms[0]}"
            if domain_terms:
                q += f" {domain_terms[0]}"
            q += f" {geo}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="core"))
            p += 1

        for dt in domain_terms[:3]:
            q = f"{focus} {dt} {base_entity}"
            if include_countries:
                q += f" {include_countries[0]}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="domain"))
            p += 1

        for st in solution_terms[:3]:
            q = f"{focus} {st} {base_entity}"
            if include_countries:
                q += f" {include_countries[0]}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="solution"))
            p += 1

        for ct in commercial_terms[:2]:
            q = f"{focus} {base_entity} {ct}"
            if domain_terms:
                q += f" {domain_terms[0]}"
            queries.append(SearchQuery(text=q.strip(), priority=p, family="commercial"))
            p += 1

        if not queries:
            queries.append(SearchQuery(text=f"{focus} {base_entity}", priority=1, family="fallback"))

        excluded_geo = self._combined_excluded_geo(exclude_countries, exclude_presence_countries)
        if entity_type == "company" and excluded_geo:
            suffixes = []
            for country in excluded_geo:
                suffixes.extend(self._country_exclusion_tokens(country))
            suffix_str = " ".join(suffixes)
            queries.append(SearchQuery(
                text=f"{focus} {base_entity} {suffix_str}".strip(),
                priority=p + 1,
                family="exclude",
            ))

        return self._deduplicate(queries)

    def _build_exa_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []
        focus = (task.industry or "").strip() or "industry"
        category = (task.target_category or "general").strip().lower()
        include_countries = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        exclude_countries = [c.strip() for c in (task.geography.exclude_countries or []) if c.strip()]
        exclude_presence_countries = [c.strip() for c in (task.geography.exclude_presence_countries or []) if c.strip()]
        solution_terms = self._solution_terms(task)
        domain_terms = self._domain_terms(task)
        commercial_terms = self._commercial_terms(task)

        entity_phrase = "software and digital companies" if category == "software_company" else "companies"
        geo_desc = ", ".join(include_countries[:3]) if include_countries else "global markets"

        templates = [
            f"{entity_phrase} serving the {focus} industry in {geo_desc}",
        ]
        if solution_terms:
            templates.append(f"{focus} companies focused on {', '.join(solution_terms[:3])}")
        if domain_terms:
            templates.append(f"{focus} companies focused on {', '.join(domain_terms[:3])}")
        if exclude_countries:
            templates.append(f"{focus} companies not headquartered in {', '.join(exclude_countries[:2])}")
        if exclude_presence_countries:
            templates.append(f"{focus} companies without offices or branch presence in {', '.join(exclude_presence_countries[:2])}")
        if commercial_terms:
            templates.append(f"{focus} vendors with distributors, resellers, channel partners, or representatives")

        for i, text in enumerate(templates, start=1):
            queries.append(SearchQuery(text=text.strip(), priority=i, family="semantic", provider_hint="exa"))

        return self._deduplicate(queries)

    def _deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        seen = set()
        out = []
        for q in queries:
            key = q.text.lower().strip()
            if key not in seen:
                seen.add(key)
                out.append(q)
        out.sort(key=lambda x: x.priority)
        return out
