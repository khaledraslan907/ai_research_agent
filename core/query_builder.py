from __future__ import annotations

from typing import List

from core.task_models import TaskSpec
from core.models import SearchQuery


COUNTRY_TLD_HINTS = {
    "egypt": "eg", "saudi arabia": "sa", "united arab emirates": "ae",
    "qatar": "qa", "oman": "om", "kuwait": "kw", "bahrain": "bh",
    "india": "in", "united kingdom": "uk", "france": "fr",
    "germany": "de", "italy": "it", "spain": "es", "canada": "ca",
    "australia": "au", "brazil": "br", "netherlands": "nl",
    "norway": "no", "sweden": "se", "denmark": "dk",
}

COUNTRY_EXCLUSION_HINTS = {
    "usa": ['-"usa"', '-"united states"', '-"texas"', '-"houston"', '-"dallas"', '-"new york"', '-"america"'],
    "egypt": ['-"egypt"', '-"cairo"', '-"alexandria"', '-"egyptian"'],
    "canada": ['-"canada"', '-"canadian"', '-"calgary"', '-"toronto"'],
    "china": ['-"china"', '-"chinese"', '-"beijing"', '-"shanghai"'],
}


class QueryBuilder:
    def __init__(self, max_core_queries: int = 6, max_fallback_queries: int = 3):
        self.max_core_queries = max_core_queries
        self.max_fallback_queries = max_fallback_queries

    def build_for_provider(self, task: TaskSpec, provider_name: str) -> List[SearchQuery]:
        provider_name = (provider_name or "").lower().strip()
        if provider_name == "exa":
            return self._build_exa_queries(task)
        return self._build_standard_queries(task)

    # ------------------------------------------------------------------
    # Standard queries (DDG, Tavily, SerpApi)
    # ------------------------------------------------------------------

    def _build_standard_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []

        focus       = (task.industry or "").strip() or "industry"
        category    = (task.target_category or "general").strip().lower()
        entity_type = (task.target_entity_types[0] if task.target_entity_types else "company").lower()

        include_countries = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        exclude_countries = [c.strip() for c in (task.geography.exclude_countries or []) if c.strip()]

        core_templates     = self._get_core_templates(focus, category, entity_type)
        fallback_templates = self._get_fallback_templates(focus, category, entity_type)

        # CORE queries: apply include-geo only (not exclude negative operators — they garble DDG)
        for i, text in enumerate(core_templates[: self.max_core_queries], start=1):
            q = self._apply_include_geo(text, include_countries)
            queries.append(SearchQuery(text=q, priority=i, family="core"))

        # LOCAL queries: country-specific searches when include_countries is set
        if entity_type == "company" and include_countries:
            for i, text in enumerate(
                self._build_local_country_queries(focus, category, include_countries), start=10
            ):
                queries.append(SearchQuery(text=text, priority=i, family="local"))

        # EXCLUDE queries: dedicated queries WITH negative operators for exclude_countries
        if entity_type == "company" and exclude_countries:
            for i, text in enumerate(
                self._build_exclusion_queries(focus, category, exclude_countries), start=15
            ):
                queries.append(SearchQuery(text=text, priority=i, family="exclude"))

        # FALLBACK queries: clean, no geo operators
        for i, text in enumerate(fallback_templates[: self.max_fallback_queries], start=20):
            q = self._apply_include_geo(text, include_countries)
            queries.append(SearchQuery(text=q, priority=i, family="fallback"))

        return self._deduplicate(queries)

    # ------------------------------------------------------------------
    # Exa semantic queries
    # ------------------------------------------------------------------

    def _build_exa_queries(self, task: TaskSpec) -> List[SearchQuery]:
        queries: List[SearchQuery] = []

        focus       = (task.industry or "").strip() or "industry"
        category    = (task.target_category or "general").strip().lower()
        entity_type = (task.target_entity_types[0] if task.target_entity_types else "company").lower()

        include_countries = [c.strip() for c in (task.geography.include_countries or []) if c.strip()]
        exclude_countries = [c.strip() for c in (task.geography.exclude_countries or []) if c.strip()]

        if entity_type == "company":
            if category == "service_company":
                templates = [
                    f"Real service companies specializing in {focus}",
                    f"Field service contractors and engineering firms in {focus}",
                    f"Oilfield and industrial service providers for {focus}",
                ]
            elif category == "software_company":
                templates = [
                    f"Software and digital technology companies in the {focus} industry",
                    f"SaaS platforms and analytics companies serving {focus}",
                    f"AI and automation vendors for {focus}",
                    f"Industrial digital solutions for {focus}",
                ]
            else:
                templates = [
                    f"Companies and businesses working in {focus}",
                    f"Vendors and suppliers serving the {focus} industry",
                    f"Active companies in the {focus} sector",
                ]
        elif entity_type == "paper":
            templates = [
                f"Research papers and academic studies on {focus}",
                f"Journal articles and technical reports about {focus}",
                f"Scientific publications on {focus}",
            ]
        elif entity_type == "organization":
            templates = [
                f"Organizations, institutes, and associations related to {focus}",
                f"Industry groups and non-profits focused on {focus}",
            ]
        else:
            templates = [
                f"Find {entity_type}s related to {focus}",
                f"Top {entity_type}s in {focus}",
            ]

        for i, text in enumerate(templates, start=1):
            q = text.strip()
            if include_countries:
                q += f" based in {' or '.join(include_countries[:2])}"
            if exclude_countries:
                q += f", not headquartered in {' or '.join(exclude_countries[:2])}"
            queries.append(SearchQuery(text=q, priority=i, family="semantic", provider_hint="exa"))

        return self._deduplicate(queries)

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _get_core_templates(self, focus: str, category: str, entity_type: str) -> List[str]:
        if entity_type == "company":
            if category == "service_company":
                return [
                    f"{focus} service company",
                    f"{focus} engineering services",
                    f"{focus} field services contractor",
                    f"{focus} oilfield services company",
                    f"{focus} services provider",
                    f"{focus} contractor site:company",
                ]
            if category == "software_company":
                return [
                    f"{focus} software company",
                    f"{focus} digital solutions provider",
                    f"{focus} technology company",
                    f"{focus} analytics platform",
                    f"{focus} automation software",
                    f"{focus} SaaS company",
                ]
            return [
                f"{focus} company",
                f"{focus} vendor",
                f"{focus} provider",
                f"{focus} supplier",
                f"{focus} business",
            ]

        if entity_type == "paper":
            return [
                f"{focus} research paper",
                f"{focus} journal article",
                f"{focus} study publication",
                f"{focus} technical report",
                f"{focus} academic paper",
            ]

        if entity_type == "organization":
            return [
                f"{focus} organization",
                f"{focus} association",
                f"{focus} institute",
                f"{focus} consortium",
            ]

        return [
            f"{focus} {entity_type}",
            f"top {entity_type}s in {focus}",
            f"{entity_type} about {focus}",
        ]

    def _get_fallback_templates(self, focus: str, category: str, entity_type: str) -> List[str]:
        if entity_type == "company":
            if category == "service_company":
                return [
                    f"{focus} services firm",
                    f"{focus} drilling services",
                    f"{focus} well services",
                ]
            if category == "software_company":
                return [
                    f"{focus} tech startup",
                    f"{focus} data company",
                    f"{focus} software vendor",
                ]
            return [f"{focus} businesses", f"{focus} firms"]

        if entity_type == "paper":
            return [f"{focus} pdf paper", f"{focus} doi article"]

        return [f"{focus} {entity_type}", f"find {entity_type} {focus}"]

    def _build_local_country_queries(
        self, focus: str, category: str, include_countries: List[str]
    ) -> List[str]:
        country = include_countries[0].lower()
        if category == "service_company":
            base = [
                f"{focus} services {country}",
                f"{focus} contractor {country}",
                f"{focus} engineering {country}",
            ]
        elif category == "software_company":
            base = [
                f"{focus} software {country}",
                f"{focus} digital {country}",
                f"{focus} technology {country}",
            ]
        else:
            base = [
                f"{focus} company {country}",
                f"{focus} provider {country}",
            ]

        tld = COUNTRY_TLD_HINTS.get(country)
        if tld:
            base.append(f"site:.{tld} {focus}")

        return [q.strip() for q in base]

    def _build_exclusion_queries(
        self, focus: str, category: str, exclude_countries: List[str]
    ) -> List[str]:
        if category == "software_company":
            base = [f"{focus} software company", f"{focus} digital vendor", f"{focus} technology provider"]
        elif category == "service_company":
            base = [f"{focus} services company", f"{focus} contractor", f"{focus} field services"]
        else:
            base = [f"{focus} company", f"{focus} provider"]

        suffixes: List[str] = []
        for country in exclude_countries:
            suffixes.extend(COUNTRY_EXCLUSION_HINTS.get(country.lower(), [f'-"{country}"']))

        suffix_str = " ".join(suffixes)
        return [f"{b} {suffix_str}".strip() for b in base]

    def _apply_include_geo(self, text: str, include_countries: List[str]) -> str:
        """Apply only include-country suffix. Never add negative operators to core queries."""
        q = text.strip()
        if include_countries:
            q += " " + " ".join(include_countries[:2])
        return q.strip()

    def _apply_geo(
        self, text: str, include_countries: List[str], exclude_countries: List[str]
    ) -> str:
        q = text.strip()
        if include_countries:
            q += " " + " ".join(include_countries[:2])
        for country in exclude_countries:
            hints = COUNTRY_EXCLUSION_HINTS.get(country.lower(), [f'-"{country}"'])
            q += " " + " ".join(hints)
        return q.strip()

    def _deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        seen = set()
        result = []
        for q in queries:
            key = q.text.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(q)
        result.sort(key=lambda x: x.priority)
        return result
