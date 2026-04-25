from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


DEFAULT_ENTITY_TYPES = ["company"]
DEFAULT_TARGET_ATTRIBUTES = ["website"]
DEFAULT_QUERY_LANGUAGES = ["en"]
DEFAULT_SEARCH_FAMILIES = [
    "core",
    "synonym",
    "broader",
    "niche",
    "geo",
    "source_specific",
    "verify",
]


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        s = str(value or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@dataclass
class GeographyRules:
    include_countries: List[str] = field(default_factory=list)
    exclude_countries: List[str] = field(default_factory=list)
    exclude_presence_countries: List[str] = field(default_factory=list)

    # New, backward-compatible geography controls
    include_regions: List[str] = field(default_factory=list)
    exclude_regions: List[str] = field(default_factory=list)
    include_locations: List[str] = field(default_factory=list)  # cities / states / provinces
    exclude_locations: List[str] = field(default_factory=list)

    strict_mode: bool = False
    require_geo_evidence: bool = True
    geo_confidence_threshold: float = 0.0

    def normalized(self) -> "GeographyRules":
        self.include_countries = _dedupe_keep_order(self.include_countries)
        self.exclude_countries = _dedupe_keep_order(self.exclude_countries)
        self.exclude_presence_countries = _dedupe_keep_order(self.exclude_presence_countries)
        self.include_regions = _dedupe_keep_order(self.include_regions)
        self.exclude_regions = _dedupe_keep_order(self.exclude_regions)
        self.include_locations = _dedupe_keep_order(self.include_locations)
        self.exclude_locations = _dedupe_keep_order(self.exclude_locations)
        self.strict_mode = bool(
            self.strict_mode
            or self.include_countries
            or self.exclude_countries
            or self.exclude_presence_countries
            or self.include_regions
            or self.exclude_regions
            or self.include_locations
            or self.exclude_locations
        )
        return self

    def has_any_filter(self) -> bool:
        return bool(
            self.include_countries
            or self.exclude_countries
            or self.exclude_presence_countries
            or self.include_regions
            or self.exclude_regions
            or self.include_locations
            or self.exclude_locations
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputSpec:
    format: str = "xlsx"  # xlsx | csv | pdf | json | ui_table
    filename: str = "results.xlsx"
    include_rejected: bool = False
    include_summary: bool = True

    # New optional output controls
    include_trace: bool = False
    include_evidence: bool = True
    include_clusters: bool = False
    include_gap_analysis: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CredentialMode:
    mode: str = "free"  # free | paid | user_keys

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    raw_prompt: str = ""
    task_type: str = "entity_discovery"
    # entity_discovery | entity_enrichment | similar_entity_expansion
    # market_research  | document_research | people_search

    target_entity_types: List[str] = field(default_factory=lambda: list(DEFAULT_ENTITY_TYPES))
    target_category: str = "general"
    # examples: general | service_company | software_company | manufacturer
    # consultant | operator | lab | university | directory

    industry: str = ""
    subdomains: List[str] = field(default_factory=list)

    # exact user-typed technical families
    solution_keywords: List[str] = field(default_factory=list)
    # exact user-typed domain/use-case phrases
    domain_keywords: List[str] = field(default_factory=list)

    seniority_keywords: List[str] = field(default_factory=list)
    company_type_keywords: List[str] = field(default_factory=list)
    include_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)

    commercial_intent: str = "general"
    # general | agent_or_distributor | reseller | partner | recruiting
    # competitor_research | supplier_discovery | investment | academic

    target_attributes: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_ATTRIBUTES))
    evidence_requirements: List[str] = field(default_factory=list)
    source_preferences: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: list(DEFAULT_QUERY_LANGUAGES))
    post_actions: List[str] = field(default_factory=list)

    geography: GeographyRules = field(default_factory=GeographyRules)
    output: OutputSpec = field(default_factory=OutputSpec)
    credential_mode: CredentialMode = field(default_factory=CredentialMode)

    use_local_llm: bool = False
    use_cloud_llm: bool = False
    use_bilingual_search: bool = False
    strict_validation: bool = False
    discovery_mode: bool = True

    max_results: int = 25
    mode: str = "Balanced"  # Fast | Balanced | Deep
    notes: str = ""

    def normalized(self) -> "TaskSpec":
        self.target_entity_types = _dedupe_keep_order(self.target_entity_types or list(DEFAULT_ENTITY_TYPES))
        self.subdomains = _dedupe_keep_order(self.subdomains)
        self.solution_keywords = _dedupe_keep_order(self.solution_keywords)
        self.domain_keywords = _dedupe_keep_order(self.domain_keywords)
        self.seniority_keywords = _dedupe_keep_order(self.seniority_keywords)
        self.company_type_keywords = _dedupe_keep_order(self.company_type_keywords)
        self.include_terms = _dedupe_keep_order(self.include_terms)
        self.exclude_terms = _dedupe_keep_order(self.exclude_terms)
        self.target_attributes = _dedupe_keep_order(self.target_attributes or list(DEFAULT_TARGET_ATTRIBUTES))
        self.evidence_requirements = _dedupe_keep_order(self.evidence_requirements)
        self.source_preferences = _dedupe_keep_order(self.source_preferences)
        self.languages = _dedupe_keep_order(self.languages or list(DEFAULT_QUERY_LANGUAGES))
        self.post_actions = _dedupe_keep_order(self.post_actions)
        self.geography = (self.geography or GeographyRules()).normalized()
        return self

    def primary_entity_type(self) -> str:
        return (self.target_entity_types or ["company"])[0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    strategy_name: str = "general_search"

    # Provider switches (kept backward-compatible)
    use_ddg: bool = True
    use_exa: bool = False
    use_tavily: bool = False
    use_serpapi: bool = False
    use_firecrawl: bool = False
    use_exa_find_similar: bool = False

    # New optional provider/search surfaces
    use_directory_search: bool = False
    use_pdf_search: bool = False
    use_linkedin_search: bool = False
    use_scholar_search: bool = False
    use_patent_search: bool = False
    use_event_search: bool = False
    use_tender_search: bool = False

    use_local_llm_parser: bool = False
    use_local_llm_classifier: bool = False
    use_cloud_llm_batch_verify: bool = False
    critique_plan_with_llm: bool = False
    critique_results_with_llm: bool = False

    stop_when_enough_valid: bool = True
    require_country_evidence: bool = False
    require_presence_evidence: bool = False
    allow_bilingual_queries: bool = False

    provider_order: List[str] = field(default_factory=lambda: ["ddg", "exa", "tavily", "serpapi"])
    search_families: List[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_FAMILIES))
    query_languages: List[str] = field(default_factory=lambda: list(DEFAULT_QUERY_LANGUAGES))

    max_queries_per_provider: Dict[str, int] = field(default_factory=lambda: {
        "ddg": 3,
        "exa": 2,
        "tavily": 1,
        "serpapi": 0,
    })
    max_candidates_to_process: int = 12
    verify_top_n: int = 8
    enrich_top_n: int = 5
    max_verification_pages: int = 15

    def normalized(self) -> "ExecutionPlan":
        self.provider_order = _dedupe_keep_order(self.provider_order)
        self.search_families = _dedupe_keep_order(self.search_families or list(DEFAULT_SEARCH_FAMILIES))
        self.query_languages = _dedupe_keep_order(self.query_languages or list(DEFAULT_QUERY_LANGUAGES))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
