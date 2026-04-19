from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Search-layer models
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    text: str
    priority: int = 0
    family: str = "core"          # core | local | exclude | fallback | semantic | linkedin
    provider_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    provider: str
    query: str
    title: str
    url: str
    snippet: str
    domain: str
    rank: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrawlResult:
    url: str
    final_url: str
    title: str = ""
    text: str = ""
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    social_links: Dict[str, str] = field(default_factory=dict)
    contact_links: List[str] = field(default_factory=list)
    detected_company_name: str = ""
    detected_country: str = ""
    detected_hq_country: str = ""
    detected_presence_countries: List[str] = field(default_factory=list)
    meta_description: str = ""
    success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Record model — unified for companies, papers, people, orgs, etc.
# ---------------------------------------------------------------------------

@dataclass
class CompanyRecord:
    # Core identity
    company_name: str = ""       # also used as: paper title, person name
    website: str = ""
    domain: str = ""
    description: str = ""

    # Geography
    country: str = ""
    city: str = ""
    hq_country: str = ""
    presence_countries: List[str] = field(default_factory=list)
    has_usa_presence: bool = False
    has_egypt_presence: bool = False

    # Contact
    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    contact_page: str = ""

    # Source metadata
    source_url: str = ""
    source_provider: str = ""
    confidence_score: float = 0.0
    page_type: str = ""           # company | blog | media | directory | document | person | unknown
    is_directory_or_media: bool = False
    matched_keywords: List[str] = field(default_factory=list)
    notes: str = ""
    raw_sources: List[Dict[str, Any]] = field(default_factory=list)

    # Paper/document specific
    authors: str = ""
    doi: str = ""
    publication_year: str = ""

    # Person / LinkedIn specific
    job_title: str = ""           # e.g. "Petroleum Engineer", "HR Manager"
    seniority: str = ""           # e.g. "Senior", "Manager", "Director"
    department: str = ""          # e.g. "Engineering", "HR", "Operations"
    employer_name: str = ""       # company they work at
    linkedin_profile: str = ""    # direct LinkedIn profile URL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Search specification (used by scorer / downstream filtering)
# ---------------------------------------------------------------------------

@dataclass
class SearchSpec:
    raw_prompt: str = ""
    entity_type: str = "company"
    intent_type: str = "general"
    sector: str = ""

    # New structured intent fields
    target_category: str = "general"   # general | service_company | software_company
    solution_keywords: List[str] = field(default_factory=list)
    commercial_intent: str = "general"  # general | agent_or_distributor | reseller | partner

    include_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    include_countries: List[str] = field(default_factory=list)
    exclude_countries: List[str] = field(default_factory=list)
    exclude_presence_countries: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    max_results: int = 25
    mode: str = "Balanced"
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class SearchBudget:
    max_total_search_calls: int = 8
    max_ddg_calls: int = 3
    max_exa_calls: int = 2
    max_tavily_calls: int = 1
    max_serpapi_calls: int = 0
    max_pages_to_scrape: int = 20
    # counters (mutated by BudgetManager)
    total_search_calls_used: int = 0
    ddg_calls_used: int = 0
    exa_calls_used: int = 0
    tavily_calls_used: int = 0
    serpapi_calls_used: int = 0
    pages_scraped_used: int = 0


# ---------------------------------------------------------------------------
# Provider settings
# ---------------------------------------------------------------------------

@dataclass
class ProviderSettings:
    use_ddg: bool = True
    use_exa: bool = True
    use_tavily: bool = True
    use_serpapi: bool = False
    use_firecrawl: bool = True
    use_llm_parser: bool = False
    use_uploaded_seed_dedupe: bool = True


# ---------------------------------------------------------------------------
# Resolved API keys
# ---------------------------------------------------------------------------

# ResolvedKeys is defined in core/provider_resolver.py
from core.provider_resolver import ResolvedKeys  # noqa: F401
