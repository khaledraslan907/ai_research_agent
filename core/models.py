from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _safe_asdict(obj) -> Dict[str, Any]:
    return asdict(obj)


@dataclass
class SearchQuery:
    text: str
    priority: int = 0
    family: str = "core"
    provider_hint: str = ""

    # New, backward-compatible metadata
    query_id: str = ""
    language: str = "en"
    entity_type: str = "company"
    source_type_hint: str = ""
    rationale: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


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

    # New evidence-friendly fields
    language: str = ""
    source_type: str = ""  # website | directory | pdf | linkedin | paper | news | tender | event
    matched_entity_type: str = ""
    matched_countries: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    trust_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


@dataclass
class EvidenceRecord:
    source_url: str = ""
    source_type: str = ""
    provider: str = ""
    snippet: str = ""
    matched_entity_type: str = ""
    matched_industry: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    matched_geography: List[str] = field(default_factory=list)
    geography_evidence_strength: float = 0.0
    negative_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    accepted: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


@dataclass
class ValidationDecision:
    accepted: bool = False
    reason: str = ""
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    missing_requirements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


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

    # New extracted evidence fields
    source_type: str = "website"
    language: str = ""
    country_mentions: List[str] = field(default_factory=list)
    region_mentions: List[str] = field(default_factory=list)
    service_evidence_snippets: List[str] = field(default_factory=list)
    geography_evidence_snippets: List[str] = field(default_factory=list)
    entity_evidence_snippets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


@dataclass
class CompanyRecord:
    # Generic identity
    entity_type: str = "company"  # company | paper | person | patent | event | tender | product | dataset
    company_name: str = ""
    canonical_name: str = ""
    aliases: List[str] = field(default_factory=list)

    website: str = ""
    domain: str = ""
    description: str = ""
    industry: str = ""
    subdomains: List[str] = field(default_factory=list)

    country: str = ""
    city: str = ""
    hq_country: str = ""
    presence_countries: List[str] = field(default_factory=list)
    has_usa_presence: bool = False
    has_egypt_presence: bool = False

    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    contact_page: str = ""

    source_url: str = ""
    source_provider: str = ""
    source_type: str = ""
    language: str = ""

    confidence_score: float = 0.0
    geography_confidence: float = 0.0
    topic_confidence: float = 0.0
    entity_confidence: float = 0.0

    page_type: str = ""
    is_directory_or_media: bool = False
    is_verified: bool = False
    validation_status: str = "pending"
    validation_reason: str = ""

    matched_keywords: List[str] = field(default_factory=list)
    solution_keywords: List[str] = field(default_factory=list)
    domain_keywords: List[str] = field(default_factory=list)
    notes: str = ""

    raw_sources: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    scoring_breakdown: Dict[str, float] = field(default_factory=dict)

    # Paper-style fields
    authors: str = ""
    doi: str = ""
    publication_year: str = ""
    abstract: str = ""
    journal: str = ""

    # People-style fields
    job_title: str = ""
    seniority: str = ""
    department: str = ""
    employer_name: str = ""
    linkedin_profile: str = ""

    # Event / tender / patent / misc fields
    deadline: str = ""
    organizer: str = ""
    event_date: str = ""
    location: str = ""
    assignee: str = ""
    inventors: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


@dataclass
class SearchSpec:
    raw_prompt: str = ""
    entity_type: str = "company"
    intent_type: str = "general"
    sector: str = ""

    target_category: str = "general"
    solution_keywords: List[str] = field(default_factory=list)
    domain_keywords: List[str] = field(default_factory=list)
    commercial_intent: str = "general"

    include_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    include_countries: List[str] = field(default_factory=list)
    exclude_countries: List[str] = field(default_factory=list)
    exclude_presence_countries: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

    # New generic-agent controls
    subdomains: List[str] = field(default_factory=list)
    source_preferences: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    post_actions: List[str] = field(default_factory=list)
    strict_mode: bool = False
    discovery_mode: bool = True

    max_results: int = 25
    mode: str = "Balanced"
    language: str = "en"
    languages: List[str] = field(default_factory=lambda: ["en"])

    def to_dict(self) -> Dict[str, Any]:
        return _safe_asdict(self)


@dataclass
class SearchBudget:
    max_total_search_calls: int = 8
    max_ddg_calls: int = 3
    max_exa_calls: int = 2
    max_tavily_calls: int = 1
    max_serpapi_calls: int = 0
    max_pages_to_scrape: int = 20
    total_search_calls_used: int = 0
    ddg_calls_used: int = 0
    exa_calls_used: int = 0
    tavily_calls_used: int = 0
    serpapi_calls_used: int = 0
    pages_scraped_used: int = 0

    # New counters
    verification_pages_used: int = 0
    pdf_pages_used: int = 0


@dataclass
class ProviderSettings:
    use_ddg: bool = True
    use_exa: bool = True
    use_tavily: bool = True
    use_serpapi: bool = False
    use_firecrawl: bool = True
    use_llm_parser: bool = False
    use_uploaded_seed_dedupe: bool = True

    # New optional search surfaces
    use_pdf_search: bool = False
    use_directory_search: bool = False
    use_linkedin_search: bool = False
    use_scholar_search: bool = False
    use_patent_search: bool = False
    use_event_search: bool = False
    use_tender_search: bool = False


from core.provider_resolver import ResolvedKeys  # noqa: F401
