from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class GeographyRules:
    include_countries: List[str] = field(default_factory=list)
    exclude_countries: List[str] = field(default_factory=list)
    exclude_presence_countries: List[str] = field(default_factory=list)
    strict_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputSpec:
    format: str = "xlsx"           # xlsx | csv | pdf | json | ui_table
    filename: str = "results.xlsx"
    include_rejected: bool = False
    include_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CredentialMode:
    mode: str = "free"             # free | paid | user_keys

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    raw_prompt: str = ""
    task_type: str = "entity_discovery"
    # entity_discovery | entity_enrichment | similar_entity_expansion
    # market_research  | document_research | people_search

    target_entity_types: List[str] = field(default_factory=lambda: ["company"])
    target_category: str = "general"   # general | service_company | software_company
    industry: str = ""                 # domain / vertical, e.g. "oil and gas"

    # exact user-typed technical families
    solution_keywords: List[str] = field(default_factory=list)
    # exact user-typed domain/use-case phrases
    domain_keywords: List[str] = field(default_factory=list)

    commercial_intent: str = "general"
    # general | agent_or_distributor | reseller | partner

    target_attributes: List[str] = field(default_factory=lambda: ["website"])
    geography: GeographyRules = field(default_factory=GeographyRules)
    output: OutputSpec = field(default_factory=OutputSpec)
    credential_mode: CredentialMode = field(default_factory=CredentialMode)
    use_local_llm: bool = False
    use_cloud_llm: bool = False
    max_results: int = 25
    mode: str = "Balanced"             # Fast | Balanced | Deep

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    strategy_name: str = "general_search"
    use_ddg: bool = True
    use_exa: bool = False
    use_tavily: bool = False
    use_serpapi: bool = False
    use_firecrawl: bool = False
    use_exa_find_similar: bool = False
    use_local_llm_parser: bool = False
    use_local_llm_classifier: bool = False
    use_cloud_llm_batch_verify: bool = False
    stop_when_enough_valid: bool = True
    max_queries_per_provider: Dict[str, int] = field(default_factory=lambda: {
        "ddg": 3, "exa": 2, "tavily": 1, "serpapi": 0,
    })
    max_candidates_to_process: int = 12

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
