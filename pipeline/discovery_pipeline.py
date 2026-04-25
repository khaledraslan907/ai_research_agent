from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from core.critic import CriticIssue, review_query_plan
from core.llm_query_planner import plan_queries
from core.models import SearchQuery
from core.query_expander import build_expanded_queries
from core.task_models import TaskSpec


@dataclass
class DiscoveryOutput:
    queries: Dict[str, List[SearchQuery]] = field(default_factory=dict)
    issues: List[CriticIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'queries': {k: [q.to_dict() for q in v] for k, v in self.queries.items()},
            'issues': [vars(x) for x in self.issues],
            'metadata': dict(self.metadata),
        }


class DiscoveryPipeline:
    """Builds provider-specific query plans plus recall-oriented expansions."""

    def __init__(self, max_extra_per_provider: int = 3):
        self.max_extra_per_provider = max_extra_per_provider

    def run(self, task_spec: TaskSpec, llm=None, include_expansions: bool = True) -> DiscoveryOutput:
        base_plan = plan_queries(task_spec, llm=llm)
        final_plan: Dict[str, List[SearchQuery]] = {k: list(v or []) for k, v in (base_plan or {}).items()}
        expansion_meta: Dict[str, Any] = {}

        if include_expansions:
            expanded = build_expanded_queries(task_spec, max_queries=12)
            expansion_meta['expanded_counts'] = {k: len(v or []) for k, v in expanded.items()}
            for provider, queries in expanded.items():
                existing = final_plan.setdefault(provider, [])
                seen = {q.text.lower().strip() for q in existing}
                appended = 0
                for query in queries or []:
                    key = query.text.lower().strip()
                    if not key or key in seen:
                        continue
                    existing.append(query)
                    seen.add(key)
                    appended += 1
                    if appended >= self.max_extra_per_provider:
                        break

        issues = review_query_plan(final_plan, task_spec)
        expansion_meta['final_counts'] = {k: len(v or []) for k, v in final_plan.items()}
        return DiscoveryOutput(queries=final_plan, issues=issues, metadata=expansion_meta)
