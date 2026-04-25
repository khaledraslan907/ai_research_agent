from __future__ import annotations

"""
provider_selector.py
====================
Selects which providers should be emphasized for a given task.
This module is advisory: it does not replace plan_builder/orchestrator, but it
helps future pipelines decide where to spend limited budget first.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from core.models import ProviderSettings
from core.task_models import TaskSpec


@dataclass
class ProviderRecommendation:
    provider: str
    priority: int = 0
    reason: str = ""


@dataclass
class ProviderSelection:
    enabled: List[str] = field(default_factory=list)
    ordered: List[ProviderRecommendation] = field(default_factory=list)
    disabled_reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "enabled": list(self.enabled),
            "ordered": [vars(x) for x in self.ordered],
            "disabled_reasons": dict(self.disabled_reasons),
        }


def recommend_providers(task_spec: TaskSpec, provider_settings: ProviderSettings) -> ProviderSelection:
    entity = (getattr(task_spec, "target_entity_types", []) or ["company"])[0]
    mode = (getattr(task_spec, "mode", "Balanced") or "Balanced").lower()
    category = getattr(task_spec, "target_category", "general") or "general"
    geo = getattr(task_spec, "geography", None)
    has_geo = bool(getattr(geo, "include_countries", []) or getattr(geo, "exclude_countries", []) or getattr(geo, "exclude_presence_countries", []))

    sel = ProviderSelection()

    def add(provider: str, priority: int, reason: str):
        if provider not in sel.enabled:
            sel.enabled.append(provider)
        sel.ordered.append(ProviderRecommendation(provider=provider, priority=priority, reason=reason))

    if getattr(provider_settings, "use_ddg", False):
        base_priority = 90 if mode == "fast" else 70
        add("ddg", base_priority, "Always-on broad recall and free fallback.")
    else:
        sel.disabled_reasons["ddg"] = "Disabled in provider settings"

    if getattr(provider_settings, "use_exa", False):
        priority = 95 if entity in {"paper", "person"} else 85
        if category == "software_company":
            priority += 5
        add("exa", min(priority, 100), "Semantic discovery and strong relevance for niche entities.")
    else:
        sel.disabled_reasons["exa"] = "Disabled or missing key"

    if getattr(provider_settings, "use_tavily", False):
        priority = 80 if mode in {"balanced", "deep"} else 45
        if entity in {"paper", "event", "tender"}:
            priority += 5
        add("tavily", min(priority, 100), "Question-style recall and research-oriented search.")
    else:
        sel.disabled_reasons["tavily"] = "Disabled or missing key"

    if getattr(provider_settings, "use_serpapi", False):
        priority = 88 if entity == "person" else 60
        if has_geo:
            priority += 3
        add("serpapi", min(priority, 100), "Precise Google-style retrieval, especially useful for LinkedIn/site queries.")
    else:
        sel.disabled_reasons["serpapi"] = "Disabled or missing key"

    if getattr(provider_settings, "use_firecrawl", False):
        add("firecrawl", 55, "Fallback for JS-heavy pages and scrape failures.")
    else:
        sel.disabled_reasons["firecrawl"] = "Disabled or missing key"

    sel.ordered.sort(key=lambda x: (-x.priority, x.provider))
    return sel


def provider_budget_weights(task_spec: TaskSpec, provider_settings: ProviderSettings) -> Dict[str, float]:
    """Return relative weights for provider budget allocation."""
    selection = recommend_providers(task_spec, provider_settings)
    if not selection.ordered:
        return {}
    total = sum(max(x.priority, 1) for x in selection.ordered)
    return {x.provider: round(max(x.priority, 1) / total, 4) for x in selection.ordered}
