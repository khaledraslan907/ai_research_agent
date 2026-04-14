from __future__ import annotations

from core.models import SearchBudget


class BudgetManager:
    def __init__(self, budget: SearchBudget):
        self.budget = budget

    def can_use_provider(self, provider_name: str) -> bool:
        provider_name = provider_name.lower().strip()

        # Global cap check first
        if self.budget.total_search_calls_used >= self.budget.max_total_search_calls:
            return False

        if provider_name == "ddg":
            return self.budget.ddg_calls_used < self.budget.max_ddg_calls
        if provider_name == "exa":
            return self.budget.exa_calls_used < self.budget.max_exa_calls
        if provider_name == "tavily":
            return self.budget.tavily_calls_used < self.budget.max_tavily_calls
        if provider_name == "serpapi":
            return self.budget.serpapi_calls_used < self.budget.max_serpapi_calls

        return False

    def register_search_call(self, provider_name: str) -> None:
        """Increment both global counter AND per-provider counter."""
        self.budget.total_search_calls_used += 1          # BUG FIX: was missing

        provider_name = provider_name.lower().strip()
        if provider_name == "ddg":
            self.budget.ddg_calls_used += 1
        elif provider_name == "exa":
            self.budget.exa_calls_used += 1
        elif provider_name == "tavily":
            self.budget.tavily_calls_used += 1
        elif provider_name == "serpapi":
            self.budget.serpapi_calls_used += 1

    def can_scrape_pages(self, pages: int = 1) -> bool:
        return (self.budget.pages_scraped_used + pages) <= self.budget.max_pages_to_scrape

    def register_scraped_pages(self, pages: int = 1) -> None:
        self.budget.pages_scraped_used += pages

    def remaining_provider_calls(self, provider_name: str) -> int:
        provider_name = provider_name.lower().strip()
        if provider_name == "ddg":
            return max(0, self.budget.max_ddg_calls - self.budget.ddg_calls_used)
        if provider_name == "exa":
            return max(0, self.budget.max_exa_calls - self.budget.exa_calls_used)
        if provider_name == "tavily":
            return max(0, self.budget.max_tavily_calls - self.budget.tavily_calls_used)
        if provider_name == "serpapi":
            return max(0, self.budget.max_serpapi_calls - self.budget.serpapi_calls_used)
        return 0

    def remaining_pages(self) -> int:
        return max(0, self.budget.max_pages_to_scrape - self.budget.pages_scraped_used)

    def summary(self) -> dict:
        return {
            "max_total_search_calls":  self.budget.max_total_search_calls,
            "total_search_calls_used": self.budget.total_search_calls_used,
            # per-provider used
            "ddg_calls_used":          self.budget.ddg_calls_used,
            "exa_calls_used":          self.budget.exa_calls_used,
            "tavily_calls_used":       self.budget.tavily_calls_used,
            "serpapi_calls_used":      self.budget.serpapi_calls_used,
            "pages_scraped_used":      self.budget.pages_scraped_used,
            # per-provider max (app.py needs these)
            "max_ddg_calls":           self.budget.max_ddg_calls,
            "max_exa_calls":           self.budget.max_exa_calls,
            "max_tavily_calls":        self.budget.max_tavily_calls,
            "max_serpapi_calls":       self.budget.max_serpapi_calls,
            "max_pages_to_scrape":     self.budget.max_pages_to_scrape,
            # remaining
            "remaining_ddg":           self.remaining_provider_calls("ddg"),
            "remaining_exa":           self.remaining_provider_calls("exa"),
            "remaining_tavily":        self.remaining_provider_calls("tavily"),
            "remaining_serpapi":       self.remaining_provider_calls("serpapi"),
            "remaining_pages":         self.remaining_pages(),
        }
