from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import SearchResult


class BaseSearchProvider(ABC):
    """
    Minimal provider contract used by the orchestrator.

    Keep this interface intentionally small so every provider can remain simple:
    - name
    - availability check
    - search(query, max_results)

    Providers may expose extra helper methods (for example Exa's
    search_linkedin_profiles or find_similar), but this base contract should stay
    stable across the project.
    """
    name: str = "base"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        raise NotImplementedError
