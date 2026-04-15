from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import SearchResult


class BaseSearchProvider(ABC):
    name: str = "base"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        ...
