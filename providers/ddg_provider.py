from __future__ import annotations

from typing import List

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # legacy name

from core.models import SearchResult
from core.utils import extract_domain, normalize_url
from providers.base import BaseSearchProvider


class DDGProvider(BaseSearchProvider):
    name = "ddg"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def is_available(self) -> bool:
        return True   # always available (no key required)

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        results: List[SearchResult] = []
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
        except Exception:
            return results

        for idx, item in enumerate(raw, start=1):
            url     = item.get("href", "") or item.get("url", "")
            title   = item.get("title", "") or ""
            snippet = item.get("body", "") or item.get("snippet", "") or ""

            if not url:
                continue

            norm = normalize_url(url)
            results.append(SearchResult(
                provider=self.name,
                query=query,
                title=title,
                url=norm,
                snippet=snippet,
                domain=extract_domain(norm),
                rank=idx,
                raw=item,
            ))

        return results

