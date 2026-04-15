from __future__ import annotations

from typing import List, Optional

from core.models import SearchResult
from core.utils import extract_domain, normalize_url
from providers.base import BaseSearchProvider


class TavilyProvider(BaseSearchProvider):
    name = "tavily"

    def __init__(self, api_key: Optional[str] = None):
        from core.config import TAVILY_API_KEY
        self.api_key = (api_key or TAVILY_API_KEY or "").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except Exception:
                self._client = None

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = (api_key or "").strip()
        self._init_client()

    def is_available(self) -> bool:
        return bool(self.api_key and self._client)

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.is_available():
            return []

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
            )
        except Exception:
            return []

        items = response.get("results", []) if isinstance(response, dict) else []
        results: List[SearchResult] = []

        for idx, item in enumerate(items, start=1):
            url     = item.get("url", "")
            title   = item.get("title", "") or ""
            snippet = item.get("content", "") or ""

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
