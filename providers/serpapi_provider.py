from __future__ import annotations

from typing import List, Optional

import requests

from core.models import SearchResult
from core.utils import extract_domain, normalize_url
from providers.base import BaseSearchProvider


class SerpApiProvider(BaseSearchProvider):
    name = "serpapi"

    def __init__(self, api_key: Optional[str] = None):
        from core.config import SERPAPI_KEY
        self.api_key = (api_key or SERPAPI_KEY or "").strip()

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = (api_key or "").strip()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.is_available():
            return []

        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self.api_key,
                    "num": max_results,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        items = data.get("organic_results", []) if isinstance(data, dict) else []
        results: List[SearchResult] = []

        for idx, item in enumerate(items, start=1):
            url     = item.get("link", "") or ""
            title   = item.get("title", "") or ""
            snippet = item.get("snippet", "") or ""

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
