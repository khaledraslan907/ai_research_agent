from __future__ import annotations

from typing import List, Optional

from core.models import SearchResult
from core.utils import extract_domain, normalize_url
from providers.base import BaseSearchProvider


class ExaProvider(BaseSearchProvider):
    name = "exa"

    def __init__(self, api_key: Optional[str] = None, search_type: str = "auto"):
        from core.config import EXA_API_KEY
        self.api_key = (api_key or EXA_API_KEY or "").strip()
        self.search_type = search_type
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from exa_py import Exa
                self._client = Exa(api_key=self.api_key)
            except Exception:
                self._client = None

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = (api_key or "").strip()
        self._init_client()

    def is_available(self) -> bool:
        return bool(self.api_key and self._client)

    # ------------------------------------------------------------------
    # Standard semantic search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        if not self.is_available():
            return []

        try:
            kwargs = dict(
                query=query,
                type=self.search_type,
                num_results=max_results,
                highlights=True,
                text=True,
            )
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains

            response = self._client.search_and_contents(**kwargs)
        except Exception:
            return []

        items = getattr(response, "results", None) or []
        return [
            self._item_to_result(item, query, idx)
            for idx, item in enumerate(items, 1)
            if self._get(item, "url")
        ]

    # ------------------------------------------------------------------
    # LinkedIn-scoped search — returns only linkedin.com/in/ URLs
    # ------------------------------------------------------------------

    def search_linkedin_profiles(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """
        Search Exa restricted to linkedin.com/in/ personal profiles.
        Tries multiple parameter styles for exa_py version compatibility:
          - include_domains   (exa_py >= 1.1)
          - includeDomains    (some older versions)
        Falls back to plain search + URL filter if both fail.
        """
        if not self.is_available():
            return []

        base_kwargs = dict(
            query=query,
            type=self.search_type,
            num_results=max_results,
            highlights=True,
            text=True,
        )

        # Try snake_case first (exa_py >= 1.1.x)
        for domain_kwarg in ("include_domains", "includeDomains"):
            try:
                kwargs = {**base_kwargs, domain_kwarg: ["linkedin.com/in"]}
                response = self._client.search_and_contents(**kwargs)
                items = getattr(response, "results", None) or []
                results = [
                    self._item_to_result(item, query, idx)
                    for idx, item in enumerate(items, 1)
                    if self._get(item, "url")
                ]
                if results:
                    return results
                # Empty result with this param is still valid — return empty
                return results
            except TypeError:
                continue  # try next variant
            except Exception:
                return []

        # Final fallback: plain search, then filter to /in/ URLs
        try:
            response = self._client.search_and_contents(**base_kwargs)
            items = getattr(response, "results", None) or []
            all_results = [
                self._item_to_result(item, query, idx)
                for idx, item in enumerate(items, 1)
                if self._get(item, "url")
            ]
            # Keep only linkedin.com/in/ personal profile URLs
            from core.people_search import is_linkedin_profile_url
            return [r for r in all_results if is_linkedin_profile_url(r.url)]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # find_similar — seed-URL based expansion
    # ------------------------------------------------------------------

    def find_similar(self, url: str, max_results: int = 5) -> List[SearchResult]:
        if not self.is_available():
            return []

        try:
            response = self._client.find_similar_and_contents(
                url=url,
                num_results=max_results,
                highlights=True,
                text=True,
            )
        except Exception:
            return []

        items = getattr(response, "results", None) or []
        return [
            self._item_to_result(item, f"find_similar:{url}", idx)
            for idx, item in enumerate(items, 1)
            if self._get(item, "url")
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _item_to_result(self, item, query: str, idx: int) -> SearchResult:
        url        = self._get(item, "url", "")
        title      = self._get(item, "title", "")
        highlights = self._get(item, "highlights", []) or []
        text       = self._get(item, "text", "") or ""

        if highlights and isinstance(highlights, list):
            snippet = " ".join(str(x) for x in highlights[:3]).strip()
        else:
            snippet = str(text[:500]).strip() if text else ""

        norm = normalize_url(url)
        return SearchResult(
            provider=self.name,
            query=query,
            title=title or "",
            url=norm,
            snippet=snippet,
            domain=extract_domain(norm),
            rank=idx,
            raw={
                "title":      title,
                "url":        url,
                "highlights": highlights,
                "text":       text[:1000] if isinstance(text, str) else "",
            },
        )

    @staticmethod
    def _get(item, key: str, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)


    # ------------------------------------------------------------------
    # find_similar — seed-URL based expansion
    # ------------------------------------------------------------------

    def find_similar(self, url: str, max_results: int = 5) -> List[SearchResult]:
        """
        Given a seed company URL, return semantically similar company pages.
        Used for similar_entity_expansion tasks.
        """
        if not self.is_available():
            return []

        try:
            response = self._client.find_similar_and_contents(
                url=url,
                num_results=max_results,
                highlights=True,
                text=True,
            )
        except Exception:
            return []

        items = getattr(response, "results", None) or []
        return [
            self._item_to_result(item, f"find_similar:{url}", idx)
            for idx, item in enumerate(items, 1)
            if self._get(item, "url")
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _item_to_result(self, item, query: str, idx: int) -> SearchResult:
        url       = self._get(item, "url", "")
        title     = self._get(item, "title", "")
        highlights = self._get(item, "highlights", []) or []
        text      = self._get(item, "text", "") or ""

        if highlights and isinstance(highlights, list):
            snippet = " ".join(str(x) for x in highlights[:3]).strip()
        else:
            snippet = str(text[:500]).strip() if text else ""

        norm = normalize_url(url)
        return SearchResult(
            provider=self.name,
            query=query,
            title=title or "",
            url=norm,
            snippet=snippet,
            domain=extract_domain(norm),
            rank=idx,
            raw={
                "title": title,
                "url": url,
                "highlights": highlights,
                "text": text[:1000] if isinstance(text, str) else "",
            },
        )

    @staticmethod
    def _get(item, key: str, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)
