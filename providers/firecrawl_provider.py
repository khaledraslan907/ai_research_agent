from __future__ import annotations

from typing import Any, Dict, Optional

from core.utils import normalize_url


class FirecrawlProvider:
    def __init__(self, api_key: Optional[str] = None):
        from core.config import FIRECRAWL_API_KEY
        self.api_key = (api_key or FIRECRAWL_API_KEY or "").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from firecrawl import FirecrawlApp
                self._client = FirecrawlApp(api_key=self.api_key)
            except Exception:
                self._client = None

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = (api_key or "").strip()
        self._init_client()

    def is_available(self) -> bool:
        return bool(self.api_key and self._client)

    def scrape(self, url: str) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None

        url = normalize_url(url)
        try:
            result = self._client.scrape_url(url, formats=["markdown", "html"])
            # normalise to dict regardless of SDK version
            if hasattr(result, "__dict__"):
                result = vars(result)
            return result if isinstance(result, dict) else None
        except Exception:
            return None
