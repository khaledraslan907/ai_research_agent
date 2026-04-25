from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import requests
import trafilatura
from bs4 import BeautifulSoup

from core.config import DEFAULT_MAX_INTERNAL_PAGES, DEFAULT_TIMEOUT
from core.geography import find_countries_in_text, find_first_country_in_text
from core.models import CrawlResult
from core.utils import (
    clean_text,
    extract_emails,
    extract_phones,
    get_default_headers,
    get_root_url,
    join_url,
    looks_like_contact_link,
    normalize_url,
    unique_list,
)

HQ_PATTERNS = [
    "headquartered in", "headquartered at", "based in",
    "our headquarters", "head office", "corporate headquarters",
    "global headquarters", "founded in", "hq in", "hq:",
]

PRESENCE_PATTERNS = [
    "offices in", "office locations", "our offices", "global presence",
    "presence in", "operates in", "serving clients in", "regional offices",
    "our presence", "global offices", "locations worldwide",
    "we operate in", "active in", "countries we serve",
]

LOCATION_PAGE_HINTS = [
    "contact", "about", "team", "company", "corporate",
    "office", "location", "locations", "global-presence",
    "presence", "offices", "reach-us", "get-in-touch",
    "اتصل", "تواصل", "المواقع", "مكاتب", "عن الشركة",
]


class WebsiteScraper:
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_internal_pages: int = DEFAULT_MAX_INTERNAL_PAGES,
        use_firecrawl: bool = True,
        firecrawl_api_key: str | None = None,
    ):
        self.timeout = timeout
        self.max_internal_pages = max_internal_pages
        from providers.firecrawl_provider import FirecrawlProvider
        self.firecrawl = FirecrawlProvider(api_key=firecrawl_api_key) if use_firecrawl else None

    def set_firecrawl_api_key(self, api_key: str | None):
        if self.firecrawl:
            self.firecrawl.set_api_key(api_key)

    def scrape(self, url: str, follow_internal_links: bool = True) -> CrawlResult:
        url = normalize_url(url)
        html, final_url, error = self._fetch_html(url)
        if error:
            fallback = self._scrape_with_firecrawl(url)
            if fallback:
                return fallback
            return CrawlResult(url=url, final_url=final_url or url, success=False, error=error)

        result = self._parse_page(url=url, final_url=final_url or url, html=html)
        setattr(result, "_raw_html", html)

        if follow_internal_links and result.contact_links:
            extra_emails: List[str] = []
            extra_phones: List[str] = []
            extra_texts: List[str] = []
            presence_countries: List[str] = list(result.detected_presence_countries)
            for link in result.contact_links[: self.max_internal_pages]:
                sub_html, sub_final, sub_err = self._fetch_html(link)
                if sub_err or not sub_html:
                    fb = self._scrape_with_firecrawl(link)
                    if fb and fb.success:
                        sub_result = fb
                    else:
                        continue
                else:
                    sub_result = self._parse_page(url=link, final_url=sub_final or link, html=sub_html)
                extra_emails.extend(sub_result.emails)
                extra_phones.extend(sub_result.phones)
                if sub_result.text:
                    extra_texts.append(sub_result.text)
                if not result.detected_company_name and sub_result.detected_company_name:
                    result.detected_company_name = sub_result.detected_company_name
                if not result.detected_country and sub_result.detected_country:
                    result.detected_country = sub_result.detected_country
                if not result.detected_hq_country and sub_result.detected_hq_country:
                    result.detected_hq_country = sub_result.detected_hq_country
                presence_countries.extend(sub_result.detected_presence_countries)
            result.emails = unique_list(result.emails + extra_emails)
            result.phones = unique_list(result.phones + extra_phones)
            result.detected_presence_countries = unique_list(presence_countries)
            if extra_texts:
                result.text = clean_text(result.text + " " + " ".join(extra_texts))
        return result

    def _scrape_with_firecrawl(self, url: str) -> CrawlResult | None:
        if not self.firecrawl or not self.firecrawl.is_available():
            return None
        data = self.firecrawl.scrape(url)
        if not data:
            return None
        markdown = clean_text(data.get("markdown", "") if isinstance(data, dict) else "")
        html = data.get("html", "") if isinstance(data, dict) else ""
        title = ""
        text = markdown
        if html:
            try:
                soup = BeautifulSoup(html, "lxml")
                title = clean_text(soup.title.get_text()) if soup.title else ""
                if not text:
                    text = clean_text(soup.get_text(" ", strip=True))
            except Exception:
                pass
        combined = clean_text(f"{title} {text}")
        result = CrawlResult(
            url=url,
            final_url=url,
            title=title,
            text=combined,
            emails=extract_emails(combined),
            phones=extract_phones(combined),
            social_links={},
            contact_links=[],
            detected_company_name=self._detect_company_name_from_text(title, combined),
            detected_country=find_first_country_in_text(combined),
            detected_hq_country=self._detect_hq_country(combined),
            detected_presence_countries=self._detect_presence_countries(combined),
            meta_description="",
            success=True,
            error="",
        )
        try:
            result.country_mentions = find_countries_in_text(combined)
        except Exception:
            pass
        return result

    def _fetch_html(self, url: str) -> Tuple[str, str, str]:
        try:
            resp = requests.get(url, headers=get_default_headers(), timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ct and "application/xhtml+xml" not in ct:
                return "", resp.url, "Non-HTML content"
            return resp.text, resp.url, ""
        except Exception as e:
            return "", url, str(e)

    def _parse_page(self, url: str, final_url: str, html: str) -> CrawlResult:
        soup = BeautifulSoup(html, "lxml")
        title = clean_text(soup.title.get_text()) if soup.title else ""
        meta_desc = self._extract_meta_description(soup)
        text = self._extract_main_text(html)
        page_text = clean_text(f"{title} {meta_desc} {text}")
        result = CrawlResult(
            url=url,
            final_url=final_url,
            title=title,
            text=page_text,
            emails=extract_emails(page_text),
            phones=extract_phones(page_text),
            social_links=self._extract_social_links(soup, final_url),
            contact_links=self._extract_contact_links(soup, final_url),
            detected_company_name=self._detect_company_name(soup, title),
            detected_country=find_first_country_in_text(page_text),
            detected_hq_country=self._detect_hq_country(page_text),
            detected_presence_countries=self._detect_presence_countries(page_text),
            meta_description=meta_desc,
            success=True,
            error="",
        )
        try:
            result.country_mentions = find_countries_in_text(page_text)
        except Exception:
            pass
        return result

    def _extract_main_text(self, html: str) -> str:
        try:
            extracted = trafilatura.extract(html, include_links=False, include_images=False)
            if extracted:
                return clean_text(extracted)
        except Exception:
            pass
        try:
            soup = BeautifulSoup(html, "lxml")
            return clean_text(soup.get_text(" ", strip=True))
        except Exception:
            return ""

    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        for attrs in [{"name": "description"}, {"property": "og:description"}]:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                return clean_text(tag["content"])
        return ""

    def _extract_social_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        mapping = {
            "linkedin": "linkedin.com",
            "twitter": "twitter.com",
            "x": "x.com",
            "facebook": "facebook.com",
            "instagram": "instagram.com",
            "youtube": "youtube.com",
        }
        social: Dict[str, str] = {}
        for a in soup.find_all("a", href=True):
            href = join_url(base_url, a["href"]).lower()
            for key, domain in mapping.items():
                if domain in href and key not in social:
                    social[key] = join_url(base_url, a["href"])
        return social

    def _extract_contact_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        root = get_root_url(base_url)
        links: List[str] = []
        seen: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            full = join_url(base_url, href)
            if not full.startswith(root):
                continue
            anchor = clean_text(a.get_text(" ", strip=True)).lower()
            candidate = f"{anchor} {full}".lower()
            if looks_like_contact_link(candidate) or any(k in candidate for k in LOCATION_PAGE_HINTS):
                if full not in seen:
                    seen.add(full)
                    links.append(full)
        return links

    def _detect_company_name(self, soup: BeautifulSoup, title: str) -> str:
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            return clean_text(og_site["content"])
        if title:
            return self._detect_company_name_from_text(title, title)
        return ""

    def _detect_company_name_from_text(self, title: str, text: str) -> str:
        candidate = clean_text(title)
        candidate = re.split(r"\s*[\-|–|:]\s*", candidate)[0].strip()
        if 2 <= len(candidate) <= 80:
            return candidate
        lines = [x.strip() for x in re.split(r"[\n\r]+", text or "") if x.strip()]
        for line in lines[:5]:
            if 2 <= len(line) <= 80:
                return line
        return ""

    def _detect_hq_country(self, text: str) -> str:
        low = (text or "").lower()
        for phrase in HQ_PATTERNS:
            idx = low.find(phrase)
            if idx >= 0:
                chunk = text[idx: idx + 300]
                country = find_first_country_in_text(chunk)
                if country:
                    return country
        return ""

    def _detect_presence_countries(self, text: str) -> List[str]:
        found: List[str] = []
        low = (text or "").lower()
        for phrase in PRESENCE_PATTERNS:
            idx = low.find(phrase)
            if idx >= 0:
                chunk = text[idx: idx + 500]
                found.extend(find_countries_in_text(chunk))
        return unique_list(found)
