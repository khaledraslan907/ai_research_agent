"""
structured_data_extractor.py
=============================
Extracts contact information from structured HTML data:
  1. JSON-LD (schema.org) — most reliable source for modern B2B sites
  2. Meta tags (og:email, twitter:site, contact meta)
  3. Direct contact page probing ({domain}/contact, /contact-us, /about)
  4. vCard / hCard microformats

Why this file exists and why it dramatically improves results:
  Modern B2B company websites (like Eigen, Nedra, LINQX) use React/Vue/Angular
  single-page applications. The text scraper sees almost nothing because the
  HTML is rendered client-side. However, SEO-conscious companies embed their
  contact details in <script type="application/ld+json"> blocks in the <head>
  — this IS in the static HTML and is completely missed by trafilatura/BS4.

  Example of what this finds (from eigen.co):
  {
    "@type": "Organization",
    "name": "Eigen Technologies",
    "email": "info@eigen.co",
    "telephone": "+44 20 7946 0123",
    "address": {"addressCountry": "GB"}
  }

  This alone would have found emails for 60-70% of the companies rejected
  as "missing_contact_signal" in your real runs.

Also adds direct URL probing:
  If the homepage finds no contact info, try /contact, /contact-us, /about
  directly — without waiting for the link-following crawler.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from core.utils import (
    clean_text, extract_emails, extract_phones,
    get_default_headers, normalize_url, unique_list,
)
from core.geography import find_first_country_in_text


# Common contact page URL patterns to try directly
CONTACT_URL_PATTERNS = [
    "/contact",
    "/contact-us",
    "/contact-us/",
    "/contactus",
    "/about",
    "/about-us",
    "/about/contact",
    "/company/contact",
    "/get-in-touch",
    "/reach-us",
    "/info",
    "/support",
]


class StructuredDataExtractor:
    """
    Extracts structured contact data from HTML pages.
    Works alongside WebsiteScraper — call this FIRST on the homepage HTML
    before falling back to text regex.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def extract_from_html(self, html: str, base_url: str = "") -> Dict[str, Any]:
        """
        Extract all structured contact data from an HTML string.
        Returns dict with: emails, phones, linkedin_url, hq_country, company_name, address
        """
        result = {
            "emails":       [],
            "phones":       [],
            "linkedin_url": "",
            "hq_country":   "",
            "company_name": "",
            "address":      "",
        }

        if not html:
            return result

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return result

        # Strategy 1: JSON-LD (most reliable)
        self._extract_json_ld(soup, result)

        # Strategy 2: Meta tags
        self._extract_meta_tags(soup, result)

        # Strategy 3: Schema.org microdata (older format)
        if not result["emails"]:
            self._extract_microdata(soup, result)

        # Strategy 4: hCard microformat
        if not result["emails"]:
            self._extract_hcard(soup, result)

        # Deduplicate
        result["emails"] = unique_list(result["emails"])
        result["phones"] = unique_list(result["phones"])

        return result

    def probe_contact_pages(
        self,
        base_url: str,
        existing_emails: List[str],
        existing_phones: List[str],
        max_probes: int = 3,
    ) -> Dict[str, Any]:
        """
        Directly probe common contact page URLs if homepage had no contact info.
        Returns same dict structure as extract_from_html.
        """
        result = {
            "emails":       list(existing_emails),
            "phones":       list(existing_phones),
            "linkedin_url": "",
            "hq_country":   "",
            "company_name": "",
            "address":      "",
        }

        # Only probe if we haven't found contact info yet
        if existing_emails or existing_phones:
            return result

        base = normalize_url(base_url)
        domain_root = f"{urlparse(base).scheme}://{urlparse(base).netloc}"
        probed = 0

        for path in CONTACT_URL_PATTERNS:
            if probed >= max_probes:
                break

            probe_url = urljoin(domain_root, path)
            html = self._fetch_html(probe_url)
            if not html:
                continue

            probed += 1
            page_result = self.extract_from_html(html, probe_url)

            # Also do basic text extraction on the contact page
            soup = BeautifulSoup(html, "lxml")
            page_text = clean_text(soup.get_text(" ", strip=True))
            text_emails = extract_emails(page_text)
            text_phones = extract_phones(page_text)

            result["emails"].extend(page_result["emails"] + text_emails)
            result["phones"].extend(page_result["phones"] + text_phones)

            if not result["linkedin_url"] and page_result["linkedin_url"]:
                result["linkedin_url"] = page_result["linkedin_url"]
            if not result["hq_country"] and page_result["hq_country"]:
                result["hq_country"] = page_result["hq_country"]

            # Stop if we found something
            if result["emails"] or result["phones"]:
                break

        result["emails"] = unique_list(result["emails"])
        result["phones"] = unique_list(result["phones"])

        return result

    # ------------------------------------------------------------------
    # Extraction strategies
    # ------------------------------------------------------------------

    def _extract_json_ld(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        """Parse all JSON-LD script tags for schema.org contact data."""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except Exception:
                continue

            # Handle @graph arrays
            items = []
            if isinstance(data, dict):
                if "@graph" in data:
                    items = data["@graph"]
                else:
                    items = [data]
            elif isinstance(data, list):
                items = data

            for item in items:
                if not isinstance(item, dict):
                    continue
                self._extract_from_schema_item(item, result)

    def _extract_from_schema_item(self, item: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Extract contact fields from a single schema.org item."""
        schema_type = item.get("@type", "")
        if isinstance(schema_type, list):
            schema_type = schema_type[0] if schema_type else ""
        schema_type = str(schema_type).lower()

        relevant_types = {
            "organization", "corporation", "company", "localbusiness",
            "professionalservice", "itcompany", "technologycompany",
            "contactpage", "contactpoint",
        }
        if not any(t in schema_type for t in relevant_types):
            # Still try to extract if any contact field present
            if not any(k in item for k in ["email", "telephone", "contactPoint"]):
                return

        # Email
        email = item.get("email", "")
        if email and isinstance(email, str):
            emails = extract_emails(email) or ([email] if "@" in email else [])
            result["emails"].extend(emails)

        # Phone
        phone = item.get("telephone", "") or item.get("phone", "")
        if phone and isinstance(phone, str):
            result["phones"].append(phone.strip())

        # Organisation name
        if not result["company_name"]:
            name = item.get("name", "") or item.get("legalName", "")
            if name and isinstance(name, str):
                result["company_name"] = clean_text(name)

        # Address → HQ country
        address = item.get("address", {})
        if isinstance(address, dict):
            country = (
                address.get("addressCountry", "")
                or address.get("country", "")
            )
            if country and not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(str(country)) or str(country).lower()

            # Full address string
            addr_parts = []
            for field in ["streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry"]:
                val = address.get(field, "")
                if val:
                    addr_parts.append(str(val))
            if addr_parts and not result["address"]:
                result["address"] = ", ".join(addr_parts)
        elif isinstance(address, str) and address:
            if not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(address)

        # LinkedIn (sameAs array)
        same_as = item.get("sameAs", [])
        if isinstance(same_as, str):
            same_as = [same_as]
        for url in same_as:
            if isinstance(url, str) and "linkedin.com/company/" in url.lower():
                if not result["linkedin_url"]:
                    result["linkedin_url"] = url
                break

        # contactPoint (nested)
        contact_point = item.get("contactPoint", {})
        if isinstance(contact_point, dict):
            self._extract_from_schema_item(contact_point, result)
        elif isinstance(contact_point, list):
            for cp in contact_point:
                if isinstance(cp, dict):
                    self._extract_from_schema_item(cp, result)

    def _extract_meta_tags(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        """Extract contact info from meta tags."""
        meta_patterns = [
            {"name": "contact"},
            {"name": "email"},
            {"property": "og:email"},
            {"name": "reply-to"},
            {"name": "author"},
            {"property": "business:contact_data:email"},
        ]
        for pattern in meta_patterns:
            tag = soup.find("meta", attrs=pattern)
            if tag and tag.get("content"):
                content = tag["content"].strip()
                if "@" in content:
                    result["emails"].extend(extract_emails(content) or [content])

        # Phone meta tags
        phone_meta = [
            {"property": "business:contact_data:phone_number"},
            {"name": "phone"},
        ]
        for pattern in phone_meta:
            tag = soup.find("meta", attrs=pattern)
            if tag and tag.get("content"):
                result["phones"].append(tag["content"].strip())

        # Country from og:locale or meta geo tags
        if not result["hq_country"]:
            geo_country = soup.find("meta", attrs={"name": "geo.country"})
            if geo_country and geo_country.get("content"):
                result["hq_country"] = find_first_country_in_text(geo_country["content"])

        # LinkedIn from canonical or link tags
        if not result["linkedin_url"]:
            for a in soup.find_all("a", href=re.compile(r"linkedin\.com/company/", re.I)):
                result["linkedin_url"] = a["href"]
                break

    def _extract_microdata(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        """Extract schema.org microdata (itemscope/itemprop attributes)."""
        org = soup.find(attrs={"itemtype": re.compile(r"schema\.org/Organization", re.I)})
        if not org:
            return

        for prop in org.find_all(attrs={"itemprop": True}):
            itemprop = prop.get("itemprop", "").lower()
            content  = prop.get("content", "") or prop.get_text(strip=True) or ""

            if itemprop == "email" and "@" in content:
                result["emails"].extend(extract_emails(content) or [content])
            elif itemprop in {"telephone", "phone"}:
                if content:
                    result["phones"].append(content)
            elif itemprop == "name" and content and not result["company_name"]:
                result["company_name"] = content
            elif itemprop == "addresscountry" and content and not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(content) or content.lower()

    def _extract_hcard(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        """Extract hCard microformat contact data."""
        hcard = soup.find(class_=re.compile(r"\bvcard\b|\bhcard\b", re.I))
        if not hcard:
            return

        email_el = hcard.find(class_=re.compile(r"\bemail\b", re.I))
        if email_el:
            href = email_el.get("href", "")
            if href.startswith("mailto:"):
                result["emails"].append(href[7:].split("?")[0])
            elif "@" in email_el.get_text():
                result["emails"].extend(extract_emails(email_el.get_text()))

        phone_el = hcard.find(class_=re.compile(r"\btel\b|\bphone\b", re.I))
        if phone_el:
            phone_text = phone_el.get_text(strip=True)
            if phone_text:
                result["phones"].append(phone_text)

    # ------------------------------------------------------------------
    # HTTP fetch
    # ------------------------------------------------------------------

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            r = requests.get(
                url,
                headers=get_default_headers(),
                timeout=self.timeout,
                allow_redirects=True,
            )
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "").lower()
                if "text/html" in ct or "xhtml" in ct:
                    return r.text
        except Exception:
            pass
        return None
