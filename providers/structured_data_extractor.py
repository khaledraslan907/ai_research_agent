"""
Extract structured contact/location data from modern websites.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from core.geography import find_countries_in_text, find_first_country_in_text
from core.utils import clean_text, extract_emails, extract_phones, get_default_headers, normalize_url, unique_list

CONTACT_URL_PATTERNS = [
    "/contact", "/contact-us", "/contact-us/", "/contactus", "/about", "/about-us",
    "/about/contact", "/company/contact", "/get-in-touch", "/reach-us", "/info", "/support",
]


class StructuredDataExtractor:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def extract_from_html(self, html: str, base_url: str = "") -> Dict[str, Any]:
        result = {
            "emails": [],
            "phones": [],
            "linkedin_url": "",
            "hq_country": "",
            "company_name": "",
            "address": "",
            "presence_countries": [],
        }
        if not html:
            return result
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return result
        self._extract_json_ld(soup, result)
        self._extract_meta_tags(soup, result)
        if not result["emails"]:
            self._extract_microdata(soup, result)
        if not result["emails"]:
            self._extract_hcard(soup, result)
        result["emails"] = unique_list(result["emails"])
        result["phones"] = unique_list(result["phones"])
        result["presence_countries"] = unique_list(result["presence_countries"])
        return result

    def probe_contact_pages(self, base_url: str, existing_emails: List[str], existing_phones: List[str], max_probes: int = 3) -> Dict[str, Any]:
        result = {
            "emails": list(existing_emails),
            "phones": list(existing_phones),
            "linkedin_url": "",
            "hq_country": "",
            "company_name": "",
            "address": "",
            "presence_countries": [],
        }
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
            soup = BeautifulSoup(html, "lxml")
            page_text = clean_text(soup.get_text(" ", strip=True))
            result["emails"].extend(page_result["emails"] + extract_emails(page_text))
            result["phones"].extend(page_result["phones"] + extract_phones(page_text))
            result["presence_countries"].extend(page_result.get("presence_countries", []))
            if not result["linkedin_url"] and page_result["linkedin_url"]:
                result["linkedin_url"] = page_result["linkedin_url"]
            if not result["hq_country"] and page_result["hq_country"]:
                result["hq_country"] = page_result["hq_country"]
            if result["emails"] or result["phones"]:
                break
        result["emails"] = unique_list(result["emails"])
        result["phones"] = unique_list(result["phones"])
        result["presence_countries"] = unique_list(result["presence_countries"])
        return result

    def _extract_json_ld(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except Exception:
                continue
            items = []
            if isinstance(data, dict):
                items = data.get("@graph", [data]) if "@graph" in data else [data]
            elif isinstance(data, list):
                items = data
            for item in items:
                if isinstance(item, dict):
                    self._extract_from_schema_item(item, result)

    def _extract_from_schema_item(self, item: Dict[str, Any], result: Dict[str, Any]) -> None:
        schema_type = item.get("@type", "")
        if isinstance(schema_type, list):
            schema_type = schema_type[0] if schema_type else ""
        schema_type = str(schema_type).lower()
        relevant_types = {"organization", "corporation", "company", "localbusiness", "professionalservice", "contactpoint"}
        if not any(t in schema_type for t in relevant_types) and not any(k in item for k in ["email", "telephone", "contactPoint"]):
            return
        email = item.get("email", "")
        if isinstance(email, str) and email:
            result["emails"].extend(extract_emails(email) or ([email] if "@" in email else []))
        phone = item.get("telephone", "") or item.get("phone", "")
        if isinstance(phone, str) and phone:
            result["phones"].append(phone.strip())
        if not result["company_name"]:
            name = item.get("name", "") or item.get("legalName", "")
            if isinstance(name, str) and name:
                result["company_name"] = clean_text(name)
        address = item.get("address", {})
        if isinstance(address, dict):
            country = address.get("addressCountry", "") or address.get("country", "")
            if country and not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(str(country)) or str(country).lower()
            addr_parts = []
            for field in ["streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry"]:
                val = address.get(field, "")
                if val:
                    addr_parts.append(str(val))
            if addr_parts and not result["address"]:
                result["address"] = ", ".join(addr_parts)
                result["presence_countries"].extend(find_countries_in_text(result["address"]))
        elif isinstance(address, str) and address:
            if not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(address)
            result["presence_countries"].extend(find_countries_in_text(address))
        same_as = item.get("sameAs", [])
        if isinstance(same_as, str):
            same_as = [same_as]
        for url in same_as:
            if isinstance(url, str) and "linkedin.com" in url.lower() and not result["linkedin_url"]:
                result["linkedin_url"] = normalize_url(url)
        for key in ["areaServed", "location", "serviceArea"]:
            val = item.get(key)
            self._extract_geo_from_value(val, result)
        contact_point = item.get("contactPoint", [])
        if isinstance(contact_point, dict):
            contact_point = [contact_point]
        for cp in contact_point or []:
            if not isinstance(cp, dict):
                continue
            email = cp.get("email", "")
            tel = cp.get("telephone", "")
            area = cp.get("areaServed")
            if isinstance(email, str) and email:
                result["emails"].extend(extract_emails(email) or ([email] if "@" in email else []))
            if isinstance(tel, str) and tel:
                result["phones"].append(tel.strip())
            self._extract_geo_from_value(area, result)

    def _extract_geo_from_value(self, val: Any, result: Dict[str, Any]) -> None:
        if not val:
            return
        if isinstance(val, str):
            result["presence_countries"].extend(find_countries_in_text(val))
        elif isinstance(val, list):
            for item in val:
                self._extract_geo_from_value(item, result)
        elif isinstance(val, dict):
            text = " ".join(str(v) for v in val.values() if isinstance(v, (str, int, float)))
            result["presence_countries"].extend(find_countries_in_text(text))

    def _extract_meta_tags(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        for tag in soup.find_all("meta"):
            key = (tag.get("name") or tag.get("property") or "").lower().strip()
            content = clean_text(tag.get("content") or "")
            if not content:
                continue
            if "email" in key:
                result["emails"].extend(extract_emails(content))
            if "phone" in key or "telephone" in key:
                result["phones"].extend(extract_phones(content))
            if "linkedin" in key and "linkedin.com" in content and not result["linkedin_url"]:
                result["linkedin_url"] = normalize_url(content)
            if any(x in key for x in ["location", "country", "region", "address"]):
                countries = find_countries_in_text(content)
                result["presence_countries"].extend(countries)
                if countries and not result["hq_country"]:
                    result["hq_country"] = countries[0]

    def _extract_microdata(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        for node in soup.select("[itemprop='email'], [itemprop='telephone'], [itemprop='address'], [itemprop='areaServed']"):
            text = clean_text(node.get_text(" ", strip=True) or node.get("content") or "")
            if node.get("itemprop") == "email":
                result["emails"].extend(extract_emails(text))
            elif node.get("itemprop") == "telephone":
                result["phones"].extend(extract_phones(text))
            else:
                result["presence_countries"].extend(find_countries_in_text(text))
                if not result["hq_country"]:
                    result["hq_country"] = find_first_country_in_text(text)

    def _extract_hcard(self, soup: BeautifulSoup, result: Dict[str, Any]) -> None:
        for node in soup.select(".vcard, .h-card"):
            text = clean_text(node.get_text(" ", strip=True))
            result["emails"].extend(extract_emails(text))
            result["phones"].extend(extract_phones(text))
            result["presence_countries"].extend(find_countries_in_text(text))
            if not result["hq_country"]:
                result["hq_country"] = find_first_country_in_text(text)

    def _fetch_html(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=get_default_headers(), timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ct and "application/xhtml+xml" not in ct:
                return ""
            return resp.text
        except Exception:
            return ""
