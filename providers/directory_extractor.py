from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from core.models import CompanyRecord
from core.utils import clean_text, extract_domain, normalize_url


_COMPANY_HINTS = re.compile(r"(?i)\b(company|services|solutions|engineering|technology|systems|group|ltd|llc|inc|corp)\b")


def extract_company_candidates_from_html(html: str, base_url: str = "") -> List[CompanyRecord]:
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    records: List[CompanyRecord] = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = clean_text(link.get_text(" ", strip=True))
        if not text or len(text) < 3:
            continue
        if not _COMPANY_HINTS.search(text) and len(text.split()) > 6:
            continue
        abs_url = normalize_url(urljoin(base_url, href)) if base_url else normalize_url(href)
        dom = extract_domain(abs_url)
        key = (text.lower(), dom)
        if key in seen:
            continue
        seen.add(key)
        records.append(CompanyRecord(
            company_name=text,
            website=abs_url,
            domain=dom,
            source_url=base_url,
            page_type="directory",
            is_directory_or_media=True,
        ))
    return records


def extract_company_names_from_text(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        line = clean_text(line)
        if not line or len(line) > 120:
            continue
        if _COMPANY_HINTS.search(line):
            out.append(line)
    seen = set()
    uniq: List[str] = []
    for item in out:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            uniq.append(item)
    return uniq
