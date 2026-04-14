from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from core.geography import normalize_country_name as _geo_normalize


def normalize_country_name(name: str) -> str:
    return _geo_normalize(name)


def _clean_name(name: str) -> str:
    name = (name or "").strip().lower()
    # remove common suffixes
    for suffix in [
        " inc", " inc.", " llc", " llc.", " ltd", " ltd.", " limited",
        " corp", " corp.", " corporation", " gmbh", " s.a.", " s.a.s",
        " b.v.", " n.v.", " ag", " plc", " co.", " co", " group",
        " holding", " holdings", " international", " intl",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"\s+", " ", name).strip()


def company_key(name: str, website: str) -> str:
    """Produce a stable dedup key from name or domain."""
    domain = _extract_domain_simple(website)
    if domain:
        return domain
    cleaned = _clean_name(name)
    return cleaned if cleaned else ""


def _extract_domain_simple(url: str) -> str:
    if not url:
        return ""
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.split("/")[0].split("?")[0].strip()


def are_company_names_similar(a: str, b: str, threshold: int = 85) -> bool:
    ca = _clean_name(a)
    cb = _clean_name(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    return fuzz.token_sort_ratio(ca, cb) >= threshold


def prefer_best_company_name(a: str, b: str) -> str:
    """Pick the better-looking name (longer, title-cased, no trailing junk)."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    # prefer the one that is not all lowercase / all upper
    def _score(s: str) -> int:
        sc = 0
        if s[0].isupper():
            sc += 2
        if len(s) > 4:
            sc += 1
        if not s.isupper() and not s.islower():
            sc += 2
        return sc
    return a if _score(a) >= _score(b) else b
