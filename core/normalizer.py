from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from core.geography import normalize_country_name as _geo_normalize


def normalize_country_name(name: str) -> str:
    return _geo_normalize(name)


_COMMON_LEGAL_SUFFIXES = {
    "inc", "inc.", "llc", "llc.", "ltd", "ltd.", "limited",
    "corp", "corp.", "corporation", "gmbh", "s.a.", "s.a.s", "sas",
    "b.v.", "bv", "n.v.", "nv", "ag", "plc", "co.", "co",
    "company", "group", "holding", "holdings", "international", "intl",
    "pty", "pty.", "pte", "pte.", "srl", "spa", "oy", "ab",
}

_NAME_NOISE_TOKENS = {
    "official", "homepage", "home", "website", "platform", "solution", "solutions",
    "technology", "technologies", "services", "service", "system", "systems",
}


def _extract_domain_simple(url: str) -> str:
    """
    Normalize a URL/domain into a canonical host:
    - lower case
    - remove protocol
    - remove www / www2 etc.
    - remove path/query/fragment/port
    """
    if not url:
        return ""

    host = url.lower().strip()
    host = re.sub(r"^https?://", "", host)
    host = re.sub(r"^www\d*\.", "", host)
    host = host.split("/")[0].split("?")[0].split("#")[0].strip()
    host = host.split(":")[0].strip().rstrip(".")

    return host


def _clean_name(name: str) -> str:
    """
    Normalize company names for matching:
    - lowercase
    - unify '&' to 'and'
    - remove punctuation
    - strip repeated legal suffixes at the end
    - strip some noisy tail tokens
    """
    s = (name or "").strip().lower()
    if not s:
        return ""

    s = s.replace("&", " and ")
    s = re.sub(r"[/|:,_()\[\]{}\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    words = s.split()

    # remove repeated legal suffixes / tail noise
    while words and words[-1] in _COMMON_LEGAL_SUFFIXES.union(_NAME_NOISE_TOKENS):
        words.pop()

    # remove a second layer like "... technologies ltd"
    while words and words[-1] in _COMMON_LEGAL_SUFFIXES:
        words.pop()

    s = " ".join(words)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _name_tokens(name: str) -> list[str]:
    return [w for w in _clean_name(name).split() if w]


def _name_acronym(name: str) -> str:
    tokens = _name_tokens(name)
    if len(tokens) < 2:
        return ""
    return "".join(t[0] for t in tokens if t and t[0].isalnum())


def company_key(name: str, website: str) -> str:
    """
    Produce a stable dedup key from website/domain if available,
    otherwise from normalized company name.
    """
    domain = _extract_domain_simple(website)
    if domain:
        return domain

    cleaned = _clean_name(name)
    return cleaned if cleaned else ""


def are_company_names_similar(a: str, b: str, threshold: int = 85) -> bool:
    """
    Safer company-name similarity:
    - exact normalized match
    - acronym match for short-vs-long forms
    - token-set similarity
    - containment with token overlap
    """
    ca = _clean_name(a)
    cb = _clean_name(b)

    if not ca or not cb:
        return False

    if ca == cb:
        return True

    ta = _name_tokens(a)
    tb = _name_tokens(b)

    if not ta or not tb:
        return False

    # Acronym logic: SLB vs Schlumberger-style cases are difficult,
    # but "abc" vs "alpha beta company" can be caught safely.
    if len(ca) <= 5 and len(tb) >= 2:
        if ca.replace(" ", "") == _name_acronym(b).lower():
            return True
    if len(cb) <= 5 and len(ta) >= 2:
        if cb.replace(" ", "") == _name_acronym(a).lower():
            return True

    # containment rule with token overlap
    sa = set(ta)
    sb = set(tb)
    overlap = len(sa & sb)

    if (ca in cb or cb in ca) and overlap >= max(1, min(len(sa), len(sb)) - 1):
        return True

    ts = fuzz.token_sort_ratio(ca, cb)
    tset = fuzz.token_set_ratio(ca, cb)
    partial = fuzz.partial_ratio(ca, cb)

    # safer combined logic
    if ts >= threshold:
        return True
    if tset >= 92:
        return True
    if partial >= 95 and overlap >= 2:
        return True

    return False


def prefer_best_company_name(a: str, b: str) -> str:
    """
    Pick the better-looking display name.
    Prefer:
    - non-empty
    - title-looking names over lowercase/uppercase-only
    - shorter clean company names over long page titles
    - names without URL/path fragments
    """
    a = (a or "").strip()
    b = (b or "").strip()

    if not a:
        return b
    if not b:
        return a

    def _score(s: str) -> int:
        sc = 0
        st = s.strip()

        if not st:
            return -999

        if "http://" in st.lower() or "https://" in st.lower() or "/" in st:
            sc -= 6

        if len(st) <= 60:
            sc += 3
        elif len(st) <= 90:
            sc += 1
        else:
            sc -= 3

        if st[0].isupper():
            sc += 2
        if not st.isupper() and not st.islower():
            sc += 2

        # penalize long title-ish strings
        if any(sep in st for sep in [" | ", " - ", " :: ", " — "]):
            sc -= 2

        cleaned = _clean_name(st)
        token_count = len(cleaned.split())
        if 1 <= token_count <= 5:
            sc += 3
        elif token_count > 8:
            sc -= 2

        if cleaned and cleaned != st.lower():
            sc += 1

        return sc

    return a if _score(a) >= _score(b) else b
