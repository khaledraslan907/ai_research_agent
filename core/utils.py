from __future__ import annotations

import re
from typing import List
from urllib.parse import urljoin, urlparse

import tldextract


def extract_domain(url: str) -> str:
    """
    Canonical registrable domain extractor.

    Examples:
    - https://www.company.com/about      -> company.com
    - company.com/contact                -> company.com
    - uk.company.co.uk/page              -> company.co.uk
    - WWW2.EXAMPLE.ORG                   -> example.org
    """
    if not url:
        return ""

    raw = (url or "").strip().lower()
    if not raw:
        return ""

    try:
        # Ensure parseable input
        parsed_input = raw if "://" in raw else f"https://{raw}"
        ext = tldextract.extract(parsed_input)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower().strip()
    except Exception:
        pass

    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        host = parsed.netloc or parsed.path
        host = host.lower().strip()
        host = re.sub(r"^www\d*\.", "", host)
        host = host.split(":")[0].split("/")[0].split("?")[0].split("#")[0].strip().rstrip(".")
        return host
    except Exception:
        return ""


def normalize_url(url: str) -> str:
    """
    Canonical URL normalizer for dedup / display.
    Keeps path, but normalizes:
    - missing scheme -> https
    - http -> https
    - strips www/www2
    - strips query / fragment
    - strips trailing slash
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
        scheme = "https"
        host = (parsed.netloc or parsed.path).lower().strip()
        host = re.sub(r"^www\d*\.", "", host)
        host = host.split(":")[0].strip().rstrip(".")
        path = (parsed.path or "").strip()
        if path == "/":
            path = ""
        normalized = f"{scheme}://{host}{path}"
        return normalized.rstrip("/")
    except Exception:
        return raw.rstrip("/")


def get_root_url(url: str) -> str:
    try:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return normalize_url(url)


def join_url(base: str, path: str) -> str:
    try:
        return urljoin(base, path)
    except Exception:
        return path


def get_default_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }


_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d[\d\s\-().]{7,18}\d)"
    r"(?!\d)"
)
_JUNK_DOMAINS = {
    "example.com", "test.com", "domain.com", "email.com",
    "yoursite.com", "website.com", "sentry.io", "wixpress.com",
}


def extract_emails(text: str) -> List[str]:
    found = _EMAIL_RE.findall(text or "")
    seen, result = set(), []
    for e in found:
        e = e.lower().strip()
        domain = e.split("@")[-1]
        if domain in _JUNK_DOMAINS:
            continue
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


def extract_phones(text: str) -> List[str]:
    found = _PHONE_RE.findall(text or "")
    seen, result = set(), []
    for p in found:
        p = re.sub(r"\s+", " ", p.strip())
        digits = re.sub(r"\D", "", p)
        if len(digits) < 7 or len(digits) > 15:
            continue
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def unique_list(lst: list) -> list:
    """
    Generic unique-preserving order helper with light canonicalization.
    Useful for mixed lists of strings like domains, URLs, countries, keywords.
    """
    seen, result = set(), []

    for item in lst:
        if item is None:
            continue

        raw = str(item).strip()
        if not raw:
            continue

        low = raw.lower()

        # URL-like
        if low.startswith(("http://", "https://")) or "/" in low:
            key = normalize_url(raw)
        # domain-like
        elif "." in low and " " not in low:
            key = extract_domain(raw) or low
        else:
            key = low.strip()

        if key and key not in seen:
            seen.add(key)
            result.append(item)

    return result


def looks_like_contact_link(text: str) -> bool:
    keywords = [
        "contact", "about", "team", "company", "corporate",
        "office", "location", "locations", "global-presence",
        "presence", "offices", "reach-us", "get-in-touch",
    ]
    t = text.lower()
    return any(k in t for k in keywords)


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text
