from __future__ import annotations

import re
from typing import List
from urllib.parse import urljoin, urlparse

import tldextract


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.netloc or parsed.path
        host = re.sub(r"^www\.", "", host).split(":")[0]
        return host.lower()
    except Exception:
        return ""


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def get_root_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return url


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
    seen, result = set(), []
    for item in lst:
        key = str(item).lower().strip()
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
