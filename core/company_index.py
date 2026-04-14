from __future__ import annotations

import re
from typing import Optional, Set

import pandas as pd

from core.normalizer import are_company_names_similar, _clean_name
from core.utils import extract_domain


class CompanyIndex:
    """
    In-memory index of already-known companies / domains / people
    used for deduplication against uploaded seed files and
    previously accepted results within a single run.

    For people_search: uses linkedin_url + person name as keys.
    For company/paper search: uses domain + company name as keys.
    """

    def __init__(self):
        self._domains:       Set[str] = set()   # company domains
        self._names:         list[str] = []      # company names
        self._linkedin_urls: Set[str] = set()   # full linkedin.com/in/... URLs
        self._person_names:  Set[str] = set()   # "firstname lastname" lowercased

    # ------------------------------------------------------------------
    # Load seed file
    # ------------------------------------------------------------------

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load known entities from an uploaded seed DataFrame."""

        # ── Company domains ──────────────────────────────────────────
        for col in ["domain", "website", "url", "Website", "URL", "Domain"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    d = extract_domain(val.strip())
                    if d and "linkedin.com" not in d:
                        self._domains.add(d.lower())

        # ── Company names ────────────────────────────────────────────
        for col in ["company_name", "name", "Company", "Company Name", "Name"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    n = _clean_name(val.strip())
                    if n:
                        self._names.append(n)

        # ── LinkedIn profile URLs (people search seed) ────────────────
        for col in ["linkedin_url", "linkedin_profile", "LinkedIn URL",
                    "LinkedIn Profile", "profile_url", "url"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    url = val.strip().lower().rstrip("/")
                    if "linkedin.com/in/" in url:
                        self._linkedin_urls.add(url)

        # ── Person names (people search seed) ────────────────────────
        for col in ["company_name", "name", "full_name", "person_name",
                    "Name", "Full Name", "Person"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    # Only treat as person name if it looks like "First Last"
                    v = val.strip()
                    words = v.split()
                    if (
                        2 <= len(words) <= 4
                        and all(w[0].isupper() for w in words if w)
                        and not any(c.isdigit() for c in v)
                    ):
                        self._person_names.add(v.lower())

    # ------------------------------------------------------------------
    # Add a result record to the index
    # ------------------------------------------------------------------

    def add_record(self, record) -> None:
        # Person profile
        linkedin = getattr(record, "linkedin_url", "") or getattr(record, "linkedin_profile", "")
        if linkedin and "linkedin.com/in/" in linkedin.lower():
            self._linkedin_urls.add(linkedin.lower().rstrip("/"))
            name = (record.company_name or "").strip().lower()
            if name and len(name.split()) >= 2:
                self._person_names.add(name)
            return  # don't add linkedin.com as a company domain

        # Company / paper
        if record.domain and "linkedin.com" not in record.domain:
            self._domains.add(record.domain.lower())
        n = _clean_name(record.company_name)
        if n and n not in self._names:
            self._names.append(n)

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def contains_linkedin_profile(self, url: str, name: str = "") -> bool:
        """Return True if this LinkedIn profile URL or person name is already known."""
        if not url:
            return False
        url_norm = url.lower().rstrip("/")
        if url_norm in self._linkedin_urls:
            return True
        # Also check by name (catches same person at slightly different URL)
        if name:
            name_lower = name.strip().lower()
            if name_lower and len(name_lower.split()) >= 2:
                if name_lower in self._person_names:
                    return True
        return False

    def contains_company(self, name: Optional[str], domain: Optional[str]) -> bool:
        """Return True if this company/paper is already known."""
        # Never block on linkedin.com domain — that would block all profiles
        if domain and "linkedin.com" in domain.lower():
            return False
        if domain and domain.lower() in self._domains:
            return True
        cleaned = _clean_name(name or "")
        if not cleaned:
            return False
        return any(are_company_names_similar(cleaned, n) for n in self._names)

    def summary(self) -> dict:
        return {
            "known_domains":       len(self._domains),
            "known_companies":     len(self._names),
            "known_linkedin_urls": len(self._linkedin_urls),
            "known_persons":       len(self._person_names),
        }
