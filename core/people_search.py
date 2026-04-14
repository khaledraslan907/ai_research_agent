"""
people_search.py  —  v2 (final)
================================
LinkedIn people search via three complementary techniques:

1. EXA include_domains=["linkedin.com/in"]  ← BEST — returns real /in/ profiles
2. SerpApi site:linkedin.com/in             ← Good when SerpApi key available
3. DDG X-ray (fallback, low yield)          ← DDG ignores site: but still worth trying

KEY RULES:
- ONLY accept URLs containing linkedin.com/in/ (personal profiles)
- REJECT linkedin.com/company/ (company pages)
- NEVER scrape LinkedIn pages — extract everything from the search snippet
- Topic = clean industry (e.g. "oil gas service") — strip job-level words
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from core.models import SearchQuery


# ---------------------------------------------------------------------------
# Job title banks by level — used to build targeted queries
# ---------------------------------------------------------------------------

JOB_TITLES: Dict[str, List[str]] = {
    "engineer": [
        "Petroleum Engineer",
        "Reservoir Engineer",
        "Drilling Engineer",
        "Production Engineer",
        "Completion Engineer",
        "Wellsite Engineer",
        "Process Engineer",
        "Mechanical Engineer",
        "Electrical Engineer",
        "Instrumentation Engineer",
        "HSE Engineer",
        "Subsurface Engineer",
        "Geologist",
        "Geophysicist",
        "Facilities Engineer",
        "Pipeline Engineer",
        "Corrosion Engineer",
        "Reliability Engineer",
    ],
    "manager": [
        "Operations Manager",
        "Project Manager",
        "Technical Manager",
        "Drilling Manager",
        "Production Manager",
        "Field Operations Manager",
        "Engineering Manager",
        "Maintenance Manager",
        "Site Manager",
        "Account Manager",
        "Business Development Manager",
        "Country Manager",
        "District Manager",
        "Area Manager",
    ],
    "director": [
        "Director",
        "Vice President",
        "VP Operations",
        "VP Engineering",
        "Technical Director",
        "Operations Director",
        "Managing Director",
        "General Manager",
        "Country Director",
        "Regional Director",
    ],
    "hr": [
        "HR Manager",
        "Human Resources Manager",
        "Talent Acquisition Manager",
        "Recruitment Manager",
        "HR Business Partner",
        "HR Director",
        "People Operations Manager",
        "Recruiter",
        "Talent Acquisition Specialist",
        "HR Specialist",
        "Learning and Development Manager",
    ],
    "executive": [
        "CEO",
        "Chief Executive Officer",
        "COO",
        "Chief Operating Officer",
        "CTO",
        "Chief Technology Officer",
        "CFO",
        "Chief Financial Officer",
        "President",
        "Executive Director",
    ],
}

# Country → geo phrases used in queries
COUNTRY_GEO: Dict[str, List[str]] = {
    "egypt":                    ["Egypt", "Cairo", "Alexandria"],
    "usa":                      ["United States", "Houston", "Texas"],
    "saudi arabia":             ["Saudi Arabia", "Riyadh", "Dhahran"],
    "united arab emirates":     ["UAE", "Dubai", "Abu Dhabi"],
    "qatar":                    ["Qatar", "Doha"],
    "kuwait":                   ["Kuwait"],
    "oman":                     ["Oman", "Muscat"],
    "iraq":                     ["Iraq", "Basra"],
    "nigeria":                  ["Nigeria", "Lagos"],
    "angola":                   ["Angola", "Luanda"],
    "norway":                   ["Norway", "Stavanger"],
    "united kingdom":           ["United Kingdom", "Aberdeen", "London"],
    "australia":                ["Australia", "Perth"],
    "canada":                   ["Canada", "Calgary"],
    "brazil":                   ["Brazil", "Rio de Janeiro"],
    "indonesia":                ["Indonesia", "Jakarta"],
    "malaysia":                 ["Malaysia", "Kuala Lumpur"],
}


def build_linkedin_queries(
    industry: str,
    job_levels: List[str],
    countries: List[str],
    max_results: int = 50,
    use_serpapi: bool = False,
) -> Dict[str, List[SearchQuery]]:
    """
    Build queries that return linkedin.com/in/ personal profile URLs.

    Strategy priority:
    1. EXA with include_domains=["linkedin.com/in"]  → tagged as family="linkedin_exa"
    2. SerpApi site:linkedin.com/in queries          → family="linkedin_serp"
    3. DDG fallback (low yield, DDG bad at site:)    → family="linkedin_ddg"

    The orchestrator checks family tag to route EXA queries through
    exa.search_linkedin_profiles() instead of exa.search().
    """
    # Get the flat list of titles to search
    titles = _titles_for_levels(job_levels)

    # Get geo terms
    geo_terms: List[str] = []
    for c in countries:
        geo_terms.extend(COUNTRY_GEO.get(c.lower(), [c.title()])[:2])
    if not geo_terms:
        geo_terms = [""]

    # Scale query count with max_results
    n_titles = min(len(titles), max(3, max_results // 8))
    titles   = titles[:n_titles]

    exa_queries:     List[SearchQuery] = []
    serpapi_queries: List[SearchQuery] = []
    ddg_queries:     List[SearchQuery] = []
    p = 1

    # ── EXA: semantic queries with include_domains=linkedin.com/in ─────────
    # These are natural language queries — Exa's neural engine finds profiles
    # Family "linkedin_exa" tells orchestrator to use search_linkedin_profiles()
    for title in titles:
        for geo in geo_terms[:2]:
            geo_part = f" in {geo}" if geo else ""
            exa_queries.append(SearchQuery(
                text=(
                    f"{title} working at {industry} company{geo_part}"
                ),
                priority=p,
                family="linkedin_exa",
                provider_hint="exa",
            ))
            p += 1

    # Also add broader level-based EXA queries for diversity
    for level in job_levels[:3]:
        label = _level_label(level)
        for geo in geo_terms[:2]:
            geo_part = f" in {geo}" if geo else ""
            exa_queries.append(SearchQuery(
                text=f"{label} professional with experience in {industry}{geo_part}",
                priority=p,
                family="linkedin_exa",
                provider_hint="exa",
            ))
            p += 1

    # ── SerpApi: site:linkedin.com/in with quoted title + geo ──────────────
    if use_serpapi:
        for title in titles[:8]:
            for geo in geo_terms[:2]:
                geo_part = f' "{geo}"' if geo else ""
                serpapi_queries.append(SearchQuery(
                    text=f'site:linkedin.com/in "{title}" "{industry}"{geo_part}',
                    priority=p,
                    family="linkedin_serp",
                    provider_hint="serpapi",
                ))
                p += 1

    # ── DDG fallback: site:linkedin.com/in (low yield but free) ────────────
    # DDG partially supports site: — results are sparse but occasionally work
    for title in titles[:4]:
        geo = geo_terms[0] if geo_terms else ""
        geo_part = f' "{geo}"' if geo else ""
        ddg_queries.append(SearchQuery(
            text=f'site:linkedin.com/in "{title}" "{industry}"{geo_part}',
            priority=p,
            family="linkedin_ddg",
            provider_hint="ddg",
        ))
        p += 1

    return {
        "exa":     exa_queries,
        "serpapi": serpapi_queries,
        "ddg":     ddg_queries,
        "tavily":  [],  # Tavily not suited for site: profile searches
    }


def is_linkedin_profile_url(url: str) -> bool:
    """Return True only for personal profile URLs (linkedin.com/in/), not company pages."""
    url_lower = url.lower()
    return (
        "linkedin.com/in/" in url_lower
        and "linkedin.com/company" not in url_lower
    )


def is_linkedin_company_url(url: str) -> bool:
    """Return True for linkedin.com/company/ pages."""
    return "linkedin.com/company" in url.lower()


def extract_person_from_linkedin_result(
    title: str,
    url: str,
    snippet: str,
) -> Dict:
    """
    Extract structured person info from a LinkedIn search result.
    Works from snippet alone — no page scraping needed.

    LinkedIn search results typically look like:
      Title:   "Ahmed Hassan - Petroleum Engineer at Schlumberger | LinkedIn"
      Snippet: "Cairo, Egypt · Petroleum Engineer at Schlumberger · 500+ connections"
    """
    result = {
        "name":           "",
        "job_title":      "",
        "employer":       "",
        "location":       "",
        "linkedin_url":   url if is_linkedin_profile_url(url) else "",
        "is_profile":     is_linkedin_profile_url(url),
    }

    # ── Name: from URL slug (most reliable) ───────────────────────────────
    url_m = re.search(r"linkedin\.com/in/([a-zA-Z0-9][a-zA-Z0-9\-]+)", url)
    if url_m:
        slug = url_m.group(1)
        # "ahmed-hassan-petroleum-1234" → "Ahmed Hassan"
        parts = slug.split("-")
        name_parts = []
        for p in parts:
            if p.isdigit() or len(p) <= 1:
                break
            name_parts.append(p.capitalize())
            if len(name_parts) >= 3:
                break
        if len(name_parts) >= 2:
            result["name"] = " ".join(name_parts)

    # ── Better name: from title "Name - Title at Company | LinkedIn" ──────
    clean_title = re.sub(r"\s*[\|\-–]\s*LinkedIn.*$", "", title, flags=re.I).strip()
    clean_title = re.sub(r"\s*\|\s*.*$", "", clean_title).strip()

    if " - " in clean_title:
        segments = clean_title.split(" - ", 2)
        candidate_name = segments[0].strip()
        # Validate: looks like a real name (2 words, no digits, not too long)
        words = candidate_name.split()
        if (
            2 <= len(words) <= 4
            and all(w[0].isupper() for w in words if w)
            and not any(c.isdigit() for c in candidate_name)
        ):
            result["name"] = candidate_name

        # Job title and employer from remaining segments
        if len(segments) >= 2:
            job_part = segments[1].strip()
            # "Petroleum Engineer at Schlumberger"
            if " at " in job_part.lower():
                jt_m = re.split(r"\s+at\s+", job_part, maxsplit=1, flags=re.I)
                result["job_title"] = jt_m[0].strip()
                result["employer"]  = jt_m[1].strip() if len(jt_m) > 1 else ""
            else:
                result["job_title"] = job_part

    # ── Location: from snippet ─────────────────────────────────────────────
    loc_patterns = [
        r"([A-Z][a-z\s]+,\s*(?:Egypt|USA|Saudi Arabia|UAE|UK|Norway|Canada|Australia|Iraq|Kuwait|Qatar|Oman|Nigeria))",
        r"(Cairo|Alexandria|Houston|Texas|Riyadh|Dubai|Abu Dhabi|London|Oslo|Calgary|Lagos|Muscat|Doha|Kuwait City)",
        r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s*[A-Z][a-z]+)?)\s*·",  # "Cairo, Egypt ·"
    ]
    for pat in loc_patterns:
        m = re.search(pat, snippet or "")
        if m:
            result["location"] = m.group(1).strip().rstrip("·").strip()
            break

    # ── Employer from snippet if not found in title ────────────────────────
    if not result["employer"]:
        emp_m = re.search(
            r"(?:at|@|with|working at|employed at)\s+([A-Z][A-Za-z\s\&\-\.]+?)(?:\s*·|\s*\||$)",
            snippet or "", re.I
        )
        if emp_m:
            result["employer"] = emp_m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _titles_for_levels(job_levels: List[str]) -> List[str]:
    """Get job titles for the requested levels, interleaved for diversity."""
    all_titles: List[str] = []
    max_per_level = 4
    for level in job_levels:
        level_lower = level.lower()
        for key, titles in JOB_TITLES.items():
            if key in level_lower or level_lower in key:
                all_titles.extend(titles[:max_per_level])
                break
    # Default if nothing matched
    if not all_titles:
        all_titles = (
            JOB_TITLES["engineer"][:3]
            + JOB_TITLES["manager"][:3]
            + JOB_TITLES["hr"][:2]
        )
    # Deduplicate preserving order
    seen: set = set()
    unique = []
    for t in all_titles:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique


def _level_label(level: str) -> str:
    labels = {
        "engineer":  "engineering and technical",
        "manager":   "management and supervisory",
        "director":  "director and VP",
        "hr":        "HR and talent acquisition",
        "executive": "executive and C-suite",
    }
    for key, label in labels.items():
        if key in level.lower():
            return label
    return level
