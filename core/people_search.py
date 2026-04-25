"""
people_search.py
================
LinkedIn people-search helpers.

Design goals:
- Build profile-oriented queries, not generic company queries.
- Prefer EXA semantic retrieval, then SerpApi, then DDG fallback.
- Never scrape LinkedIn pages directly.
- Extract structured person information from search snippets only.
- Remain generic across industries, while still working well in oil & gas.
"""
from __future__ import annotations

import re
from typing import Dict, List

from core.models import SearchQuery


JOB_TITLES: Dict[str, List[str]] = {
    "engineer": [
        "Petroleum Engineer", "Reservoir Engineer", "Drilling Engineer", "Production Engineer",
        "Completion Engineer", "Process Engineer", "Mechanical Engineer", "Electrical Engineer",
        "Instrumentation Engineer", "Reliability Engineer", "Pipeline Engineer", "Facilities Engineer",
        "Software Engineer", "Data Engineer", "Automation Engineer", "Field Engineer",
    ],
    "manager": [
        "Operations Manager", "Project Manager", "Engineering Manager", "Maintenance Manager",
        "Production Manager", "Drilling Manager", "Country Manager", "Area Manager",
        "Business Development Manager", "Account Manager", "Site Manager",
    ],
    "director": [
        "Director", "Technical Director", "Operations Director", "Managing Director",
        "Regional Director", "Country Director", "VP Engineering", "VP Operations",
    ],
    "hr": [
        "HR Manager", "Human Resources Manager", "Talent Acquisition Manager",
        "Recruitment Manager", "HR Director", "Recruiter", "Talent Acquisition Specialist",
    ],
    "executive": [
        "CEO", "Chief Executive Officer", "COO", "Chief Operating Officer",
        "CTO", "Chief Technology Officer", "CFO", "Chief Financial Officer", "President",
    ],
    "research": [
        "Research Scientist", "Research Engineer", "Principal Scientist",
        "Professor", "Associate Professor", "Postdoctoral Researcher", "R&D Manager",
    ],
    "sales": [
        "Sales Manager", "Sales Director", "Business Development Director",
        "Regional Sales Manager", "Account Executive", "Commercial Manager",
    ],
}

COUNTRY_GEO: Dict[str, List[str]] = {
    "egypt": ["Egypt", "Cairo", "Alexandria", "مصر", "القاهرة"],
    "saudi arabia": ["Saudi Arabia", "Riyadh", "Dhahran", "السعودية", "الرياض"],
    "united arab emirates": ["UAE", "Dubai", "Abu Dhabi", "الإمارات", "دبي"],
    "qatar": ["Qatar", "Doha", "قطر", "الدوحة"],
    "oman": ["Oman", "Muscat", "عمان", "مسقط"],
    "kuwait": ["Kuwait", "الكويت"],
    "bahrain": ["Bahrain", "البحرين"],
    "iraq": ["Iraq", "Basra", "العراق", "البصرة"],
    "norway": ["Norway", "Stavanger", "النرويج"],
    "united kingdom": ["United Kingdom", "Aberdeen", "London", "المملكة المتحدة", "بريطانيا"],
    "united states": ["United States", "Houston", "Texas", "USA"],
    "canada": ["Canada", "Calgary"],
    "australia": ["Australia", "Perth"],
    "india": ["India", "Mumbai", "Bengaluru"],
    "germany": ["Germany", "Berlin"],
    "netherlands": ["Netherlands", "Amsterdam"],
    "france": ["France", "Paris"],
    "brazil": ["Brazil", "Rio de Janeiro"],
    "nigeria": ["Nigeria", "Lagos"],
    "angola": ["Angola", "Luanda"],
    "indonesia": ["Indonesia", "Jakarta"],
    "malaysia": ["Malaysia", "Kuala Lumpur"],
}

_JOB_WORDS = {
    "engineer", "manager", "director", "recruiter", "talent", "human resources",
    "hr", "executive", "ceo", "coo", "cto", "cfo", "president", "professor",
}

_GENERIC_COMPANY_WORDS = {
    "company", "companies", "vendor", "vendors", "provider", "providers", "firm", "firms",
    "contractor", "contractors", "organization", "organisations", "business", "businesses",
}


def build_linkedin_queries(
    industry: str,
    job_levels: List[str],
    countries: List[str],
    max_results: int = 50,
    use_serpapi: bool = False,
) -> Dict[str, List[SearchQuery]]:
    """
    Build provider-specific queries that target personal LinkedIn profiles.
    """
    clean_industry = clean_people_topic(industry)
    titles = _titles_for_levels(job_levels)
    geo_terms: List[str] = []
    for c in countries:
        geo_terms.extend(COUNTRY_GEO.get(c.lower(), [c.title()])[:2])
    if not geo_terms:
        geo_terms = [""]

    n_titles = min(len(titles), max(3, max_results // 8))
    titles = titles[:n_titles]

    exa_queries: List[SearchQuery] = []
    serpapi_queries: List[SearchQuery] = []
    ddg_queries: List[SearchQuery] = []
    priority = 1

    for title in titles:
        for geo in geo_terms[:2]:
            geo_part = f" in {geo}" if geo else ""
            exa_queries.append(SearchQuery(
                text=f"{title} working at {clean_industry} company{geo_part}",
                priority=priority,
                family="linkedin_exa",
                provider_hint="exa",
            ))
            priority += 1

    for level in job_levels[:3]:
        label = _level_label(level)
        for geo in geo_terms[:2]:
            geo_part = f" in {geo}" if geo else ""
            exa_queries.append(SearchQuery(
                text=f"{label} professional with experience in {clean_industry}{geo_part}",
                priority=priority,
                family="linkedin_exa",
                provider_hint="exa",
            ))
            priority += 1

    if use_serpapi:
        for title in titles[:8]:
            for geo in geo_terms[:2]:
                geo_part = f' "{geo}"' if geo else ""
                serpapi_queries.append(SearchQuery(
                    text=f'site:linkedin.com/in "{title}" "{clean_industry}"{geo_part}',
                    priority=priority,
                    family="linkedin_serp",
                    provider_hint="serpapi",
                ))
                priority += 1

    for title in titles[:4]:
        geo = geo_terms[0] if geo_terms else ""
        geo_part = f' "{geo}"' if geo else ""
        ddg_queries.append(SearchQuery(
            text=f'site:linkedin.com/in "{title}" "{clean_industry}"{geo_part}',
            priority=priority,
            family="linkedin_ddg",
            provider_hint="ddg",
        ))
        priority += 1

    return {
        "exa": exa_queries,
        "serpapi": serpapi_queries,
        "ddg": ddg_queries,
        "tavily": [],
    }


def clean_people_topic(industry: str) -> str:
    """
    Remove obvious job-level noise so LinkedIn profile queries stay focused on
    sector/domain keywords.
    """
    text = re.sub(r"\s+", " ", (industry or "").strip())
    if not text:
        return "industry"
    low = text.lower()

    tokens = []
    for token in re.split(r"[\s,/]+", low):
        if not token:
            continue
        if token in _JOB_WORDS or token in _GENERIC_COMPANY_WORDS:
            continue
        tokens.append(token)

    cleaned = " ".join(tokens).strip()
    return cleaned or low or "industry"


def is_linkedin_profile_url(url: str) -> bool:
    url_lower = (url or "").lower()
    return "linkedin.com/in/" in url_lower and "linkedin.com/company/" not in url_lower


def is_linkedin_company_url(url: str) -> bool:
    return "linkedin.com/company/" in (url or "").lower()


def extract_person_from_linkedin_result(title: str, url: str, snippet: str) -> Dict:
    """
    Extract structured person information from a LinkedIn search result snippet.
    """
    result = {
        "name": "",
        "job_title": "",
        "employer": "",
        "location": "",
        "linkedin_url": url if is_linkedin_profile_url(url) else "",
        "is_profile": is_linkedin_profile_url(url),
    }

    url_m = re.search(r"linkedin\.com/in/([a-zA-Z0-9][a-zA-Z0-9\-]+)", url or "")
    if url_m:
        slug = url_m.group(1)
        parts = slug.split("-")
        name_parts = []
        for part in parts:
            if part.isdigit() or len(part) <= 1:
                break
            name_parts.append(part.capitalize())
            if len(name_parts) >= 3:
                break
        if len(name_parts) >= 2:
            result["name"] = " ".join(name_parts)

    clean_title = re.sub(r"\s*[\|\-–]\s*LinkedIn.*$", "", title or "", flags=re.I).strip()
    clean_title = re.sub(r"\s*\|\s*.*$", "", clean_title).strip()

    if " - " in clean_title:
        segments = clean_title.split(" - ", 2)
        candidate_name = segments[0].strip()
        words = candidate_name.split()
        if 2 <= len(words) <= 4 and not any(ch.isdigit() for ch in candidate_name):
            result["name"] = candidate_name

        if len(segments) >= 2:
            job_part = segments[1].strip()
            if " at " in job_part.lower():
                jt = re.split(r"\s+at\s+", job_part, maxsplit=1, flags=re.I)
                result["job_title"] = jt[0].strip()
                result["employer"] = jt[1].strip() if len(jt) > 1 else ""
            else:
                result["job_title"] = job_part

    loc_patterns = [
        r"([A-Z][a-z\s]+,\s*(?:Egypt|USA|Saudi Arabia|UAE|UK|Norway|Canada|Australia|Iraq|Kuwait|Qatar|Oman|Nigeria))",
        r"(Cairo|Alexandria|Houston|Texas|Riyadh|Dubai|Abu Dhabi|London|Oslo|Calgary|Lagos|Muscat|Doha|Kuwait City)",
        r"([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s*[A-Z][a-z]+)?)\s*·",
    ]
    for pat in loc_patterns:
        m = re.search(pat, snippet or "")
        if m:
            result["location"] = m.group(1).strip().rstrip("·").strip()
            break

    if not result["employer"]:
        emp_m = re.search(
            r"(?:at|@|with|working at|employed at)\s+([A-Z][A-Za-z\s&\-\.]+?)(?:\s*·|\s*\||$)",
            snippet or "", re.I
        )
        if emp_m:
            result["employer"] = emp_m.group(1).strip()

    return result


def _titles_for_levels(job_levels: List[str]) -> List[str]:
    all_titles: List[str] = []
    max_per_level = 4
    for level in job_levels or []:
        level_lower = level.lower()
        for key, titles in JOB_TITLES.items():
            if key in level_lower or level_lower in key:
                all_titles.extend(titles[:max_per_level])
                break

    if not all_titles:
        all_titles = JOB_TITLES["engineer"][:3] + JOB_TITLES["manager"][:3] + JOB_TITLES["hr"][:2]

    seen = set()
    unique: List[str] = []
    for title in all_titles:
        if title.lower() not in seen:
            seen.add(title.lower())
            unique.append(title)
    return unique


def _level_label(level: str) -> str:
    labels = {
        "engineer": "engineering and technical",
        "manager": "management and supervisory",
        "director": "director and VP",
        "hr": "HR and talent acquisition",
        "executive": "executive and C-suite",
        "research": "research and academic",
        "sales": "sales and business development",
    }
    for key, label in labels.items():
        if key in (level or "").lower():
            return label
    return level or "professional"
