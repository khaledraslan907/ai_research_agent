from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord, SearchResult
from core.utils import normalize_url, extract_domain, clean_text


PROFILE_RE = re.compile(r"linkedin\.com/in/", re.I)
COMPANY_RE = re.compile(r"linkedin\.com/company/", re.I)


def is_linkedin_profile_url(url: str) -> bool:
    return bool(url and PROFILE_RE.search(url) and not COMPANY_RE.search(url))


def is_linkedin_company_url(url: str) -> bool:
    return bool(url and COMPANY_RE.search(url))


def extract_person_from_linkedin_result(title: str, url: str, snippet: str) -> Dict[str, str]:
    result = {
        "name": "",
        "job_title": "",
        "employer": "",
        "location": "",
        "linkedin_url": normalize_url(url) if is_linkedin_profile_url(url) else "",
        "is_profile": str(is_linkedin_profile_url(url)).lower(),
    }
    clean_title = re.sub(r"\s*[|\-–]\s*LinkedIn.*$", "", title or "", flags=re.I).strip()
    if " - " in clean_title:
        first, rest = clean_title.split(" - ", 1)
        if 1 < len(first.split()) <= 5 and not any(ch.isdigit() for ch in first):
            result["name"] = first.strip()
        if re.search(r"\bat\b", rest, re.I):
            jt, emp = re.split(r"\s+at\s+", rest, maxsplit=1, flags=re.I)
            result["job_title"] = jt.strip()
            result["employer"] = emp.strip()
        else:
            result["job_title"] = rest.strip()
    loc_match = re.search(r"([A-Z][A-Za-z\s]+,\s*[A-Z][A-Za-z\s]+)\s*(?:·|\||$)", snippet or "")
    if loc_match:
        result["location"] = clean_text(loc_match.group(1))
    return result


def extract_company_from_linkedin_result(title: str, url: str, snippet: str) -> Dict[str, str]:
    title_clean = re.sub(r"\s*[|\-–]\s*LinkedIn.*$", "", title or "", flags=re.I).strip()
    return {
        "company_name": clean_text(title_clean),
        "linkedin_url": normalize_url(url) if is_linkedin_company_url(url) else "",
        "description": clean_text(snippet),
        "domain": extract_domain(url),
    }


def search_result_to_person_record(result: SearchResult) -> CompanyRecord:
    data = extract_person_from_linkedin_result(result.title, result.url, result.snippet)
    return CompanyRecord(
        company_name=data.get("name", ""),
        job_title=data.get("job_title", ""),
        employer_name=data.get("employer", ""),
        city=data.get("location", ""),
        linkedin_profile=data.get("linkedin_url", ""),
        linkedin_url=data.get("linkedin_url", ""),
        website=data.get("linkedin_url", ""),
        domain=extract_domain(data.get("linkedin_url", "")),
        description=clean_text(result.snippet),
        source_url=result.url,
        source_provider=result.provider,
        page_type="person",
        raw_sources=[result.to_dict()],
    )
