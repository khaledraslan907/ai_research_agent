from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord, SearchResult
from core.utils import clean_text, extract_domain


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


def is_likely_academic_result(result: SearchResult) -> bool:
    text = f"{result.title} {result.snippet} {result.url}".lower()
    hints = ["doi", "abstract", "journal", "conference", "proceedings", "pdf", "scholar", "research"]
    return any(h in text for h in hints)


def extract_paper_fields(title: str, snippet: str, url: str = "") -> Dict[str, str]:
    text = clean_text(f"{title} {snippet}")
    year_match = _YEAR_RE.search(text)
    doi_match = _DOI_RE.search(text)
    authors = ""
    author_match = re.match(r"^([^\-–|]{3,120})\s+[\-–|]", snippet or "")
    if author_match:
        authors = clean_text(author_match.group(1))
    return {
        "title": clean_text(title),
        "authors": authors,
        "year": year_match.group(0) if year_match else "",
        "doi": doi_match.group(0) if doi_match else "",
        "url": url,
    }


def search_results_to_paper_records(results: List[SearchResult]) -> List[CompanyRecord]:
    records: List[CompanyRecord] = []
    for result in results:
        fields = extract_paper_fields(result.title, result.snippet, result.url)
        records.append(CompanyRecord(
            company_name=fields["title"],
            website=fields["url"],
            domain=extract_domain(fields["url"]),
            description=clean_text(result.snippet),
            authors=fields["authors"],
            publication_year=fields["year"],
            doi=fields["doi"],
            source_url=result.url,
            source_provider=result.provider,
            page_type="document",
            raw_sources=[result.to_dict()],
        ))
    return records
