from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord, SearchResult
from core.utils import clean_text, extract_domain


PATENT_RE = re.compile(
    r"\b(?:US|WO|EP|CN|JP|KR|AU|CA)?\s?\d{4,11}[A-Z]?\d?\b",
    re.I,
)


def extract_patent_numbers(text: str) -> List[str]:
    found = [re.sub(r"\s+", "", m.group(0).upper()) for m in PATENT_RE.finditer(text or "")]
    seen = set()
    out: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_patent_record(result: SearchResult) -> CompanyRecord:
    text = clean_text(f"{result.title} {result.snippet}")
    patents = extract_patent_numbers(text)
    notes = f"patent_numbers={', '.join(patents)}" if patents else ""
    return CompanyRecord(
        company_name=clean_text(result.title),
        website=result.url,
        domain=extract_domain(result.url),
        description=clean_text(result.snippet),
        source_url=result.url,
        source_provider=result.provider,
        page_type="document",
        notes=notes,
        matched_keywords=patents,
        raw_sources=[result.to_dict()],
    )
