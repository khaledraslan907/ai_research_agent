from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord, SearchResult
from core.utils import clean_text, extract_domain


_DEADLINE_RE = re.compile(
    r"(?i)\b(?:deadline|closing date|closing|due date|submission date)\b[:\s-]*([A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"
)
_BUYER_RE = re.compile(r"(?i)\b(?:buyer|issuer|procuring entity|authority|owner)\b[:\s-]*([^\n|]{3,120})")


def extract_tender_fields(text: str) -> Dict[str, str]:
    text = clean_text(text)
    dl = _DEADLINE_RE.search(text)
    buyer = _BUYER_RE.search(text)
    return {
        "deadline": clean_text(dl.group(1)) if dl else "",
        "buyer": clean_text(buyer.group(1)) if buyer else "",
    }


def search_result_to_tender_record(result: SearchResult) -> CompanyRecord:
    fields = extract_tender_fields(f"{result.title} {result.snippet}")
    notes_parts = []
    if fields["deadline"]:
        notes_parts.append(f"deadline={fields['deadline']}")
    if fields["buyer"]:
        notes_parts.append(f"buyer={fields['buyer']}")
    return CompanyRecord(
        company_name=clean_text(result.title),
        website=result.url,
        domain=extract_domain(result.url),
        description=clean_text(result.snippet),
        source_url=result.url,
        source_provider=result.provider,
        page_type="document",
        notes=" | ".join(notes_parts),
        raw_sources=[result.to_dict()],
    )
