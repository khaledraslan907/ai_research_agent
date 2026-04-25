from __future__ import annotations

import re
from typing import Dict, List

from core.models import CompanyRecord, SearchResult
from core.utils import clean_text, extract_domain


_EXHIBITOR_LINE_RE = re.compile(r"(?m)^\s*(?:•|-|\*|\d+[.)])\s+(.{2,120})$")
_SPEAKER_RE = re.compile(r"(?i)\b(speaker|speakers|panelist|presenter|exhibitor|exhibitors)\b")


def extract_exhibitor_names(text: str) -> List[str]:
    names = [clean_text(m.group(1)) for m in _EXHIBITOR_LINE_RE.finditer(text or "")]
    return [n for n in names if 1 < len(n.split()) <= 8]


def is_event_like_result(result: SearchResult) -> bool:
    text = f"{result.title} {result.snippet}".lower()
    hints = ["conference", "summit", "expo", "event", "speaker", "exhibitor", "workshop", "forum"]
    return any(h in text for h in hints)


def search_result_to_event_record(result: SearchResult) -> CompanyRecord:
    return CompanyRecord(
        company_name=clean_text(result.title),
        website=result.url,
        domain=extract_domain(result.url),
        description=clean_text(result.snippet),
        source_url=result.url,
        source_provider=result.provider,
        page_type="document",
        notes="event_like=true" if is_event_like_result(result) else "",
        raw_sources=[result.to_dict()],
    )
