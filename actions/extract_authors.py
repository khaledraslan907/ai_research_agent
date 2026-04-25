from __future__ import annotations

import re
from typing import List, Sequence


def normalize_author_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, (list, tuple)):
        raw = ", ".join(str(x) for x in value)
    else:
        raw = str(value or "")
    if not raw:
        return []
    parts = re.split(r"\s*(?:,|;|\band\b)\s*", raw)
    out: List[str] = []
    seen = set()
    for part in parts:
        s = re.sub(r"\s+", " ", part).strip()
        if not s or len(s) < 2:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def extract_authors_from_records(records: Sequence) -> List[dict]:
    rows: List[dict] = []
    for rec in records or []:
        authors = rec.get("authors", "") if isinstance(rec, dict) else getattr(rec, "authors", "")
        rows.append({
            "title": rec.get("company_name", "") if isinstance(rec, dict) else getattr(rec, "company_name", ""),
            "authors": normalize_author_list(authors),
        })
    return rows
