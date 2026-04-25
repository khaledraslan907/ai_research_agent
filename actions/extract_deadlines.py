from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional


_DATE_PATTERNS = [
    r"\b(20\d{2}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}/\d{1,2}/20\d{2})\b",
    r"\b(\d{1,2}-\d{1,2}-20\d{2})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*20\d{2})\b",
]


def _parse_date(text: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def extract_deadlines_from_text(text: str) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    blob = str(text or "")
    for pat in _DATE_PATTERNS:
        for match in re.findall(pat, blob, flags=re.IGNORECASE):
            dt = _parse_date(match)
            found.append({
                "raw": match,
                "normalized": dt.strftime("%Y-%m-%d") if dt else "",
            })
    # dedupe preserve order
    seen = set()
    out = []
    for item in found:
        key = item["raw"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def nearest_upcoming_deadline(text: str, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
    now = now or datetime.utcnow()
    candidates = []
    for item in extract_deadlines_from_text(text):
        if item["normalized"]:
            dt = datetime.strptime(item["normalized"], "%Y-%m-%d")
            if dt >= now:
                candidates.append((dt, item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
