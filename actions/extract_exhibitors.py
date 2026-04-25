from __future__ import annotations

import re
from typing import List, Sequence


_BAD_LINES = {
    "exhibitors", "sponsors", "partners", "conference", "event", "download", "register", "login"
}


def normalize_exhibitor_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names or []:
        s = re.sub(r"\s+", " ", str(name or "")).strip(" -•\t")
        if not s or s.lower() in _BAD_LINES:
            continue
        if len(s) < 2 or len(s) > 120:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def extract_exhibitors_from_text(text: str) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    for line in re.split(r"[\n\r]+|•|\|", text):
        s = re.sub(r"\s+", " ", line).strip()
        if not s:
            continue
        if re.match(r"^(?:[A-Z][A-Za-z0-9&+\-.,()/' ]{2,80})$", s):
            candidates.append(s)
    return normalize_exhibitor_names(candidates)
