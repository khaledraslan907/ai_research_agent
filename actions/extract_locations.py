from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def _get(record, name: str):
    return record.get(name, "") if isinstance(record, dict) else getattr(record, name, "")


def extract_locations_from_records(records: Sequence) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for rec in records or []:
        presence = _get(rec, "presence_countries")
        if isinstance(presence, list):
            presence = ", ".join(str(x) for x in presence if str(x).strip())
        rows.append({
            "company_name": str(_get(rec, "company_name") or "").strip(),
            "country": str(_get(rec, "country") or "").strip(),
            "city": str(_get(rec, "city") or "").strip(),
            "hq_country": str(_get(rec, "hq_country") or "").strip(),
            "presence_countries": str(presence or "").strip(),
        })
    return rows


def summarize_geo_footprint(records: Sequence) -> Dict[str, int]:
    counts = Counter()
    for rec in records or []:
        hq = str(_get(rec, "hq_country") or _get(rec, "country") or "").strip()
        if hq:
            counts[hq] += 1
    return dict(counts)
