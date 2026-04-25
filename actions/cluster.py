from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence


def _get(record, name: str):
    return record.get(name, "") if isinstance(record, dict) else getattr(record, name, "")


def cluster_records_by_field(records: Sequence, field: str = "hq_country") -> Dict[str, List]:
    groups: Dict[str, List] = defaultdict(list)
    for rec in records or []:
        value = _get(rec, field)
        if isinstance(value, list):
            if not value:
                groups["Unknown"].append(rec)
            else:
                for item in value:
                    key = str(item).strip() or "Unknown"
                    groups[key].append(rec)
        else:
            key = str(value).strip() or "Unknown"
            groups[key].append(rec)
    return dict(groups)


def cluster_records_by_keyword(records: Sequence, keywords: Sequence[str]) -> Dict[str, List]:
    groups: Dict[str, List] = defaultdict(list)
    for rec in records or []:
        text = " ".join([
            str(_get(rec, "company_name") or ""),
            str(_get(rec, "description") or ""),
            str(_get(rec, "notes") or ""),
        ]).lower()
        matched = False
        for kw in keywords or []:
            key = str(kw or "").strip()
            if key and key.lower() in text:
                groups[key].append(rec)
                matched = True
        if not matched:
            groups["Unclustered"].append(rec)
    return dict(groups)
