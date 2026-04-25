from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def _get(record, name: str):
    return record.get(name, "") if isinstance(record, dict) else getattr(record, name, "")


def build_comparison_table(records: Sequence, fields: Sequence[str] | None = None) -> pd.DataFrame:
    fields = list(fields or [
        "company_name", "website", "hq_country", "presence_countries", "email", "phone",
        "authors", "publication_year", "job_title", "employer_name", "confidence_score",
    ])
    rows: List[dict] = []
    for rec in records or []:
        row = {}
        for field in fields:
            value = _get(rec, field)
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value if str(x).strip())
            row[field] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    keep = [c for c in df.columns if c in df and df[c].astype(str).str.strip().replace({"nan": ""}).ne("").any()]
    return df[keep].copy() if not df.empty else pd.DataFrame(columns=list(fields))


def compare_records(records: Sequence, fields: Sequence[str] | None = None, as_markdown: bool = False):
    df = build_comparison_table(records, fields=fields)
    if as_markdown:
        return df.to_markdown(index=False) if not df.empty else ""
    return df
