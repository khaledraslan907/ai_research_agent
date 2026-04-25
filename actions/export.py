from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _rec_to_dict(rec):
    if isinstance(rec, dict):
        return dict(rec)
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    return rec.__dict__.copy()


def records_to_dataframe(records: Sequence) -> pd.DataFrame:
    rows = [_rec_to_dict(r) for r in (records or [])]
    return pd.DataFrame(rows)


def export_records_action(records: Sequence, output_path: str, fmt: str = "xlsx") -> str:
    df = records_to_dataframe(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or path.suffix.lstrip(".") or "xlsx").lower()

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        # default xlsx
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        df.to_excel(path, index=False)
    return str(path)
