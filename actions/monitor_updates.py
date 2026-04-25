from __future__ import annotations

from typing import Dict, List, Sequence


def _key(rec) -> str:
    if isinstance(rec, dict):
        name = str(rec.get("company_name") or rec.get("title") or "").strip().lower()
        website = str(rec.get("website") or rec.get("source_url") or rec.get("linkedin_url") or "").strip().lower()
    else:
        name = str(getattr(rec, "company_name", "") or "").strip().lower()
        website = str(getattr(rec, "website", "") or getattr(rec, "source_url", "") or getattr(rec, "linkedin_url", "") or "").strip().lower()
    return f"{name}|{website}"


def compare_snapshots(old_records: Sequence, new_records: Sequence) -> Dict[str, List]:
    old_map = {_key(r): r for r in old_records or [] if _key(r) != "|"}
    new_map = {_key(r): r for r in new_records or [] if _key(r) != "|"}

    added = [new_map[k] for k in new_map.keys() - old_map.keys()]
    removed = [old_map[k] for k in old_map.keys() - new_map.keys()]
    kept = [new_map[k] for k in new_map.keys() & old_map.keys()]
    return {"added": added, "removed": removed, "kept": kept}


def summarize_snapshot_changes(old_records: Sequence, new_records: Sequence) -> str:
    diff = compare_snapshots(old_records, new_records)
    return (
        f"Added: {len(diff['added'])} | Removed: {len(diff['removed'])} | "
        f"Unchanged/retained: {len(diff['kept'])}"
    )
