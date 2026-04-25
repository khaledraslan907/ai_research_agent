from __future__ import annotations

from typing import Dict, List, Sequence


def _get(record, name: str):
    return record.get(name, "") if isinstance(record, dict) else getattr(record, name, "")


def best_contact_bundle(record) -> Dict[str, str]:
    return {
        "company_name": str(_get(record, "company_name") or "").strip(),
        "website": str(_get(record, "website") or _get(record, "source_url") or "").strip(),
        "email": str(_get(record, "email") or "").strip(),
        "phone": str(_get(record, "phone") or "").strip(),
        "linkedin_url": str(_get(record, "linkedin_url") or _get(record, "linkedin_profile") or "").strip(),
        "contact_page": str(_get(record, "contact_page") or "").strip(),
    }


def extract_contacts_from_records(records: Sequence) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for rec in records or []:
        bundle = best_contact_bundle(rec)
        if any(bundle.get(k) for k in ["email", "phone", "linkedin_url", "contact_page", "website"]):
            rows.append(bundle)
    return rows
