from __future__ import annotations


def _get(record, name: str):
    return record.get(name, "") if isinstance(record, dict) else getattr(record, name, "")


def build_outreach_brief(record, target_market: str = "", reason: str = "") -> str:
    company = str(_get(record, "company_name") or "this company").strip()
    website = str(_get(record, "website") or "").strip()
    description = str(_get(record, "description") or "").strip()
    hq = str(_get(record, "hq_country") or _get(record, "country") or "").strip()
    contacts = [
        str(_get(record, "email") or "").strip(),
        str(_get(record, "phone") or "").strip(),
        str(_get(record, "linkedin_url") or "").strip(),
    ]
    contacts = [c for c in contacts if c]

    lines = [f"Target: {company}"]
    if website:
        lines.append(f"Website: {website}")
    if hq:
        lines.append(f"HQ: {hq}")
    if target_market:
        lines.append(f"Target market: {target_market}")
    if description:
        lines.append(f"Why it may fit: {description[:400]}")
    if reason:
        lines.append(f"Outreach angle: {reason}")
    if contacts:
        lines.append(f"Available contact signals: {', '.join(contacts)}")
    return "\n".join(lines)
