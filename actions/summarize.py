from __future__ import annotations

import re
from typing import Iterable, List, Sequence


def _record_get(record, *names: str) -> str:
    for name in names:
        if isinstance(record, dict):
            value = record.get(name, "")
        else:
            value = getattr(record, name, "")
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _heuristic_summary(text: str, max_sentences: int = 2) -> str:
    sentences = _sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])


def _company_prompt_text(record) -> str:
    return " ".join([
        _record_get(record, "company_name"),
        _record_get(record, "description", "summary"),
        _record_get(record, "hq_country", "country"),
        ", ".join(getattr(record, "presence_countries", []) or []) if not isinstance(record, dict) else ", ".join(record.get("presence_countries", []) or []),
        _record_get(record, "website"),
    ]).strip()


def _paper_prompt_text(record) -> str:
    return " ".join([
        _record_get(record, "company_name", "title"),
        _record_get(record, "authors"),
        _record_get(record, "publication_year", "year"),
        _record_get(record, "description", "abstract", "summary"),
    ]).strip()


def _person_prompt_text(record) -> str:
    return " ".join([
        _record_get(record, "company_name"),
        _record_get(record, "job_title"),
        _record_get(record, "employer_name"),
        _record_get(record, "city", "country"),
        _record_get(record, "description", "summary"),
    ]).strip()


def summarize_record(record, entity_type: str = "company", llm=None, max_sentences: int = 2) -> str:
    entity_type = (entity_type or "company").lower().strip()
    if entity_type == "paper":
        text = _paper_prompt_text(record)
    elif entity_type == "person":
        text = _person_prompt_text(record)
    else:
        text = _company_prompt_text(record)

    if llm and getattr(llm, "is_available", lambda: False)():
        prompt = (
            f"Summarize this {entity_type} in {max_sentences} short sentences. "
            f"Focus only on explicit information.\n\n{text}"
        )
        try:
            summary = llm.generate_text(prompt, timeout=25)
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        except Exception:
            pass

    return _heuristic_summary(text, max_sentences=max_sentences)


def summarize_records(records: Sequence, entity_type: str = "company", llm=None, max_sentences: int = 2) -> List[str]:
    return [summarize_record(r, entity_type=entity_type, llm=llm, max_sentences=max_sentences) for r in (records or [])]
