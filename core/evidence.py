from __future__ import annotations

"""Evidence sidecar objects and helper functions."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Sequence

from core.geography import contains_country_or_city, find_countries_in_text, normalize_country_name


@dataclass
class EvidenceItem:
    evidence_type: str = "text"
    source_type: str = "unknown"
    source_url: str = ""
    label: str = ""
    snippet: str = ""
    matched_terms: List[str] = field(default_factory=list)
    matched_countries: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    items: List[EvidenceItem] = field(default_factory=list)

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def extend(self, items: Iterable[EvidenceItem]) -> None:
        self.items.extend(list(items or []))

    def matched_terms(self) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in self.items:
            for term in item.matched_terms:
                s = str(term or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    def matched_countries(self) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in self.items:
            for country in item.matched_countries:
                s = str(country or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    def summary(self) -> str:
        if not self.items:
            return ""
        parts: List[str] = []
        for item in self.items[:5]:
            label = item.label or item.source_type or item.evidence_type
            snippet = (item.snippet or "").strip()
            if snippet:
                parts.append(f"{label}: {snippet[:140]}")
            else:
                parts.append(label)
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}


@dataclass
class ValidationDecision:
    accepted: bool = True
    score_delta: float = 0.0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "score_delta": self.score_delta,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "evidence": [x.to_dict() for x in self.evidence],
        }


def text_evidence(text: str, source_url: str = "", source_type: str = "text", label: str = "") -> EvidenceItem:
    snippet = str(text or "").strip()
    return EvidenceItem(
        evidence_type="text",
        source_type=source_type,
        source_url=source_url,
        label=label or source_type,
        snippet=snippet[:500],
        matched_countries=find_countries_in_text(snippet),
        confidence=0.5 if snippet else 0.0,
    )


def country_evidence(text: str, countries: Sequence[str], source_url: str = "", source_type: str = "text") -> List[EvidenceItem]:
    snippet = str(text or "").strip()
    items: List[EvidenceItem] = []
    for country in countries or []:
        norm = normalize_country_name(country)
        if norm and contains_country_or_city(snippet, norm):
            items.append(EvidenceItem(
                evidence_type="geography",
                source_type=source_type,
                source_url=source_url,
                label=f"geo:{norm}",
                snippet=snippet[:500],
                matched_countries=[norm],
                confidence=0.8,
            ))
    return items


def keyword_evidence(text: str, keywords: Sequence[str], source_url: str = "", source_type: str = "text", label: str = "keywords") -> List[EvidenceItem]:
    low = str(text or "").lower()
    matched = [kw for kw in (keywords or []) if str(kw or "").strip() and str(kw).lower() in low]
    if not matched:
        return []
    return [EvidenceItem(
        evidence_type="keyword",
        source_type=source_type,
        source_url=source_url,
        label=label,
        snippet=str(text or "")[:500],
        matched_terms=list(matched),
        confidence=min(1.0, 0.4 + 0.1 * len(matched)),
    )]


def attach_evidence_to_record(record, evidence_items: Sequence[EvidenceItem]) -> None:
    if record is None or not evidence_items:
        return
    raw_sources = list(getattr(record, "raw_sources", []) or [])
    raw_sources.extend(item.to_dict() for item in evidence_items)
    try:
        record.raw_sources = raw_sources
    except Exception:
        pass
    summary = EvidenceBundle(list(evidence_items)).summary()
    if summary:
        notes = str(getattr(record, "notes", "") or "").strip()
        joined = f"{notes} | evidence:{summary}".strip(" |") if notes else f"evidence:{summary}"
        try:
            record.notes = joined
        except Exception:
            pass
