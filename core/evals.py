from __future__ import annotations

"""
evals.py
========
Lightweight evaluation helpers for the research agent.

This is intentionally simple so you can start benchmarking prompts and outputs
before building a full evaluation pipeline.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class EvalCase:
    name: str = ""
    prompt: str = ""
    entity_type: str = "company"
    must_include: List[str] = field(default_factory=list)
    must_exclude: List[str] = field(default_factory=list)
    min_results: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    case_name: str = ""
    prompt: str = ""
    total_results: int = 0
    include_hits: List[str] = field(default_factory=list)
    missed_includes: List[str] = field(default_factory=list)
    exclude_leaks: List[str] = field(default_factory=list)
    precision_proxy: float = 0.0
    recall_proxy: float = 0.0
    pass_case: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSuiteSummary:
    total_cases: int = 0
    passed_cases: int = 0
    avg_precision_proxy: float = 0.0
    avg_recall_proxy: float = 0.0
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "avg_precision_proxy": self.avg_precision_proxy,
            "avg_recall_proxy": self.avg_recall_proxy,
            "results": [r.to_dict() for r in self.results],
        }


def load_eval_cases(path: str | Path) -> List[EvalCase]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    out: List[EvalCase] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        out.append(EvalCase(**item))
    return out


def save_eval_cases(path: str | Path, cases: Iterable[EvalCase]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": [c.to_dict() for c in cases]}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_case(case: EvalCase, records: Iterable[dict | Any]) -> EvalResult:
    rows = list(records or [])
    names: List[str] = []
    texts: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            name = str(row.get("company_name") or row.get("title") or row.get("name") or "")
            blob = " ".join(str(v) for v in row.values() if v is not None)
        else:
            name = str(getattr(row, "company_name", "") or getattr(row, "title", "") or getattr(row, "name", ""))
            blob = " ".join([
                str(getattr(row, "company_name", "") or ""),
                str(getattr(row, "website", "") or ""),
                str(getattr(row, "description", "") or ""),
                str(getattr(row, "notes", "") or ""),
            ])
        names.append(name.lower())
        texts.append(blob.lower())

    include_hits: List[str] = []
    missed_includes: List[str] = []
    for must in case.must_include:
        low = str(must or "").lower().strip()
        if low and any(low in text for text in texts):
            include_hits.append(must)
        else:
            missed_includes.append(must)

    exclude_leaks: List[str] = []
    for bad in case.must_exclude:
        low = str(bad or "").lower().strip()
        if low and any(low in text for text in texts):
            exclude_leaks.append(bad)

    recall_proxy = (len(include_hits) / len(case.must_include)) if case.must_include else 1.0
    precision_proxy = 1.0
    if rows:
        precision_proxy = max(0.0, 1.0 - (len(exclude_leaks) / max(1, len(rows))))

    pass_case = (
        len(rows) >= int(case.min_results or 0)
        and not exclude_leaks
        and not missed_includes
    )

    notes = []
    if len(rows) < int(case.min_results or 0):
        notes.append(f"Only {len(rows)} results, expected at least {case.min_results}.")
    if missed_includes:
        notes.append(f"Missing expected items: {', '.join(missed_includes[:6])}")
    if exclude_leaks:
        notes.append(f"Excluded items leaked: {', '.join(exclude_leaks[:6])}")

    return EvalResult(
        case_name=case.name,
        prompt=case.prompt,
        total_results=len(rows),
        include_hits=include_hits,
        missed_includes=missed_includes,
        exclude_leaks=exclude_leaks,
        precision_proxy=round(precision_proxy, 4),
        recall_proxy=round(recall_proxy, 4),
        pass_case=pass_case,
        notes=" | ".join(notes) or case.notes,
    )


def evaluate_suite(cases: Iterable[EvalCase], case_to_records: Dict[str, Iterable[dict | Any]]) -> EvalSuiteSummary:
    results: List[EvalResult] = []
    for case in cases:
        records = case_to_records.get(case.name, [])
        results.append(evaluate_case(case, records))

    total = len(results)
    passed = sum(1 for r in results if r.pass_case)
    avg_precision = round(sum(r.precision_proxy for r in results) / total, 4) if total else 0.0
    avg_recall = round(sum(r.recall_proxy for r in results) / total, 4) if total else 0.0

    return EvalSuiteSummary(
        total_cases=total,
        passed_cases=passed,
        avg_precision_proxy=avg_precision,
        avg_recall_proxy=avg_recall,
        results=results,
    )
