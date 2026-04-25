from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from core.critic import CriticIssue, review_results
from core.evidence import EvidenceItem, ValidationDecision, attach_evidence_to_record
from core.task_models import TaskSpec
from core.validators import validate_record


@dataclass
class VerificationOutput:
    accepted: List[Any] = field(default_factory=list)
    rejected: List[Any] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[CriticIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accepted_count': len(self.accepted),
            'rejected_count': len(self.rejected),
            'decisions': list(self.decisions),
            'issues': [vars(x) for x in self.issues],
        }


class VerificationPipeline:
    """Applies entity-aware validation and separates accepted vs rejected records."""

    def __init__(self, min_score: float = 0.0):
        self.min_score = float(min_score or 0.0)

    def run(self, records: Iterable[Any], task_spec: TaskSpec) -> VerificationOutput:
        accepted: List[Any] = []
        rejected: List[Any] = []
        decisions_payload: List[Dict[str, Any]] = []

        for record in (records or []):
            decision: ValidationDecision = validate_record(record, task_spec)
            base_score = float(getattr(record, 'confidence_score', 0.0) or 0.0)
            final_score = max(0.0, base_score + float(decision.score_delta or 0.0))
            try:
                record.confidence_score = final_score
            except Exception:
                pass

            if decision.evidence:
                attach_evidence_to_record(record, decision.evidence)

            payload = {
                'name': getattr(record, 'company_name', '') or getattr(record, 'title', ''),
                'accepted': bool(decision.accepted and final_score >= self.min_score),
                'base_score': base_score,
                'final_score': final_score,
                'reasons': list(decision.reasons),
                'warnings': list(decision.warnings),
                'evidence_count': len(decision.evidence or []),
            }
            decisions_payload.append(payload)

            if payload['accepted']:
                accepted.append(record)
            else:
                note = getattr(record, 'notes', '') or ''
                reasons = '|'.join(payload['reasons']) or 'verification_failed'
                try:
                    record.notes = f"{note} | rejected:{reasons}".strip(' |')
                except Exception:
                    pass
                rejected.append(record)

        issues = review_results(accepted, task_spec)
        return VerificationOutput(accepted=accepted, rejected=rejected, decisions=decisions_payload, issues=issues)
