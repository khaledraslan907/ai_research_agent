"""
llm_ranker.py
=============
LLM-based result re-ranking and verification.

Why this is the most important new file:
  - Keyword scoring (scoring.py) catches obvious garbage but misses subtle cases
  - The LLM actually reads the description and makes an intelligent judgment
  - Runs as a BATCH call (all candidates in one prompt) — very efficient
  - With Groq free tier: 14,400 free calls/day — costs you nothing
  - Runs AFTER keyword scoring pre-filters obvious junk, so only top-30
    candidates reach the LLM, keeping each call under token limits

Example of what this catches that keyword scoring misses:
  - "Top 10 Oil Gas Software Companies in 2024" (directory, not a company)
  - "Schlumberger announces partnership with X" (news article)
  - Football websites that happen to mention "energy" once
  - Companies that serve the oil industry but are based in the USA (to exclude)

Cost analysis for personal use:
  - Groq llama-3.1-8b-instant: FREE, 14,400 req/day
  - Each search session: 1-2 ranking calls (batch of 20-30 candidates)
  - Daily capacity: 7,000+ search sessions — more than enough for personal use
"""

from __future__ import annotations

from typing import List, Optional

from core.free_llm_client import FreeLLMClient
from core.models import CompanyRecord
from core.prompt_templates import RERANK_PROMPT
from core.task_models import TaskSpec


def rerank_records(
    records: List[CompanyRecord],
    task_spec: TaskSpec,
    llm: Optional[FreeLLMClient],
    batch_size: int = 30,
) -> List[CompanyRecord]:
    """
    Re-rank and filter records using LLM judgment.

    Records with LLM score < 4 get their confidence_score reduced by 25 points.
    Records with LLM score >= 7 get a +10 bonus.
    This softly adjusts scores rather than hard-rejecting, preserving audit trail.

    Args:
        records:    Candidate records (should already be keyword-pre-filtered)
        task_spec:  Original task for context
        llm:        LLM client (skips if unavailable)
        batch_size: Max candidates per LLM call (keep under token limit)

    Returns:
        Same records list with adjusted confidence_scores and notes.
    """
    if not llm or not llm.is_available() or not records:
        return records

    exclude = ", ".join(task_spec.geography.exclude_countries) if task_spec.geography.exclude_countries else "none"
    entity  = task_spec.target_entity_types[0] if task_spec.target_entity_types else "company"

    # Process in batches
    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        _apply_llm_ranking(batch, task_spec, entity, exclude, llm)

    return records


def _apply_llm_ranking(
    batch: List[CompanyRecord],
    task_spec: TaskSpec,
    entity_type: str,
    exclude_countries: str,
    llm: FreeLLMClient,
) -> None:
    """Apply LLM ranking to a batch of records. Modifies records in place."""

    # Build candidate summary for LLM (concise to stay within token limits)
    lines = []
    for i, r in enumerate(batch):
        name  = (r.company_name or "Unknown")[:50]
        domain = r.domain or ""
        desc  = (r.description or "")[:150].replace("\n", " ")
        lines.append(f"{i} | {name} | {domain} | {desc}")

    candidates_text = "\n".join(lines)

    prompt = RERANK_PROMPT.format(
        user_request=task_spec.raw_prompt[:200],
        topic=task_spec.industry or "unknown",
        entity_type=entity_type,
        exclude_countries=exclude_countries,
        candidates=candidates_text,
    )

    verdicts = llm.generate_json(prompt, timeout=45)

    if not verdicts or not isinstance(verdicts, list):
        return

    # Apply verdicts to records
    verdict_map = {}
    for v in verdicts:
        if isinstance(v, dict) and "index" in v:
            idx = int(v["index"])
            verdict_map[idx] = v

    for i, rec in enumerate(batch):
        verdict = verdict_map.get(i)
        if not verdict:
            continue

        score  = int(verdict.get("score", 5))
        keep   = bool(verdict.get("keep", True))
        reason = str(verdict.get("reason", ""))[:120]

        # Adjust confidence score based on LLM judgment
        if score >= 8:
            rec.confidence_score = min(100.0, rec.confidence_score + 15)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10").strip(" |")
        elif score >= 5:
            # Neutral — no change
            rec.notes = (rec.notes + f" | llm_rank:{score}/10").strip(" |")
        elif score >= 3:
            rec.confidence_score = max(0.0, rec.confidence_score - 20)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10 | {reason}").strip(" |")
        else:
            # score 0-2: clearly garbage — drop hard so it falls below any threshold
            rec.confidence_score = max(0.0, rec.confidence_score - 45)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10 | {reason}").strip(" |")

        if not keep:
            rec.confidence_score = max(0.0, rec.confidence_score - 15)


def quick_relevance_check(
    record: CompanyRecord,
    topic: str,
    llm: Optional[FreeLLMClient],
) -> bool:
    """
    Quick single-record relevance check.
    Use sparingly — batch ranking (rerank_records) is much more efficient.
    Returns True if record seems relevant, True by default if LLM unavailable.
    """
    if not llm or not llm.is_available():
        return True

    prompt = f"""Is this page relevant to the topic "{topic}"?

Name: {record.company_name}
URL: {record.website}
Description: {(record.description or '')[:300]}

Answer with just: {{"relevant": true/false, "reason": "one sentence"}}"""

    result = llm.generate_json(prompt, timeout=15)
    if not result or not isinstance(result, dict):
        return True  # default to keeping if unsure

    return bool(result.get("relevant", True))
