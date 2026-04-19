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
    if not llm or not llm.is_available() or not records:
        return records

    entity = task_spec.target_entity_types[0] if task_spec.target_entity_types else "company"
    exclude_hq = ", ".join(task_spec.geography.exclude_countries) if task_spec.geography.exclude_countries else "none"
    exclude_presence = ", ".join(task_spec.geography.exclude_presence_countries) if task_spec.geography.exclude_presence_countries else "none"
    exclude_context = f"HQ exclude: {exclude_hq}; Presence exclude: {exclude_presence}"

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        _apply_llm_ranking(batch, task_spec, entity, exclude_context, llm)

    return records


def _task_context_text(task_spec: TaskSpec, entity_type: str) -> tuple[str, str]:
    category = (getattr(task_spec, "target_category", "general") or "general").strip()
    solution_keywords = list(getattr(task_spec, "solution_keywords", []) or [])
    domain_keywords = list(getattr(task_spec, "domain_keywords", []) or [])
    commercial_intent = (getattr(task_spec, "commercial_intent", "general") or "general").strip()

    user_request = (task_spec.raw_prompt or "")[:300]
    topic = task_spec.industry or "unknown"
    topic_parts = [topic]

    if category and category != "general":
        topic_parts.append(f"category={category}")
    if solution_keywords:
        topic_parts.append(f"solution_keywords={', '.join(solution_keywords[:6])}")
    if domain_keywords:
        topic_parts.append(f"domain_keywords={', '.join(domain_keywords[:6])}")
    if commercial_intent != "general":
        topic_parts.append(f"commercial_intent={commercial_intent}")
    topic_parts.append(f"entity_type={entity_type}")

    return user_request, " | ".join(topic_parts)


def _apply_llm_ranking(
    batch: List[CompanyRecord],
    task_spec: TaskSpec,
    entity_type: str,
    exclude_countries: str,
    llm: FreeLLMClient,
) -> None:
    lines = []
    for i, r in enumerate(batch):
        name = (r.company_name or "Unknown")[:60]
        domain = r.domain or ""
        hq = (r.hq_country or r.country or "")[:40]
        presence = ", ".join((r.presence_countries or [])[:4])
        page_type = r.page_type or ""
        desc = (r.description or "")[:180].replace("\n", " ")
        packed_desc = f"HQ:{hq}; Presence:{presence}; PageType:{page_type}; {desc}"
        lines.append(f"{i} | {name} | {domain} | {packed_desc}")

    candidates_text = "\n".join(lines)
    user_request, topic_context = _task_context_text(task_spec, entity_type)

    prompt = RERANK_PROMPT.format(
        user_request=user_request,
        topic=topic_context,
        entity_type=entity_type,
        exclude_countries=exclude_countries,
        candidates=candidates_text,
    )

    verdicts = llm.generate_json(prompt, timeout=45)
    if not verdicts or not isinstance(verdicts, list):
        return

    verdict_map = {}
    for v in verdicts:
        if isinstance(v, dict) and "index" in v:
            verdict_map[int(v["index"])] = v

    for i, rec in enumerate(batch):
        verdict = verdict_map.get(i)
        if not verdict:
            continue

        score = int(verdict.get("score", 5))
        keep = bool(verdict.get("keep", True))
        reason = str(verdict.get("reason", ""))[:140]

        if score >= 8:
            rec.confidence_score = min(100.0, rec.confidence_score + 15)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10").strip(" |")
        elif score >= 5:
            rec.notes = (rec.notes + f" | llm_rank:{score}/10").strip(" |")
        elif score >= 3:
            rec.confidence_score = max(0.0, rec.confidence_score - 20)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10 | {reason}").strip(" |")
        else:
            rec.confidence_score = max(0.0, rec.confidence_score - 45)
            rec.notes = (rec.notes + f" | llm_rank:{score}/10 | {reason}").strip(" |")

        if not keep:
            rec.confidence_score = max(0.0, rec.confidence_score - 15)

        reason_l = reason.lower()
        if any(x in reason_l for x in ["directory", "news", "media", "blog", "list page", "ranking"]):
            rec.is_directory_or_media = True


def quick_relevance_check(record: CompanyRecord, topic: str, llm: Optional[FreeLLMClient]) -> bool:
    if not llm or not llm.is_available():
        return True

    prompt = f"""Is this page relevant to the topic "{topic}"?

Name: {record.company_name}
URL: {record.website}
Description: {(record.description or '')[:300]}

Answer with just: {{"relevant": true/false, "reason": "one sentence"}}"""

    result = llm.generate_json(prompt, timeout=15)
    if not result or not isinstance(result, dict):
        return True

    return bool(result.get("relevant", True))
