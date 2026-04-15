"""
local_llm_provider.py
=====================
Thin compatibility shim kept so orchestrator code that calls
classify_company_page() and classify_presence() still works.

In v5 the real LLM logic lives in:
  core/free_llm_client.py   — all backends (Groq, Gemini, Ollama, ...)
  core/prompt_templates.py  — all prompts in one place
  core/llm_ranker.py        — batch result re-ranking

This file delegates to FreeLLMClient so we get the free backends
(Groq first, Gemini second) instead of the old hard-coded Anthropic/OpenAI.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.free_llm_client import FreeLLMClient
from core.prompt_templates import PAGE_CLASSIFY_PROMPT, GEO_VERIFY_PROMPT


class LocalLLMProvider:
    """
    Compatibility wrapper around FreeLLMClient.
    Orchestrator calls classify_company_page() and classify_presence()
    — these now run through the free-first LLM chain.
    """

    def __init__(
        self,
        openai_api_key:    Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        groq_api_key:      Optional[str] = None,
        gemini_api_key:    Optional[str] = None,
        **kwargs,
    ):
        self._client = FreeLLMClient(
            groq_api_key      = groq_api_key      or "",
            gemini_api_key    = gemini_api_key    or "",
            anthropic_api_key = anthropic_api_key or "",
            openai_api_key    = openai_api_key    or "",
        )

    def set_keys(
        self,
        groq_api_key:      str = "",
        gemini_api_key:    str = "",
        anthropic_api_key: str = "",
        openai_api_key:    str = "",
    ) -> None:
        self._client = FreeLLMClient(
            groq_api_key      = groq_api_key,
            gemini_api_key    = gemini_api_key,
            anthropic_api_key = anthropic_api_key,
            openai_api_key    = openai_api_key,
        )

    def is_available(self) -> bool:
        return self._client.is_available()

    # ------------------------------------------------------------------
    # Page classification
    # ------------------------------------------------------------------

    def classify_company_page(
        self,
        title:  str,
        url:    str,
        text:   str,
        sector: str,
    ) -> Dict[str, Any]:
        prompt = PAGE_CLASSIFY_PROMPT.format(
            sector=sector or "industry",
            title=title or "",
            url=url or "",
            text=(text or "")[:3000],
        )
        result = self._client.generate_json(prompt, timeout=30)
        if result and isinstance(result, dict):
            return result
        return {
            "page_type":   "unknown",
            "is_relevant": True,   # default to keeping — don't lose results on failure
            "confidence":  0,
            "reason":      "LLM unavailable or parse failed",
        }

    # ------------------------------------------------------------------
    # Geo / presence classification
    # ------------------------------------------------------------------

    def classify_presence(
        self,
        company_name: str,
        text:         str,
    ) -> Dict[str, Any]:
        prompt = GEO_VERIFY_PROMPT.format(
            company_name=company_name or "Unknown",
            website="",
            description="",
            page_text=(text or "")[:800],
        )
        result = self._client.generate_json(prompt, timeout=30)
        if result and isinstance(result, dict):
            # normalise key names — GEO_VERIFY_PROMPT uses slightly different keys
            return {
                "hq_country":         result.get("hq_country", ""),
                "presence_countries": result.get("presence_countries", []),
                "has_usa_presence":   bool(result.get("has_usa_presence", False)),
                "has_egypt_presence": bool(result.get("has_egypt_presence", False)),
            }
        return {
            "hq_country":         "",
            "presence_countries": [],
            "has_usa_presence":   False,
            "has_egypt_presence": False,
        }

    # ------------------------------------------------------------------
    # Batch verification (legacy — now superseded by llm_ranker.py)
    # ------------------------------------------------------------------

    def batch_verify_records(
        self,
        records:          List[Dict[str, Any]],
        task_description: str,
        exclude_countries: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Legacy method kept for compatibility.
        The main pipeline now uses core/llm_ranker.py instead.
        """
        from core.llm_ranker import rerank_records
        from core.models import CompanyRecord
        from core.task_models import TaskSpec, GeographyRules

        # Convert dicts to CompanyRecord objects for rerank_records
        recs = []
        for d in records:
            r = CompanyRecord(
                company_name=d.get("company_name", ""),
                website=d.get("website", ""),
                domain=d.get("domain", ""),
                description=d.get("description", ""),
                confidence_score=float(d.get("confidence_score", 50)),
            )
            recs.append(r)

        task = TaskSpec(
            raw_prompt=task_description,
            geography=GeographyRules(exclude_countries=exclude_countries),
        )

        rerank_records(recs, task, self._client, batch_size=40)

        return [
            {
                "index": i,
                "keep":  r.confidence_score >= 40,
                "reason": r.notes,
            }
            for i, r in enumerate(recs)
        ]
