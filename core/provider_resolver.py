from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from core.config import (
    EXA_API_KEY, TAVILY_API_KEY, SERPAPI_KEY, FIRECRAWL_API_KEY,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,
    OPENROUTER_API_KEY,
)


@dataclass
class ResolvedKeys:
    exa_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_key: str = ""
    firecrawl_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        def _mask(v: str) -> str:
            v = (v or "").strip()
            if not v:
                return ""
            if len(v) <= 8:
                return "***"
            return v[:4] + "***" + v[-2:]
        return {k: _mask(v) for k, v in asdict(self).items()}

    def has_any_free_llm(self) -> bool:
        return bool(self.groq_api_key or self.gemini_api_key or self.openrouter_api_key)

    def has_any_paid_llm(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key)

    def has_any_llm(self) -> bool:
        return self.has_any_free_llm() or self.has_any_paid_llm()

    def has_any_search_provider(self) -> bool:
        return bool(self.exa_api_key or self.tavily_api_key or self.serpapi_key)


def resolve_provider_keys(mode: str, user_keys: dict | None) -> ResolvedKeys:
    user_keys = user_keys or {}

    def _pick(user_val: str, env_val: str) -> str:
        u = (user_val or "").strip()
        e = (env_val or "").strip()
        return u or e

    # mode kept for future policy branching; currently user values always override .env.
    _ = mode
    return ResolvedKeys(
        exa_api_key=_pick(user_keys.get("exa_api_key", ""), EXA_API_KEY),
        tavily_api_key=_pick(user_keys.get("tavily_api_key", ""), TAVILY_API_KEY),
        serpapi_key=_pick(user_keys.get("serpapi_key", ""), SERPAPI_KEY),
        firecrawl_api_key=_pick(user_keys.get("firecrawl_api_key", ""), FIRECRAWL_API_KEY),
        groq_api_key=_pick(user_keys.get("groq_api_key", ""), GROQ_API_KEY),
        gemini_api_key=_pick(user_keys.get("gemini_api_key", ""), GEMINI_API_KEY),
        openrouter_api_key=_pick(user_keys.get("openrouter_api_key", ""), OPENROUTER_API_KEY),
        openai_api_key=_pick(user_keys.get("openai_api_key", ""), OPENAI_API_KEY),
        anthropic_api_key=_pick(user_keys.get("anthropic_api_key", ""), ANTHROPIC_API_KEY),
    )
