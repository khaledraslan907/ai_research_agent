from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

from core.config import (
    EXA_API_KEY, TAVILY_API_KEY, SERPAPI_KEY,
    FIRECRAWL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY,
)


@dataclass
class ResolvedKeys:
    """All provider keys resolved from .env + user-supplied overrides."""
    # Search providers
    exa_api_key:        str = ""
    tavily_api_key:     str = ""
    serpapi_key:        str = ""
    firecrawl_api_key:  str = ""
    # Free LLMs (no credit card needed)
    groq_api_key:       str = ""
    gemini_api_key:     str = ""
    openrouter_api_key: str = ""
    # Paid LLMs (user-supplied keys only)
    openai_api_key:     str = ""
    anthropic_api_key:  str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Mask keys for safe display — never expose raw keys in UI."""
        def _mask(v: str) -> str:
            return v[:6] + "***" if len(v) > 8 else ("***" if v else "")
        return {k: _mask(v) for k, v in asdict(self).items()}

    def has_any_free_llm(self) -> bool:
        return bool(self.groq_api_key or self.gemini_api_key or self.openrouter_api_key)

    def has_any_llm(self) -> bool:
        return self.has_any_free_llm() or bool(self.anthropic_api_key or self.openai_api_key)

    def has_any_paid_search(self) -> bool:
        return bool(self.exa_api_key or self.tavily_api_key or self.serpapi_key)


def resolve_provider_keys(mode: str, user_keys: dict) -> ResolvedKeys:
    """
    Merge .env keys with user-supplied keys.
    User keys always take priority over .env values.
    """
    def _pick(user_val: str, env_val: str) -> str:
        u = (user_val or "").strip()
        e = (env_val  or "").strip()
        return u if u else e

    return ResolvedKeys(
        exa_api_key        = _pick(user_keys.get("exa_api_key",        ""), EXA_API_KEY),
        tavily_api_key     = _pick(user_keys.get("tavily_api_key",     ""), TAVILY_API_KEY),
        serpapi_key        = _pick(user_keys.get("serpapi_key",        ""), SERPAPI_KEY),
        firecrawl_api_key  = _pick(user_keys.get("firecrawl_api_key",  ""), FIRECRAWL_API_KEY),
        groq_api_key       = _pick(user_keys.get("groq_api_key",       ""), GROQ_API_KEY),
        gemini_api_key     = _pick(user_keys.get("gemini_api_key",     ""), GEMINI_API_KEY),
        openrouter_api_key = _pick(user_keys.get("openrouter_api_key", ""), OPENROUTER_API_KEY),
        openai_api_key     = _pick(user_keys.get("openai_api_key",     ""), OPENAI_API_KEY),
        anthropic_api_key  = _pick(user_keys.get("anthropic_api_key",  ""), ANTHROPIC_API_KEY),
    )
