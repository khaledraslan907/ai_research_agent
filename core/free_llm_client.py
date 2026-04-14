"""
free_llm_client.py
==================
Free-first LLM gateway for personal use.

Priority chain (all free tiers, no payment needed for personal use):
  1. Groq  — llama-3.1-8b-instant   — 30 req/min, 14,400 req/DAY free
  2. Gemini — gemini-1.5-flash-8b   — 15 req/min, 1,500 req/DAY free, 1M tokens/day
  3. Ollama — any local model        — unlimited, private, needs local install
  4. OpenRouter — free model tier    — mixed limits, requires free account
  5. Anthropic / OpenAI              — user-supplied paid keys (deployment mode)

Usage:
    client = FreeLLMClient.from_env()          # reads .env
    client = FreeLLMClient(groq_key="gsk_...") # explicit keys
    text = client.generate("Your prompt here")
    data = client.generate_json("Prompt...", schema_hint="Return JSON: {...}")

Why Groq + Gemini as primary free options:
  - Groq: sub-second latency, 14k free req/day — perfect for batch classification
  - Gemini Flash: excellent reasoning, huge free quota, Google reliability
  - Both support JSON mode / structured output
  - Together they cover essentially unlimited personal use
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests

from core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Model constants — all free tier options
# ---------------------------------------------------------------------------

GROQ_MODEL       = "llama-3.1-8b-instant"      # fastest, free forever
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"   # smarter, still free
GEMINI_MODEL     = "gemini-1.5-flash-8b"        # free, 1M tokens/day
GEMINI_MODEL_PRO = "gemini-1.5-flash"           # better, still free
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
ANTHROPIC_MODEL  = "claude-haiku-4-5-20251022"  # cheapest paid (~$0.001/call)
OPENAI_MODEL     = "gpt-4o-mini"                # cheapest OpenAI paid


class FreeLLMClient:
    """
    Single interface to all LLM backends.
    Automatically uses free options first; falls back to paid keys
    only if user explicitly provides them.
    """

    def __init__(
        self,
        # Free keys (get from their websites, no credit card needed)
        groq_api_key:       Optional[str] = None,
        gemini_api_key:     Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        # Local
        ollama_base_url:    Optional[str] = None,
        ollama_model:       Optional[str] = None,
        # Paid fallbacks (user-supplied in deployment)
        anthropic_api_key:  Optional[str] = None,
        openai_api_key:     Optional[str] = None,
        # Behaviour
        prefer_smart: bool = False,    # use larger model when available
        cache_ttl:    int  = 3600,     # in-memory cache seconds (0 = off)
    ):
        self.groq_api_key       = (groq_api_key       or "").strip()
        self.gemini_api_key     = (gemini_api_key     or "").strip()
        self.openrouter_api_key = (openrouter_api_key or "").strip()
        self.ollama_base_url    = (ollama_base_url    or OLLAMA_BASE_URL).rstrip("/")
        self.ollama_model       = (ollama_model       or OLLAMA_MODEL).strip()
        self.anthropic_api_key  = (anthropic_api_key  or "").strip()
        self.openai_api_key     = (openai_api_key     or "").strip()
        self.prefer_smart       = prefer_smart
        self.cache_ttl          = cache_ttl

        # Simple in-process cache {hash: (response_text, timestamp)}
        self._cache: Dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides) -> "FreeLLMClient":
        """Construct from environment variables + optional overrides."""
        return cls(
            groq_api_key       = overrides.get("groq_api_key",       os.getenv("GROQ_API_KEY", "")),
            gemini_api_key     = overrides.get("gemini_api_key",     os.getenv("GEMINI_API_KEY", "")),
            openrouter_api_key = overrides.get("openrouter_api_key", os.getenv("OPENROUTER_API_KEY", "")),
            ollama_base_url    = overrides.get("ollama_base_url",    os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)),
            ollama_model       = overrides.get("ollama_model",       os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)),
            anthropic_api_key  = overrides.get("anthropic_api_key",  os.getenv("ANTHROPIC_API_KEY", "")),
            openai_api_key     = overrides.get("openai_api_key",     os.getenv("OPENAI_API_KEY", "")),
        )

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(
            self.groq_api_key
            or self.gemini_api_key
            or self.openrouter_api_key
            or self.anthropic_api_key
            or self.openai_api_key
            or self._ollama_alive()
        )

    def available_backends(self) -> list[str]:
        """Return list of usable backends in priority order."""
        backends = []
        if self.groq_api_key:       backends.append("groq")
        if self.gemini_api_key:     backends.append("gemini")
        if self._ollama_alive():    backends.append("ollama")
        if self.openrouter_api_key: backends.append("openrouter")
        if self.anthropic_api_key:  backends.append("anthropic")
        if self.openai_api_key:     backends.append("openai")
        return backends

    def _ollama_alive(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str, timeout: int = 60, use_cache: bool = True) -> str:
        """
        Generate text using the best available free backend.
        Returns empty string if all backends fail.
        """
        # Cache check
        if use_cache and self.cache_ttl > 0:
            cached = self._cache_get(prompt)
            if cached:
                return cached

        text = ""

        # Priority order: free → local → paid
        if self.groq_api_key:
            text = self._groq(prompt, timeout)
        if not text and self.gemini_api_key:
            text = self._gemini(prompt, timeout)
        if not text and self._ollama_alive():
            text = self._ollama(prompt, timeout)
        if not text and self.openrouter_api_key:
            text = self._openrouter(prompt, timeout)
        if not text and self.anthropic_api_key:
            text = self._anthropic(prompt, timeout)
        if not text and self.openai_api_key:
            text = self._openai(prompt, timeout)

        if text and use_cache and self.cache_ttl > 0:
            self._cache_set(prompt, text)

        return text

    def generate_json(self, prompt: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        Generate and parse JSON. Strips markdown fences. Returns None on failure.
        Retries once with a reminder if the first response is not valid JSON.
        """
        raw = self.generate(prompt, timeout=timeout)
        result = self._parse_json(raw)
        if result is not None:
            return result

        # Retry with explicit reminder
        retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY raw JSON. No markdown. No explanation. Start with { or ["
        raw2 = self.generate(retry_prompt, timeout=timeout, use_cache=False)
        return self._parse_json(raw2)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _groq(self, prompt: str, timeout: int) -> str:
        """Groq — free tier, very fast, llama3 models."""
        model = GROQ_MODEL_SMART if self.prefer_smart else GROQ_MODEL
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.1,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _gemini(self, prompt: str, timeout: int) -> str:
        """Google Gemini — free tier, 1M tokens/day."""
        model = GEMINI_MODEL_PRO if self.prefer_smart else GEMINI_MODEL
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": self.gemini_api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 2048,
                    },
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return " ".join(p.get("text", "") for p in parts).strip()
        except Exception:
            return ""

    def _ollama(self, prompt: str, timeout: int) -> str:
        """Ollama — fully local, private, unlimited."""
        try:
            r = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            return ""

    def _openrouter(self, prompt: str, timeout: int) -> str:
        """OpenRouter — free model tier (rate limited but free)."""
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/research-agent",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _anthropic(self, prompt: str, timeout: int) -> str:
        """Anthropic Claude — paid, user-supplied key."""
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
            content = r.json().get("content", [])
            return "\n".join(i.get("text", "") for i in content if i.get("type") == "text").strip()
        except Exception:
            return ""

    def _openai(self, prompt: str, timeout: int) -> str:
        """OpenAI — paid, user-supplied key."""
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.1,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        # strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            # try to find JSON object/array in response
            for pattern in [r"\{.*\}", r"\[.*\]"]:
                m = re.search(pattern, text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except Exception:
                        pass
        return None

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _cache_get(self, prompt: str) -> Optional[str]:
        key = self._cache_key(prompt)
        if key in self._cache:
            text, ts = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return text
            del self._cache[key]
        return None

    def _cache_set(self, prompt: str, text: str) -> None:
        self._cache[self._cache_key(prompt)] = (text, time.time())
