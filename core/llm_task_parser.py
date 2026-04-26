from __future__ import annotations
from typing import Optional
from core.free_llm_client import FreeLLMClient
from core.task_models import TaskSpec
from core.task_parser import parse_task_prompt

def parse_task_prompt_llm_first(prompt: str, llm: Optional[FreeLLMClient] = None) -> TaskSpec:
    # Stable fallback-first approach: trust regex parser for structure, optionally let LLM enrich later.
    regex_spec = parse_task_prompt(prompt)
    # keep current implementation simple/stable to avoid NameError and topic pollution
    return regex_spec
