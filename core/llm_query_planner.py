from __future__ import annotations

import re
from typing import Dict, List

from core.models import SearchQuery
from core.task_models import TaskSpec
from core.prompt_templates import QUERY_PLAN_PROMPT

_REGIONS_GLOBAL = [
    "europe", "uk", "norway", "germany", "netherlands", "france",
    "canada", "australia", "singapore", "india", "japan",
    "middle east", "uae", "saudi arabia", "brazil", "south korea",
]

_OG_SUBSEGMENTS = [
    "upstream", "downstream", "midstream", "drilling", "production",
    "reservoir", "well", "oilfield services", "SCADA", "IoT",
]

_GENERIC_TOPIC_WORDS = {
    "petroleum", "petroleum engineering", "oil", "gas", "oil and gas",
    "oil & gas", "energy", "engineering", "research", "paper", "papers",
    "study", "studies", "article", "articles", "journal", "journals",
}

_OUTPUT_NOISE_PATTERNS = [
    r"\bwith\s+authors\b.*$",
    r"\bwith\s+abstracts?\b.*$",
    r"\bwith\s+doi\b.*$",
    r"\bexport\s+as\s+\w+\b.*$",
    r"\bexport\b.*$",
    r"\bas\s+pdf\b.*$",
    r"\bas\s+csv\b.*$",
    r"\bas\s+xlsx\b.*$",
    r"\bas\s+excel\b.*$",
    r"\bdownload\b.*$",
]

_TOPIC_ALIAS_RULES = [
    (["sucker rod pump", "rod pump", "beam pump", "rod lift", "srp"], ['"sucker rod pump"', "SRP", '"rod pump"', '"beam pump"', '"rod lift"']),
    (["electrical submersible pump", "electric submersible pump", "esp"], ['"electrical submersible pump"', "ESP", '"electric submersible pump"']),
    (["progressing cavity pump", "pcp"], ['"progressing cavity pump"', "PCP"]),
    (["gas lift"], ['"gas lift"']),
    (["artificial lift"], ['"artificial lift"']),
]

_SOLUTION_SYNONYMS = {
    "machine learning": ["machine learning", "ML"],
    "artificial intelligence": ["artificial intelligence"],
    "ai": ["AI"],
    "analytics": ["analytics", "analytic platform"],
    "monitoring": ["monitoring", "remote monitoring"],
    "optimization": ["optimization", "optimisation"],
    "automation": ["automation", "automated"],
    "iot": ["IoT", "internet of things"],
    "scada": ["SCADA"],
    "digital twin": ["digital twin"],
    "predictive maintenance": ["predictive maintenance"],
}

_DOMAIN_SYNONYMS = {
    "esp": ["ESP", "electrical submersible pump"],
    "virtual flow metering": ["virtual flow metering", "virtual flow meter"],
    "well performance": ["well performance"],
    "artificial lift": ["artificial lift"],
    "production optimization": ["production optimization"],
    "well surveillance": ["well surveillance"],
    "multiphase metering": ["multiphase metering", "multiphase meter"],
    "flow assurance": ["flow assurance"],
    "production monitoring": ["production monitoring"],
    "reservoir simulation": ["reservoir simulation"],
    "reservoir modeling": ["reservoir modeling"],
    "drilling optimization": ["drilling optimization"],
    "production engineering": ["production engineering"],
}


def plan_queries(task_spec: TaskSpec, llm=None) -> Dict[str, List[SearchQuery]]:
    max_r = getattr(task_spec, "max_results", 25) or 25
    templates = _plan_from_templates(task_spec, max_results=max_r)

    if task_spec.task_type in {"people_search", "document_research"}:
        return templates

    if llm and llm.is_available():
        try:
            llm_result = _plan_with_llm(task_spec, llm)
            if llm_result and any(len(v) >= 1 for v in llm_result.values()):
                merged = {}
                for provider in ("ddg", "exa", "tavily", "serpapi"):
                    llm_q = llm_result.get(provider, [])
                    tmpl_q = templates.get(provider, [])
                    seen = {q.text.lower()[:160] for q in llm_q}
                    extras = [q for q in tmpl_q if q.text.lower()[:160] not in seen]
                    merged[provider] = llm_q + extras
                return merged
        except Exception:
            pass

    return templates


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_output_noise(text: str) -> str:
    cleaned = _norm_spaces(text)
    for pattern in _OUTPUT_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _norm_spaces(cleaned.strip(" -,:;|"))


def _clean_topic(topic: str, raw_prompt: str = "") -> str:
    t = _norm_spaces(topic)
    return _strip_output_noise(t) or _norm_spaces(raw_prompt)


def _solution_terms(task_spec: TaskSpec) -> List[str]:
    out: List[str] = []
    seen = set()
    for kw in (getattr(task_spec, "solution_keywords", []) or []):
        key = str(kw).strip().lower()
        if not key:
            continue
        for synonym in _SOLUTION_SYNONYMS.get(key, [key]):
            if synonym.lower() not in seen:
                seen.add(synonym.lower())
                out.append(synonym)
    return out[:6]


def _domain_terms(task_spec: TaskSpec) -> List[str]:
    out: List[str] = []
    seen = set()
    for kw in (getattr(task_spec, "domain_keywords", []) or []):
        key = str(kw).strip().lower()
        if not key:
            continue
        for synonym in _DOMAIN_SYNONYMS.get(key, [key]):
            if synonym.lower() not in seen:
                seen.add(synonym.lower())
                out.append(synonym)
    return out[:6]


def _commercial_terms(task_spec: TaskSpec) -> List[str]:
    ci = (getattr(task_spec, "commercial_intent", "general") or "general").strip().lower()
    if ci == "agent_or_distributor":
        return ["distributor", "local representative", "channel partner", "reseller"]
    if ci == "reseller":
        return ["reseller", "channel partner", "partner program"]
    if ci == "partner":
        return ["partner", "channel partner", "partner program", "alliance"]
    return []


def _category_profile(task_spec: TaskSpec) -> dict:
    category = getattr(task_spec, "target_category", "general") or "general"
    if category == "software_company":
        return {
            "entity_kw": "software company",
            "semantic_prefix": "B2B digital, software, AI, analytics, automation, SCADA and IoT vendors",
            "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking -stock -marketcap',
        }
    if category == "service_company":
        return {
            "entity_kw": "service company",
            "semantic_prefix": "oilfield service and engineering companies",
            "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking',
        }
    return {
        "entity_kw": "company",
        "semantic_prefix": "real companies",
        "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking',
    }


def _plan_with_llm(task_spec: TaskSpec, llm) -> Dict[str, List[SearchQuery]]:
    topic = _clean_topic(task_spec.industry or task_spec.raw_prompt, task_spec.raw_prompt or "")
    ent_type = (task_spec.target_entity_types or ["company"])[0]
    inc_c = task_spec.geography.include_countries or []
    exc_c = task_spec.geography.exclude_countries or []
    exc_presence = task_spec.geography.exclude_presence_countries or []
    solution_terms = _solution_terms(task_spec)
    domain_terms = _domain_terms(task_spec)
    commercial_terms = _commercial_terms(task_spec)

    profile = _category_profile(task_spec)
    category_desc = getattr(task_spec, "target_category", "general") or "general"

    prompt = QUERY_PLAN_PROMPT.format(
        topic_description=f"{topic} {profile['entity_kw']} ({category_desc})",
        task_type=task_spec.task_type,
        entity_category=task_spec.target_category or "general",
        entity_type=ent_type,
        include_countries=", ".join(inc_c) if inc_c else "any",
        exclude_countries=", ".join(exc_c) if exc_c else "none",
        exclude_presence_countries=", ".join(exc_presence) if exc_presence else "none",
        solution_keywords=", ".join(solution_terms) if solution_terms else "none",
        domain_keywords=", ".join(domain_terms) if domain_terms else "none",
        commercial_intent=(getattr(task_spec, "commercial_intent", "general") or "general"),
        topic_words=" ".join((topic.split() + solution_terms + domain_terms + commercial_terms)[:10]),
    )

    raw = llm.generate_json(prompt, timeout=40)
    if not raw or not isinstance(raw, dict):
        return {}

    result: Dict[str, List[SearchQuery]] = {}
    for provider in ("ddg", "exa", "tavily", "serpapi"):
        items = raw.get(provider, [])
        if not isinstance(items, list):
            continue
        queries = []
        for i, item in enumerate(items):
            text = item if isinstance(item, str) else (item.get("text") or item.get("query") or "")
            text = text.strip()
            if text and len(text) > 4:
                queries.append(SearchQuery(text=text, priority=i + 1, family="llm_generated", provider_hint=provider))
        result[provider] = queries
    return result


def _plan_from_templates(task_spec: TaskSpec, max_results: int = 25) -> Dict[str, List[SearchQuery]]:
    topic = _clean_topic(task_spec.industry or "", task_spec.raw_prompt or "")
    task_type = task_spec.task_type
    inc_c = task_spec.geography.include_countries or []
    exc_c = task_spec.geography.exclude_countries or []
    exc_presence = task_spec.geography.exclude_presence_countries or []
    solution_terms = _solution_terms(task_spec)
    domain_terms = _domain_terms(task_spec)
    commercial_terms = _commercial_terms(task_spec)

    if task_type == "document_research":
        return {"ddg": [], "exa": [], "tavily": [], "serpapi": []}

    profile = _category_profile(task_spec)
    entity_kw = profile["entity_kw"]

    ddg_queries: List[SearchQuery] = []
    p = 1
    geo_terms = inc_c[:4] if inc_c else _REGIONS_GLOBAL[:6]

    for geo in geo_terms:
        base_terms = [topic, entity_kw]
        if solution_terms:
            base_terms.append(solution_terms[0])
        if domain_terms:
            base_terms.append(domain_terms[0])
        q = " ".join([x for x in base_terms if x] + [geo])
        ddg_queries.append(SearchQuery(text=q, priority=p, family="geo", provider_hint="ddg"))
        p += 1

    for dt in domain_terms[:3]:
        q = f"{topic} {dt} {entity_kw}"
        if inc_c:
            q += f" {inc_c[0]}"
        ddg_queries.append(SearchQuery(text=q, priority=p, family="domain", provider_hint="ddg"))
        p += 1

    for st in solution_terms[:3]:
        q = f"{topic} {st} {entity_kw}"
        if inc_c:
            q += f" {inc_c[0]}"
        ddg_queries.append(SearchQuery(text=q, priority=p, family="solution", provider_hint="ddg"))
        p += 1

    for ct in commercial_terms[:2]:
        q = f"{topic} {entity_kw} {ct}"
        if domain_terms:
            q += f" {domain_terms[0]}"
        ddg_queries.append(SearchQuery(text=q, priority=p, family="commercial", provider_hint="ddg"))
        p += 1

    exa_queries: List[SearchQuery] = []
    geo_desc = ", ".join(inc_c[:3]) if inc_c else "global markets"
    exa_sentences = [
        f"{profile['semantic_prefix']} serving the {topic} industry in {geo_desc}",
        f"{topic} companies focused on {', '.join(solution_terms[:3])}" if solution_terms else f"{topic} vendors and providers in {geo_desc}",
        f"{topic} companies focused on {', '.join(domain_terms[:3])}" if domain_terms else f"{topic} software vendors in {geo_desc}",
    ]
    if commercial_terms:
        exa_sentences.append(
            f"{topic} vendors with partner programs, distributors, resellers, or representatives"
        )
    for i, sentence in enumerate(exa_sentences, start=1):
        exa_queries.append(SearchQuery(text=_norm_spaces(sentence), priority=i, family="semantic", provider_hint="exa"))

    tavily_queries: List[SearchQuery] = []
    if solution_terms or domain_terms:
        mixed = ", ".join((solution_terms + domain_terms)[:4])
        tavily_queries.append(SearchQuery(
            text=f"Which {topic} companies specialize in {mixed}?",
            priority=1, family="question", provider_hint="tavily"
        ))
    tavily_queries.append(SearchQuery(
        text=f"What are the top {topic} {entity_kw}s?",
        priority=2, family="question", provider_hint="tavily"
    ))

    serpapi_queries: List[SearchQuery] = []
    neg = profile["serp_negatives"]
    if any(c.lower() in {"usa", "united states"} for c in exc_c + exc_presence):
        neg += ' -"houston" -"texas" -"united states" -"new york"'
    if any(c.lower() == "egypt" for c in exc_c + exc_presence):
        neg += ' -"cairo" -"egypt"'

    main_solution = solution_terms[0] if solution_terms else "software"
    main_domain = domain_terms[0] if domain_terms else ""
    geo_kw = inc_c[0] if inc_c else ""
    serpapi_queries.append(SearchQuery(
        text=_norm_spaces(f'"{topic}" "{main_solution}" "{main_domain}" {geo_kw} {neg}'),
        priority=1, family="core", provider_hint="serpapi"
    ))
    serpapi_queries.append(SearchQuery(
        text=_norm_spaces(f'"{topic}" {entity_kw} {geo_kw} {neg}'),
        priority=2, family="core", provider_hint="serpapi"
    ))
    if commercial_terms:
        serpapi_queries.append(SearchQuery(
            text=_norm_spaces(f'"{topic}" vendor distributor OR reseller OR "partner program" {neg}'),
            priority=3, family="commercial", provider_hint="serpapi"
        ))

    return {
        "ddg": ddg_queries,
        "exa": exa_queries,
        "tavily": tavily_queries,
        "serpapi": serpapi_queries,
    }
