"""
llm_query_planner.py  —  v5 (final)
=====================================
Generates maximally diverse, targeted search queries.

KEY PRINCIPLES:
1. DIVERSITY — each query must target a different geographic region, industry
   sub-segment, or company type. Diversity = more unique domains = more results.
2. GEO-ANCHORED — every DDG query includes a region/country to prevent
   returning Dutch/French/Spanish garbage pages.
3. SCALABLE — generates more queries when max_results is large.
4. PAPER MODE — dedicated academic queries with site: operators.
5. LLM-ENHANCED — LLM produces additional queries; merged with templates.
"""
from __future__ import annotations

import re
from typing import Dict, List

from core.models import SearchQuery
from core.task_models import TaskSpec
from core.prompt_templates import QUERY_PLAN_PROMPT


# Geographic regions for diversity — DDG needs these to stay on-topic
_REGIONS_GLOBAL = [
    "europe", "uk", "norway", "germany", "netherlands", "france",
    "canada", "australia", "singapore", "india", "japan",
    "middle east", "uae", "saudi arabia", "brazil", "south korea",
]

_REGIONS_EUROPE = [
    "uk", "norway", "germany", "netherlands", "france",
    "sweden", "denmark", "finland", "spain", "italy",
]

# Industry sub-segments for oil & gas to generate diverse queries
_OG_SUBSEGMENTS = [
    "upstream", "downstream", "midstream",
    "drilling", "production", "reservoir", "well",
    "oilfield services", "SCADA", "IoT",
]


def plan_queries(task_spec: TaskSpec, llm=None) -> Dict[str, List[SearchQuery]]:
    """
    Build a comprehensive, diverse set of queries.
    LLM-enhanced when available; always falls back to templates.
    Scales the number of queries with max_results.

    NOTE: people_search and document_research ALWAYS use templates only.
    The LLM generates company-finding queries by default which are wrong
    for these task types.
    """
    max_r = getattr(task_spec, "max_results", 25) or 25
    templates = _plan_from_templates(task_spec, max_results=max_r)

    # For people_search: NEVER use LLM queries — it generates company searches
    # For document_research: templates are already academic-specific
    if task_spec.task_type in {"people_search", "document_research"}:
        return templates

    if llm and llm.is_available():
        try:
            llm_result = _plan_with_llm(task_spec, llm)
            if llm_result and any(len(v) >= 1 for v in llm_result.values()):
                merged = {}
                for provider in ("ddg", "exa", "tavily", "serpapi"):
                    llm_q  = llm_result.get(provider, [])
                    tmpl_q = templates.get(provider, [])
                    seen   = {q.text.lower()[:60] for q in llm_q}
                    extras = [q for q in tmpl_q if q.text.lower()[:60] not in seen]
                    merged[provider] = llm_q + extras
                return merged
        except Exception:
            pass

    return templates


def _clean_topic(topic: str) -> str:
    """Strip LLM hallucination 1-2 letter prefixes ('al ', 'a ') but keep valid acronyms."""
    t = (topic or "").strip()
    VALID_SHORT = {"ai", "bi", "it", "ml", "ar", "vr", "ev", "hl", "er", "hr", "iot", "esg"}
    m = re.match(r"^([a-z]{1,2})\s+(.+)", t, re.I)
    if m and m.group(1).lower() not in VALID_SHORT:
        t = m.group(2).strip()
    return t.strip()


def _plan_with_llm(task_spec: TaskSpec, llm) -> Dict[str, List[SearchQuery]]:
    topic    = _clean_topic(task_spec.industry or task_spec.raw_prompt)
    ent_type = (task_spec.target_entity_types or ["company"])[0]
    inc_c    = task_spec.geography.include_countries or []
    exc_c    = task_spec.geography.exclude_countries or []

    prompt = QUERY_PLAN_PROMPT.format(
        topic_description=f"{topic} {ent_type}",
        task_type=task_spec.task_type,
        entity_category=task_spec.target_category or "general",
        entity_type=ent_type,
        include_countries=", ".join(inc_c) if inc_c else "any",
        exclude_countries=", ".join(exc_c) if exc_c else "none",
        topic_words=" ".join(topic.split()[:6]),
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
            text = item if isinstance(item, str) else (
                item.get("text") or item.get("query") or ""
            )
            text = text.strip()
            if text and len(text) > 4:
                queries.append(SearchQuery(
                    text=text, priority=i + 1,
                    family="llm_generated", provider_hint=provider,
                ))
        result[provider] = queries
    return result


def _plan_from_templates(
    task_spec: TaskSpec,
    max_results: int = 25,
) -> Dict[str, List[SearchQuery]]:
    """
    Template queries. Always geo-anchored for DDG.
    Generates more queries when more results are needed.
    """
    topic     = _clean_topic(task_spec.industry or "")
    task_type = task_spec.task_type
    ent_type  = (task_spec.target_entity_types or ["company"])[0]
    inc_c     = task_spec.geography.include_countries or []
    exc_c     = task_spec.geography.exclude_countries or []

    # Paper searches get specialist queries
    if task_type == "document_research":
        return _paper_queries(topic, max_results)

    # People / LinkedIn search
    if task_type == "people_search":
        return _people_queries(topic, task_spec, max_results)

    # ── Determine how many queries to generate ─────────────────────────────
    # More results needed → more geographic diversity
    n_geo_anchors  = min(4, max(2, max_results // 15))   # 2-4 for ≤60, 4+ for >60
    n_subsegments  = min(4, max(0, max_results // 25))   # 0 for ≤25, up to 4 for >100

    # ── Build geographic anchors ───────────────────────────────────────────
    if inc_c:
        # Search within these countries — one query per country
        geo_anchors = inc_c[:max(4, n_geo_anchors)]
    elif exc_c:
        # Searching globally but excluding certain countries
        # Use diverse regional anchors to maximize coverage
        geo_anchors = _REGIONS_GLOBAL[:max(6, n_geo_anchors * 2)]
    else:
        geo_anchors = _REGIONS_GLOBAL[:n_geo_anchors]

    entity_kw = {
        "company":      "company",
        "organization": "organization",
        "person":       "expert",
    }.get(ent_type, "company")

    # ── DDG: keyword queries — ALWAYS with geo anchor ─────────────────────
    ddg_queries: List[SearchQuery] = []
    p = 1

    # Core: topic + entity + each geo anchor
    for geo in geo_anchors:
        ddg_queries.append(SearchQuery(
            text=f"{topic} {entity_kw} {geo}",
            priority=p, family="geo", provider_hint="ddg",
        ))
        p += 1

    # Industry sub-segment variants (only when max_results is large)
    if n_subsegments > 0 and topic:
        # For oil & gas, use specific sub-segments
        og_topic = any(w in topic.lower() for w in ["oil", "gas", "petroleum", "energy"])
        if og_topic:
            segs = _OG_SUBSEGMENTS[:n_subsegments]
        else:
            # Generic: use "software", "technology", "services" variants
            segs = ["software", "technology", "platform", "solutions"][:n_subsegments]

        primary_geo = inc_c[0] if inc_c else geo_anchors[0]
        for seg in segs:
            ddg_queries.append(SearchQuery(
                text=f"{topic} {seg} {entity_kw} {primary_geo}",
                priority=p, family="subsegment", provider_hint="ddg",
            ))
            p += 1

    # Extra geo diversity for large requests
    if max_results >= 50:
        extra_regions = _REGIONS_EUROPE[:3] if not inc_c else inc_c[1:4]
        for geo in extra_regions:
            if geo not in geo_anchors:
                ddg_queries.append(SearchQuery(
                    text=f"{topic} vendor {geo}",
                    priority=p, family="extra", provider_hint="ddg",
                ))
                p += 1

    # ── EXA: semantic sentences ────────────────────────────────────────────
    exa_queries: List[SearchQuery] = []
    p = 1

    if inc_c:
        geo_desc = f"based in or operating in {', '.join(inc_c[:3])}"
    elif exc_c:
        geo_desc = (
            f"not headquartered in {', '.join(exc_c[:2])}, "
            "operating globally, in Europe, Middle East, Asia or rest of world"
        )
    else:
        geo_desc = "operating globally"

    # Core EXA semantic sentences
    exa_sentences = [
        f"Real {topic} companies providing solutions {geo_desc}",
        f"Technology vendors and service providers for the {topic} sector {geo_desc}",
        f"SaaS platforms and analytics software in the {topic} industry {geo_desc}",
        f"Digital transformation and AI companies serving {topic} operators {geo_desc}",
        f"Engineering firms and specialized contractors in {topic} {geo_desc}",
    ]

    # Extra EXA sentences for large requests — regional focus
    if max_results >= 50:
        if inc_c:
            extra_geos = inc_c
        else:
            extra_geos = ["Europe", "Middle East", "Asia Pacific", "Canada", "Australia"]

        for geo in extra_geos[:3]:
            exa_sentences.append(
                f"Leading {topic} technology companies and software vendors in {geo}"
            )

    for sentence in exa_sentences:
        exa_queries.append(SearchQuery(
            text=re.sub(r"\s+", " ", sentence).strip(),
            priority=p, family="semantic", provider_hint="exa",
        ))
        p += 1

    # ── Tavily: question style ─────────────────────────────────────────────
    tavily_queries: List[SearchQuery] = []
    p = 1

    if inc_c:
        geo_q = f"in {', '.join(inc_c[:2])}"
    elif exc_c:
        geo_q = "outside USA"
    else:
        geo_q = "globally"

    tavily_base = [
        f"What are the top {topic} companies {geo_q}?",
        f"Which {topic} vendors and providers are leading {geo_q}?",
        f"Who provides the best {topic} technology solutions {geo_q}?",
        f"What are the best {topic} software platforms available {geo_q}?",
    ]
    if max_results >= 50:
        tavily_base += [
            f"Which {topic} companies are known in Europe and UK?",
            f"What are the emerging {topic} startups {geo_q}?",
        ]

    for q in tavily_base:
        tavily_queries.append(SearchQuery(
            text=re.sub(r"\s+", " ", q.strip()),
            priority=p, family="question", provider_hint="tavily",
        ))
        p += 1

    # ── SerpApi: keyword with negative operators ───────────────────────────
    serpapi_queries: List[SearchQuery] = []
    p = 1

    neg = ""
    if "usa" in exc_c:
        neg += ' -"houston" -"texas" -"united states" -"new york"'
    if "egypt" in exc_c:
        neg += ' -"cairo" -"egypt"'

    geo_kw = inc_c[0] if inc_c else ""

    serp_base = [
        f"{topic} {entity_kw} {geo_kw}{neg}".strip(),
        f"{topic} solutions vendor {geo_kw}{neg}".strip(),
        f"best {topic} technology {geo_kw}{neg}".strip(),
        f"{topic} software provider europe{neg}".strip(),
    ]
    for q in serp_base:
        serpapi_queries.append(SearchQuery(
            text=re.sub(r"\s+", " ", q).strip(),
            priority=p, family="core", provider_hint="serpapi",
        ))
        p += 1

    return {
        "ddg":     ddg_queries,
        "exa":     exa_queries,
        "tavily":  tavily_queries,
        "serpapi": serpapi_queries,
    }


def _paper_queries(topic: str, max_results: int = 25) -> Dict[str, List[SearchQuery]]:
    """Targeted academic queries for document_research tasks."""
    ddg = [
        SearchQuery(text=f"{topic} research paper pubmed OR sciencedirect OR onepetro",
                    priority=1, family="academic"),
        SearchQuery(text=f"{topic} journal article authors doi abstract",
                    priority=2, family="academic"),
        SearchQuery(text=f"{topic} peer reviewed study systematic review",
                    priority=3, family="academic"),
        SearchQuery(text=f"{topic} clinical trial randomized controlled",
                    priority=4, family="academic"),
        SearchQuery(text=f"{topic} literature review meta-analysis",
                    priority=5, family="academic"),
    ]
    exa = [
        SearchQuery(
            text=f"Find peer-reviewed research papers and journal articles about {topic}, "
                 f"including clinical studies, case series, and systematic reviews.",
            priority=1, family="semantic"),
        SearchQuery(
            text=f"Academic publications and conference papers on {topic} "
                 f"with author names, DOI, and abstract.",
            priority=2, family="semantic"),
        SearchQuery(
            text=f"Technical studies and experimental investigations on {topic} "
                 f"published in medical or engineering journals.",
            priority=3, family="semantic"),
        SearchQuery(
            text=f"Recent advances and original research articles about {topic} "
                 f"with full author affiliations.",
            priority=4, family="semantic"),
    ]
    if max_results >= 50:
        exa += [
            SearchQuery(
                text=f"Systematic reviews and meta-analyses on {topic} outcomes and efficacy.",
                priority=5, family="semantic"),
            SearchQuery(
                text=f"Case reports and observational studies on {topic} management.",
                priority=6, family="semantic"),
        ]
    tavily = [
        SearchQuery(text=f"What are the key research findings on {topic}?",
                    priority=1, family="question"),
        SearchQuery(text=f"Latest published studies about {topic} treatment and outcomes",
                    priority=2, family="question"),
        SearchQuery(text=f"Who are the leading researchers studying {topic}?",
                    priority=3, family="question"),
    ]
    serpapi = [
        SearchQuery(text=f"{topic} research paper doi abstract", priority=1, family="core"),
        SearchQuery(text=f"{topic} journal article authors",     priority=2, family="core"),
        SearchQuery(text=f"{topic} systematic review",           priority=3, family="core"),
    ]
    return {"ddg": ddg, "exa": exa, "tavily": tavily, "serpapi": serpapi}


def _people_queries(
    topic: str,
    task_spec: TaskSpec,
    max_results: int = 25,
) -> Dict[str, List[SearchQuery]]:
    """
    LinkedIn people search queries.
    Primary: EXA with include_domains=["linkedin.com/in"] — returns ONLY /in/ profiles.
    Secondary: SerpApi site:linkedin.com/in when key available.
    Fallback: DDG (low yield).
    """
    from core.people_search import build_linkedin_queries

    # Clean industry: remove any people-search noise that leaked through
    PEOPLE_NOISE = {
        "profiles", "profile", "engineers", "managers", "manager", "engineer",
        "director", "hr", "executives", "professionals", "employees", "staff",
        "personnel", "linkedin", "accounts", "account",
    }
    clean_topic = " ".join(
        w for w in topic.lower().split()
        if w not in PEOPLE_NOISE
    ).strip() or "oil gas service"

    job_levels = getattr(task_spec, "job_levels", None) or ["engineer", "manager", "hr"]
    countries  = task_spec.geography.include_countries or []
    if not countries:
        countries = task_spec.geography.exclude_countries or []

    use_serpapi = getattr(task_spec, "_use_serpapi_for_people", False)

    return build_linkedin_queries(
        industry=clean_topic,
        job_levels=job_levels,
        countries=countries,
        max_results=max_results,
        use_serpapi=use_serpapi,
    )
