"""
llm_query_planner.py
====================
Generates targeted, diverse search queries.

Main improvements in this version:
- Uses target_category to bias discovery toward software/digital vendors
  instead of generic oil & gas operators
- Preserves document-research query quality from earlier versions
- Adds stronger negative terms for search-engine style providers to reduce
  rankings, news, directories, jobs, and other false positives
- Distinguishes include-country, exclude-country, and exclude-presence intent
"""

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

_REGIONS_EUROPE = [
    "uk", "norway", "germany", "netherlands", "france",
    "sweden", "denmark", "finland", "spain", "italy",
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

_DOC_TOPIC_PATTERNS = [
    r"(?:find|search|show|get|give me|list)\s+(?:research\s+)?(?:papers?|articles?|studies?|publications?)\s+(?:about|on|regarding|for)\s+(.+)$",
    r"(?:research\s+)?(?:papers?|articles?|studies?|publications?)\s+(?:about|on|regarding|for)\s+(.+)$",
]

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
    (
        ["sucker rod pump", "rod pump", "beam pump", "rod lift", "srp"],
        ['"sucker rod pump"', "SRP", '"rod pump"', '"beam pump"', '"rod lift"'],
    ),
    (
        ["electrical submersible pump", "electric submersible pump", "esp"],
        ['"electrical submersible pump"', "ESP", '"electric submersible pump"'],
    ),
    (
        ["progressing cavity pump", "pcp"],
        ['"progressing cavity pump"', "PCP"],
    ),
    (
        ["gas lift"],
        ['"gas lift"'],
    ),
    (
        ["artificial lift"],
        ['"artificial lift"'],
    ),
    (
        ["asphaltene"],
        ['"asphaltene"', '"asphaltene deposition"'],
    ),
]


def plan_queries(task_spec: TaskSpec, llm=None) -> Dict[str, List[SearchQuery]]:
    """
    Build a comprehensive, diverse set of queries.
    LLM-enhanced when available; always falls back to templates.
    """
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
                    seen = {q.text.lower()[:100] for q in llm_q}
                    extras = [q for q in tmpl_q if q.text.lower()[:100] not in seen]
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


def _strip_broad_domain_tail(text: str) -> str:
    cleaned = _norm_spaces(text)
    cleaned = re.sub(
        r"""
        \s+
        (?:in|within|for|related\ to|used\ in|applied\ to)
        \s+
        (?:the\s+)?
        (?:
            petroleum(?:\s+engineering)? |
            oil(?:\s+and\s+gas|\s*&\s*gas)? |
            gas |
            energy |
            upstream |
            downstream
        )
        \b.*$
        """,
        "",
        cleaned,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    cleaned = _norm_spaces(cleaned.strip(" -,:;|"))

    words = cleaned.split()
    while len(words) > 1 and words[-1].lower() in {
        "petroleum", "engineering", "energy", "research", "journal", "journals", "studies"
    }:
        words = words[:-1]
    return _norm_spaces(" ".join(words))


def _extract_specific_topic_from_prompt(raw_prompt: str) -> str:
    prompt = _norm_spaces(raw_prompt)
    for pattern in _DOC_TOPIC_PATTERNS:
        m = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = _strip_broad_domain_tail(_strip_output_noise(m.group(1)))
        candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = _norm_spaces(candidate.strip(" .,:;|-"))
        if candidate:
            return candidate
    return ""


def _is_generic_topic(text: str) -> bool:
    t = _norm_spaces(text.lower())
    if not t:
        return True
    if t in _GENERIC_TOPIC_WORDS:
        return True
    words = [
        w for w in re.findall(r"[a-z0-9&+/.-]+", t)
        if w not in {"the", "a", "an", "of", "in", "for", "on", "and", "to", "with", "about"}
    ]
    if not words:
        return True
    return all(w in _GENERIC_TOPIC_WORDS for w in words)


def _clean_topic(topic: str, raw_prompt: str = "") -> str:
    t = _norm_spaces(topic)

    valid_short = {"ai", "bi", "it", "ml", "ar", "vr", "ev", "hl", "er", "hr", "iot", "esg"}
    m = re.match(r"^([a-z]{1,2})\s+(.+)", t, re.I)
    if m and m.group(1).lower() not in valid_short:
        t = m.group(2).strip()

    t = _strip_broad_domain_tail(_strip_output_noise(t))

    extracted = _extract_specific_topic_from_prompt(raw_prompt) if raw_prompt else ""
    if extracted:
        if _is_generic_topic(t):
            t = extracted
        elif t.lower() in extracted.lower() and len(extracted.split()) >= len(t.split()):
            t = extracted

    return t or extracted or _norm_spaces(raw_prompt)


def _topic_aliases(topic: str) -> List[str]:
    lower = topic.lower()

    for keys, aliases in _TOPIC_ALIAS_RULES:
        if any(k in lower for k in keys):
            seen = set()
            out = []
            for a in aliases:
                if a.lower() not in seen:
                    out.append(a)
                    seen.add(a.lower())
            return out

    aliases = [f'"{topic}"']
    if " " in topic:
        aliases.append(topic)
    else:
        aliases.append(f'"{topic}"')

    seen = set()
    out = []
    for a in aliases:
        key = a.lower()
        if key not in seen:
            out.append(a)
            seen.add(key)
    return out


def _category_profile(task_spec: TaskSpec) -> dict:
    category = getattr(task_spec, "target_category", "general") or "general"

    if category == "software_company":
        return {
            "entity_kw": "software company",
            "variants": [
                "digital company",
                "software company",
                "technology vendor",
                "platform provider",
                "analytics company",
                "automation company",
                "AI company",
                "IoT company",
            ],
            "subsegments": ["software", "platform", "analytics", "automation", "SCADA", "IoT"],
            "semantic_prefix": "B2B digital, software, AI, analytics, automation, SCADA and IoT vendors",
            "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking -stock -marketcap -market-cap -price -photo -images',
        }

    if category == "service_company":
        return {
            "entity_kw": "service company",
            "variants": ["service company", "contractor", "engineering services provider", "oilfield service company"],
            "subsegments": ["drilling", "completion", "inspection", "maintenance"],
            "semantic_prefix": "oilfield service and engineering companies",
            "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking',
        }

    return {
        "entity_kw": "company",
        "variants": ["company", "vendor", "provider"],
        "subsegments": ["solutions", "technology", "platform", "services"],
        "semantic_prefix": "real companies",
        "serp_negatives": '-jobs -job -career -careers -news -article -blog -wiki -directory -ranking -stock -marketcap',
    }


def _geography_description(task_spec: TaskSpec, inc_c: list[str], exc_c: list[str], exc_presence: list[str]) -> str:
    if inc_c:
        return f"based in or operating in {', '.join(inc_c[:3])}"

    if exc_presence:
        return (
            f"excluding presence in {', '.join(exc_presence[:2])}, "
            "operating in Europe, Middle East, Asia, Canada or Australia"
        )

    if exc_c:
        return (
            f"not headquartered in {', '.join(exc_c[:2])}, "
            "operating globally, in Europe, Middle East, Asia or rest of world"
        )

    return "operating globally"


def _plan_with_llm(task_spec: TaskSpec, llm) -> Dict[str, List[SearchQuery]]:
    topic = _clean_topic(task_spec.industry or task_spec.raw_prompt, task_spec.raw_prompt or "")
    ent_type = (task_spec.target_entity_types or ["company"])[0]
    inc_c = task_spec.geography.include_countries or []
    exc_c = task_spec.geography.exclude_countries or []
    exc_presence = task_spec.geography.exclude_presence_countries or []

    profile = _category_profile(task_spec)
    category_desc = getattr(task_spec, "target_category", "general") or "general"

    prompt = QUERY_PLAN_PROMPT.format(
        topic_description=f"{topic} {profile['entity_kw']} ({category_desc})",
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
            text = item if isinstance(item, str) else (item.get("text") or item.get("query") or "")
            text = text.strip()
            if text and len(text) > 4:
                queries.append(
                    SearchQuery(
                        text=text,
                        priority=i + 1,
                        family="llm_generated",
                        provider_hint=provider,
                    )
                )
        result[provider] = queries
    return result


def _plan_from_templates(task_spec: TaskSpec, max_results: int = 25) -> Dict[str, List[SearchQuery]]:
    topic = _clean_topic(task_spec.industry or "", task_spec.raw_prompt or "")
    task_type = task_spec.task_type
    ent_type = (task_spec.target_entity_types or ["company"])[0]
    inc_c = task_spec.geography.include_countries or []
    exc_c = task_spec.geography.exclude_countries or []
    exc_presence = task_spec.geography.exclude_presence_countries or []

    if task_type == "document_research":
        return _paper_queries(topic, raw_prompt=task_spec.raw_prompt or "", max_results=max_results)

    if task_type == "people_search":
        return _people_queries(topic, task_spec, max_results)

    profile = _category_profile(task_spec)
    entity_kw = profile["entity_kw"]
    variants = profile["variants"]

    n_geo_anchors = min(4, max(2, max_results // 15))
    n_subsegments = min(4, max(0, max_results // 25))

    if inc_c:
        geo_anchors = inc_c[:max(4, n_geo_anchors)]
    elif exc_presence:
        geo_anchors = _REGIONS_GLOBAL[:max(6, n_geo_anchors * 2)]
    elif exc_c:
        geo_anchors = _REGIONS_GLOBAL[:max(6, n_geo_anchors * 2)]
    else:
        geo_anchors = _REGIONS_GLOBAL[:n_geo_anchors]

    ddg_queries: List[SearchQuery] = []
    p = 1

    for geo in geo_anchors:
        for variant in variants[:3]:
            ddg_queries.append(
                SearchQuery(
                    text=f"{topic} {variant} {geo}",
                    priority=p,
                    family="geo",
                    provider_hint="ddg",
                )
            )
            p += 1

    if n_subsegments > 0 and topic:
        if getattr(task_spec, "target_category", "general") == "software_company":
            segs = profile["subsegments"][:max(4, n_subsegments)]
        else:
            og_topic = any(w in topic.lower() for w in ["oil", "gas", "petroleum", "energy"])
            segs = _OG_SUBSEGMENTS[:n_subsegments] if og_topic else profile["subsegments"][:n_subsegments]

        primary_geo = inc_c[0] if inc_c else geo_anchors[0]
        for seg in segs:
            ddg_queries.append(
                SearchQuery(
                    text=f"{topic} {seg} {entity_kw} {primary_geo}",
                    priority=p,
                    family="subsegment",
                    provider_hint="ddg",
                )
            )
            p += 1

    if max_results >= 50:
        extra_regions = _REGIONS_EUROPE[:3] if not inc_c else inc_c[1:4]
        for geo in extra_regions:
            if geo not in geo_anchors:
                ddg_queries.append(
                    SearchQuery(
                        text=f"{topic} {entity_kw} {geo}",
                        priority=p,
                        family="extra",
                        provider_hint="ddg",
                    )
                )
                p += 1

    exa_queries: List[SearchQuery] = []
    geo_desc = _geography_description(task_spec, inc_c, exc_c, exc_presence)

    exa_sentences = [
        f"{profile['semantic_prefix']} serving the {topic} industry, {geo_desc}",
        f"{profile['semantic_prefix']} for the {topic} sector, {geo_desc}",
        f"{topic} vendors, providers and companies {geo_desc}",
    ]

    if getattr(task_spec, "target_category", "general") == "software_company":
        exa_sentences.extend([
            f"{topic} SaaS, analytics, automation and platform vendors, {geo_desc}",
            f"digital transformation and AI companies serving {topic} operators, {geo_desc}",
        ])
    else:
        exa_sentences.extend([
            f"technology vendors and service providers for the {topic} sector, {geo_desc}",
            f"specialized companies serving {topic} operators, {geo_desc}",
        ])

    if max_results >= 50:
        extra_geos = inc_c if inc_c else ["Europe", "Middle East", "Asia Pacific", "Canada", "Australia"]
        for geo in extra_geos[:3]:
            exa_sentences.append(f"Leading {topic} {entity_kw} and vendors in {geo}")

    for i, sentence in enumerate(exa_sentences, start=1):
        exa_queries.append(
            SearchQuery(
                text=re.sub(r"\s+", " ", sentence).strip(),
                priority=i,
                family="semantic",
                provider_hint="exa",
            )
        )

    tavily_queries: List[SearchQuery] = []
    if inc_c:
        geo_q = f"in {', '.join(inc_c[:2])}"
    elif exc_presence:
        geo_q = f"excluding presence in {', '.join(exc_presence[:2])}"
    elif exc_c:
        geo_q = f"excluding {', '.join(exc_c[:2])}"
    else:
        geo_q = "globally"

    tavily_base = [
        f"What are the top {topic} {entity_kw} {geo_q}?",
        f"Which {topic} vendors and providers are leading {geo_q}?",
        f"Who provides the best {topic} technology solutions {geo_q}?",
    ]
    if getattr(task_spec, "target_category", "general") == "software_company":
        tavily_base.extend([
            f"What are the best {topic} software platforms available {geo_q}?",
            f"Which digital and analytics vendors serve the {topic} industry {geo_q}?",
        ])
    else:
        tavily_base.append(f"What are the best {topic} companies available {geo_q}?")

    if max_results >= 50:
        tavily_base += [
            f"Which {topic} companies are known in Europe and UK?",
            f"What are the emerging {topic} startups {geo_q}?",
        ]

    for i, q in enumerate(tavily_base, start=1):
        tavily_queries.append(
            SearchQuery(
                text=re.sub(r"\s+", " ", q.strip()),
                priority=i,
                family="question",
                provider_hint="tavily",
            )
        )

    serpapi_queries: List[SearchQuery] = []
    neg = profile["serp_negatives"]
    if "usa" in exc_c or "usa" in exc_presence:
        neg += ' -"houston" -"texas" -"united states" -"new york"'
    if "egypt" in exc_c or "egypt" in exc_presence:
        neg += ' -"cairo" -"egypt"'

    geo_kw = inc_c[0] if inc_c else ""
    if getattr(task_spec, "target_category", "general") == "software_company":
        serp_base = [
            f'"{topic}" software company {geo_kw} {neg}'.strip(),
            f'"{topic}" digital platform vendor europe {neg}'.strip(),
            f'"{topic}" analytics company canada {neg}'.strip(),
            f'"{topic}" automation company australia {neg}'.strip(),
        ]
    else:
        serp_base = [
            f"{topic} {entity_kw} {geo_kw} {neg}".strip(),
            f"{topic} solutions vendor {geo_kw} {neg}".strip(),
            f"best {topic} technology {geo_kw} {neg}".strip(),
            f"{topic} software provider europe {neg}".strip(),
        ]

    for i, q in enumerate(serp_base, start=1):
        serpapi_queries.append(
            SearchQuery(
                text=re.sub(r"\s+", " ", q).strip(),
                priority=i,
                family="core",
                provider_hint="serpapi",
            )
        )

    return {
        "ddg": ddg_queries,
        "exa": exa_queries,
        "tavily": tavily_queries,
        "serpapi": serpapi_queries,
    }


def _paper_queries(topic: str, raw_prompt: str = "", max_results: int = 25) -> Dict[str, List[SearchQuery]]:
    """
    Targeted academic/document queries for engineering and petroleum literature.
    """
    topic = _clean_topic(topic, raw_prompt)
    aliases = _topic_aliases(topic)
    a1 = aliases[0]
    a2 = aliases[1] if len(aliases) > 1 else aliases[0]
    a3 = aliases[2] if len(aliases) > 2 else aliases[0]

    sucker_rod_mode = any(k in topic.lower() for k in ["sucker rod pump", "rod pump", "beam pump", "rod lift", "srp"])

    ddg_texts = [
        f'{a1} petroleum site:onepetro.org',
        f'{a2} petroleum site:spe.org',
        f'{a3} petroleum site:sciencedirect.com',
        f'{a1} "artificial lift" paper authors abstract',
        f'{a2} production optimization case study petroleum',
        f'{a3} failure analysis petroleum paper',
    ]

    if sucker_rod_mode:
        ddg_texts += [
            f'{a1} dynamometer card diagnosis onepetro',
            f'{a2} pump fillage rod string design paper',
            f'{a3} rod lift performance optimization study',
        ]

    if max_results >= 50:
        ddg_texts += [
            f'{a1} conference paper authors doi',
            f'{a2} review paper petroleum engineering',
        ]

    ddg = [
        SearchQuery(text=q, priority=i + 1, family="academic", provider_hint="ddg")
        for i, q in enumerate(ddg_texts)
    ]

    exa_sentences = [
        f"Peer-reviewed papers, conference papers, and case studies on {topic} in petroleum production, with authors, DOI, and abstract.",
        f"Academic research on {topic}, including performance analysis, optimization, and failure diagnosis in oilfield operations.",
        f"Engineering journal articles and OnePetro-style technical papers about {topic} for petroleum wells and artificial lift systems.",
        f"Recent technical studies on {topic} in petroleum engineering with field applications, models, and experiments.",
    ]

    if sucker_rod_mode:
        exa_sentences += [
            f"Research papers on sucker rod pump systems covering rod lift design, pump fillage, dynamometer cards, and production optimization.",
            f"Technical studies on rod pump failures, diagnostics, and artificial lift performance in petroleum wells.",
        ]

    if max_results >= 50:
        exa_sentences += [
            f"Review papers and comparative studies on {topic} methods, field performance, and operating parameters.",
            f"Laboratory, modeling, and field-validation studies on {topic} in petroleum engineering.",
        ]

    exa = [
        SearchQuery(text=re.sub(r"\s+", " ", q).strip(), priority=i + 1, family="semantic", provider_hint="exa")
        for i, q in enumerate(exa_sentences)
    ]

    tavily_questions = [
        f"What are the key research papers on {topic} in petroleum production?",
        f"What are the latest technical studies on {topic} for artificial lift and well performance?",
        f"Which authors and papers are most cited for {topic} in petroleum engineering?",
    ]
    if sucker_rod_mode:
        tavily_questions += [
            f"What papers discuss sucker rod pump optimization, pump fillage, and rod-string behavior?",
            f"What studies use dynamometer cards to diagnose sucker rod pump problems?",
        ]

    tavily = [
        SearchQuery(text=q, priority=i + 1, family="question", provider_hint="tavily")
        for i, q in enumerate(tavily_questions)
    ]

    serpapi_texts = [
        f'{a1} petroleum paper site:onepetro.org',
        f'{a2} petroleum journal article authors site:sciencedirect.com',
        f'{a3} abstract doi site:spe.org',
        f'{a1} "artificial lift" paper',
    ]
    if sucker_rod_mode:
        serpapi_texts += [
            f'{a1} dynamometer card paper site:onepetro.org',
            f'{a2} rod lift optimization paper',
        ]

    serpapi = [
        SearchQuery(text=re.sub(r"\s+", " ", q).strip(), priority=i + 1, family="core", provider_hint="serpapi")
        for i, q in enumerate(serpapi_texts)
    ]

    return {
        "ddg": ddg,
        "exa": exa,
        "tavily": tavily,
        "serpapi": serpapi,
    }


def _people_queries(topic: str, task_spec: TaskSpec, max_results: int = 25) -> Dict[str, List[SearchQuery]]:
    """
    LinkedIn people search queries.
    """
    from core.people_search import build_linkedin_queries

    people_noise = {
        "profiles", "profile", "engineers", "managers", "manager", "engineer",
        "director", "hr", "executives", "professionals", "employees", "staff",
        "personnel", "linkedin", "accounts", "account",
    }
    clean_topic = " ".join(w for w in topic.lower().split() if w not in people_noise).strip() or "oil gas service"

    job_levels = getattr(task_spec, "job_levels", None) or ["engineer", "manager", "hr"]
    countries = task_spec.geography.include_countries or []
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
