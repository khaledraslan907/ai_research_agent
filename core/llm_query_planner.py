



Pasted text.txt
Document

Pasted text (2).txt
Document
2026 04 18T15 33 export


i made a comprehensive search agent one purpose of it is to find companies but when i type that does not understand so please solve this issue also results of companies does not accurate enough this is the search: Find digital  companies working in oil and gas industry but operate outside Egypt and USA and this is what system understand: Looking for: 🏢 Companies

Topic: oil gas

Target: 40 results in Balanced mode

🚫
Excluding: Egypt, Usa

I’m checking the parser and query-planning path that turns your prompt into “Topic: oil gas.” Then I’ll point to the exact failure and give you a patch you can drop in.

I found the main parser issue: it treats “digital” as a category, but not as part of the actual search topic. That makes the planner fall back to broad oil-and-gas queries, which explains both the weak understanding and weaker company accuracy.

Thought for 5m 9s
I found the core problem.

Your parser already understands that “digital” means software/digital vendors via target_category = "software_company", but your UI summary only shows entity type, topic, and exclusions, so that signal disappears in the display. At the same time, _extract_focus_term() gives priority to the phrase “working in oil and gas industry”, so the topic collapses to oil and gas. That is why the system shows only “Topic: oil gas” even though the prompt also asked for digital companies.

The larger accuracy issue is in the planner: _plan_from_templates() builds very broad queries like {topic} company {geo}, and when the topic contains oil/gas it even injects upstream/downstream-style subsegments. It does not really use target_category="software_company" to force software/digital-company searches, so you end up retrieving operators, articles, rankings, directories, and other false positives. 


There is one more mismatch: your phrase “operate outside Egypt and USA” should behave like a presence exclusion, but the current geography logic only fills exclude_presence_countries for phrases like “no branches in / no offices in / no presence in.” So your wording is likely being treated too loosely. 


What to change
1) Fix the displayed understanding
Your UI should show category separately, not bury it.

def humanize_target_category(cat: str) -> str:
    return {
        "software_company": "Digital / software companies",
        "service_company": "Service / engineering companies",
        "general": "General companies",
    }.get(cat, cat.replace("_", " ").title())

# Example display
print(f"Looking for: 🏢 {entity_label}")
print(f"Category: {humanize_target_category(task_spec.target_category)}")
print(f"Industry: {task_spec.industry or 'Any'}")
print(f"Target: {task_spec.max_results} results in {task_spec.mode} mode")
if task_spec.geography.exclude_presence_countries:
    print("🚫 Excluding presence in:", ", ".join(task_spec.geography.exclude_presence_countries))
elif task_spec.geography.exclude_countries:
    print("🚫 Excluding:", ", ".join(task_spec.geography.exclude_countries))
After this, your prompt should render more like:

Looking for: Companies

Category: Digital / software companies

Industry: Oil and gas

Excluding presence in: Egypt, USA

2) Treat “operate outside …” as presence exclusion
Patch task_parser.py so wording like yours maps to exclude_presence_countries.

# inside _extract_geography()

exclude_presence.extend(_extract_grouped_geo(prompt_lower, [
    r"no branches in", r"no offices in",
    r"without branches in", r"without offices in", r"no presence in",
    r"operate outside", r"operating outside",
    r"work outside", r"working outside",
    r"serve outside", r"serving outside",
    r"active outside", r"present outside",
]))

# if presence exclusion is explicitly detected, don't also treat the same countries
# as generic headquarters exclusions
exclude = [c for c in exclude if c not in exclude_presence]
3) Make the planner honor software_company
This is the main accuracy fix.

Add a category profile in llm_query_planner.py:

def _category_profile(category: str) -> dict:
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
            ],
            "subsegments": ["software", "platform", "analytics", "automation", "SCADA", "IoT"],
            "semantic_prefix": "B2B digital, software, AI, analytics, automation, SCADA and IoT vendors",
            "serp_negatives": '-jobs -career -news -article -blog -wiki -directory -ranking -stock -market -price -photo',
        }

    if category == "service_company":
        return {
            "entity_kw": "service company",
            "variants": ["service company", "contractor", "engineering services provider"],
            "subsegments": ["drilling", "completion", "inspection", "maintenance"],
            "semantic_prefix": "oilfield service and engineering companies",
            "serp_negatives": '-jobs -career -news -article -blog -wiki -directory -ranking',
        }

    return {
        "entity_kw": "company",
        "variants": ["company", "vendor", "provider"],
        "subsegments": ["solutions", "technology", "platform", "services"],
        "semantic_prefix": "real companies",
        "serp_negatives": '-jobs -career -news -article -blog -wiki -directory -ranking',
    }
Then use it inside _plan_from_templates():

profile = _category_profile(task_spec.target_category or "general")
entity_kw = profile["entity_kw"]
variants = profile["variants"]
neg_common = profile["serp_negatives"]

ddg_queries = []
p = 1
for geo in geo_anchors:
    for v in variants[:3]:
        ddg_queries.append(
            SearchQuery(
                text=f'{topic} {v} {geo}',
                priority=p,
                family="geo",
                provider_hint="ddg",
            )
        )
        p += 1

# important: do NOT switch to upstream/downstream subsegments for software companies
if n_subsegments > 0 and topic:
    if task_spec.target_category == "software_company":
        segs = profile["subsegments"][:max(4, n_subsegments)]
    else:
        og_topic = any(w in topic.lower() for w in ["oil", "gas", "petroleum", "energy"])
        segs = _OG_SUBSEGMENTS[:n_subsegments] if og_topic else ["software", "technology", "platform", "solutions"][:n_subsegments]

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

# EXA
if inc_c:
    geo_desc = f"based in or operating in {', '.join(inc_c[:3])}"
elif exc_c:
    geo_desc = f"excluding headquarters/presence in {', '.join(exc_c[:2])}, operating in Europe, Middle East, Asia, Canada or Australia"
else:
    geo_desc = "operating globally"

exa_sentences = [
    f"{profile['semantic_prefix']} serving the {topic} industry, {geo_desc}",
    f"{profile['semantic_prefix']} for oil and gas operators, {geo_desc}",
    f"{topic} SaaS, analytics, automation and platform vendors, {geo_desc}",
]

# Tavily
if inc_c:
    geo_q = f"in {', '.join(inc_c[:2])}"
elif exc_c:
    geo_q = f"excluding Egypt and USA"
else:
    geo_q = "globally"

tavily_base = [
    f"What are the top {topic} software and digital vendors {geo_q}?",
    f"Which B2B technology companies serve the {topic} industry {geo_q}?",
    f"What are the best {topic} software platforms and analytics vendors {geo_q}?",
]

# SerpApi
neg = neg_common
if "usa" in exc_c:
    neg += ' -"houston" -"texas" -"united states" -"new york"'
if "egypt" in exc_c:
    neg += ' -"cairo" -"egypt"'

serp_base = [
    f'"{topic}" software company europe {neg}'.strip(),
    f'"{topic}" digital platform vendor middle east {neg}'.strip(),
    f'"{topic}" analytics company canada {neg}'.strip(),
    f'"{topic}" automation company australia {neg}'.strip(),
]
4) Add a strict post-filter before accepting a result
Your current output shows the agent is accepting non-company pages. A lightweight validator will remove most of that.

from urllib.parse import urlparse

BAD_HOST_HINTS = [
    "alamy.com", "tradingview.com", "wikipedia.org", "euronews.com",
    "nytimes.com", "worldpopulationreview.com", "companiesmarketcap.com",
    "loc.gov", "facebook.com", "instagram.com", "youtube.com"
]

BAD_TEXT_HINTS = [
    "largest oil and gas companies", "market cap", "stock photo", "by country",
    "news", "article", "guide", "introduction", "infographic", "ranking",
]

DIGITAL_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital",
    "ai", "iot", "scada", "optimization", "monitoring", "data platform",
]

OG_HINTS = [
    "oil", "gas", "petroleum", "energy", "upstream", "midstream",
    "downstream", "oilfield",
]

COMPANY_HINTS = [
    "company", "vendor", "provider", "technology", "solutions",
    "platform", "software",
]

def is_valid_company_candidate(title: str, snippet: str, url: str, target_category: str) -> bool:
    host = urlparse(url).netloc.lower()
    text = f"{title} {snippet} {url}".lower()

    if any(h in host for h in BAD_HOST_HINTS):
        return False
    if any(b in text for b in BAD_TEXT_HINTS):
        return False
    if "/news/" in url or "/article/" in url or "/blog/" in url:
        return False

    if not any(x in text for x in OG_HINTS):
        return False
    if not any(x in text for x in COMPANY_HINTS):
        return False

    if target_category == "software_company" and not any(x in text for x in DIGITAL_HINTS):
        return False

    return True
Why this will help immediately
Right now your system is effectively searching for oil and gas companies, not digital/software companies serving oil and gas. That is why you get false positives like news pages, rankings, and general energy firms.

With the fixes above, the agent will:

display the intent correctly

treat “operate outside Egypt and USA” more correctly

search for software/digital vendors instead of general operators

reject obvious non-company pages before export

That should materially improve both understanding and accuracy.



Pasted text.txt
Document
this is app user code also if you want to change in it, please rewrite full update code

I’ve got the app file now. I’m updating it so the UI shows “digital/software companies” correctly, distinguishes “exclude headquarters” from “exclude presence,” and surfaces what the agent actually understood instead of collapsing it to only “oil gas.”

Thought for 9m 5s
I rewrote the full app file and saved it here:

app_user_updated.py

What changed in this updated app:

It now shows Category separately, so prompts like “digital companies in oil and gas” are displayed as Digital / software companies instead of only collapsing to oil gas.

It adds an app-layer fix for phrases like “operate outside Egypt and USA” by treating them as presence exclusion when the parser misses that nuance.

It adds an app-side refinement filter to remove obvious non-company / non-vendor results from what gets shown and exported.

Important note: this app file improves the displayed understanding and cleans the final shown results, but the biggest accuracy gain still comes from updating the backend parser and query planner too, because those are what currently produce the broad oil gas interpretation and generic company queries. Here are the three files that matter together:


Pasted text.txt
Document

Pasted text (2).txt
Document

Pasted text.txt
Document
If you want, next I’ll rewrite the full updated task_parser.py and llm_query_planner.py the same way so the backend itself starts returning much more accurate company results.




Generates targeted, diverse search queries.

Fixes in this version:

Uses the specific paper topic, not the broad domain

Document research queries are engineering/petroleum focused

Adds domain-specific synonym expansion (e.g. sucker rod pump → SRP, rod pump, beam pump)

Removes medical-style paper templates
"""

from future import annotations

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
"upstream", "downstream", "midstream",
"drilling", "production", "reservoir", "well",
"oilfield services", "SCADA", "IoT",
]

_GENERIC_TOPIC_WORDS = {
"petroleum", "petroleum engineering", "oil", "gas", "oil and gas",
"oil & gas", "energy", "engineering", "research", "paper", "papers",
"study", "studies", "article", "articles", "journal", "journals",
}

_DOC_TOPIC_PATTERNS = [
r"(?|search|show|get|give me|list)\s+(?\s+)?(??|articles?|studies?|publications?)\s+(?|on|regarding|for)\s+(.+)$",
r"(?\s+)?(??|articles?|studies?|publications?)\s+(?|on|regarding|for)\s+(.+)$",
]

_OUTPUT_NOISE_PATTERNS = [
r"\bwith\s+authors\b.$",
r"\bwith\s+abstracts?\b.$",
r"\bwith\s+doi\b.$",
r"\bexport\s+as\s+\w+\b.$",
r"\bexport\b.$",
r"\bas\s+pdf\b.$",
r"\bas\s+csv\b.$",
r"\bas\s+xlsx\b.$",
r"\bas\s+excel\b.$",
r"\bdownload\b.$",
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

NOTE:
- people_search and document_research use templates only
- this avoids the LLM drifting into wrong query families
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
                seen = {q.text.lower()[:80] for q in llm_q}
                extras = [q for q in tmpl_q if q.text.lower()[:80] not in seen]
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
(?|within|for|related\ to|used\ in|applied\ to)
\s+
(?\s+)?
(?:
petroleum(?:\s+engineering)? |
oil(?:\s+and\s+gas|\s*&\sgas)? |
gas |
energy |
upstream |
downstream
)
\b.$
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
candidate = re.sub(r"^(?|a|an)\s+", "", candidate, flags=re.IGNORECASE)
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
"""
Clean topic string and repair generic LLM drift from the raw prompt.
"""
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
def _plan_with_llm(task_spec: TaskSpec, llm) -> Dict[str, List[SearchQuery]]:
topic = _clean_topic(task_spec.industry or task_spec.raw_prompt, task_spec.raw_prompt or "")
ent_type = (task_spec.target_entity_types or ["company"])[0]
inc_c = task_spec.geography.include_countries or []
exc_c = task_spec.geography.exclude_countries or []

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
def _plan_from_templates(
task_spec: TaskSpec,
max_results: int = 25,
) -> Dict[str, List[SearchQuery]]:
topic = _clean_topic(task_spec.industry or "", task_spec.raw_prompt or "")
task_type = task_spec.task_type
ent_type = (task_spec.target_entity_types or ["company"])[0]
inc_c = task_spec.geography.include_countries or []
exc_c = task_spec.geography.exclude_countries or []

if task_type == "document_research":
    return _paper_queries(topic, raw_prompt=task_spec.raw_prompt or "", max_results=max_results)

if task_type == "people_search":
    return _people_queries(topic, task_spec, max_results)

n_geo_anchors = min(4, max(2, max_results // 15))
n_subsegments = min(4, max(0, max_results // 25))

if inc_c:
    geo_anchors = inc_c[:max(4, n_geo_anchors)]
elif exc_c:
    geo_anchors = _REGIONS_GLOBAL[:max(6, n_geo_anchors * 2)]
else:
    geo_anchors = _REGIONS_GLOBAL[:n_geo_anchors]

entity_kw = {
    "company": "company",
    "organization": "organization",
    "person": "expert",
}.get(ent_type, "company")

ddg_queries: List[SearchQuery] = []
p = 1

for geo in geo_anchors:
    ddg_queries.append(
        SearchQuery(
            text=f"{topic} {entity_kw} {geo}",
            priority=p,
            family="geo",
            provider_hint="ddg",
        )
    )
    p += 1

if n_subsegments > 0 and topic:
    og_topic = any(w in topic.lower() for w in ["oil", "gas", "petroleum", "energy"])
    segs = _OG_SUBSEGMENTS[:n_subsegments] if og_topic else ["software", "technology", "platform", "solutions"][:n_subsegments]

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
                    text=f"{topic} vendor {geo}",
                    priority=p,
                    family="extra",
                    provider_hint="ddg",
                )
            )
            p += 1

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

exa_sentences = [
    f"Real {topic} companies providing solutions {geo_desc}",
    f"Technology vendors and service providers for the {topic} sector {geo_desc}",
    f"SaaS platforms and analytics software in the {topic} industry {geo_desc}",
    f"Digital transformation and AI companies serving {topic} operators {geo_desc}",
    f"Engineering firms and specialized contractors in {topic} {geo_desc}",
]

if max_results >= 50:
    extra_geos = inc_c if inc_c else ["Europe", "Middle East", "Asia Pacific", "Canada", "Australia"]
    for geo in extra_geos[:3]:
        exa_sentences.append(f"Leading {topic} technology companies and software vendors in {geo}")

for sentence in exa_sentences:
    exa_queries.append(
        SearchQuery(
            text=re.sub(r"\s+", " ", sentence).strip(),
            priority=p,
            family="semantic",
            provider_hint="exa",
        )
    )
    p += 1

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
    tavily_queries.append(
        SearchQuery(
            text=re.sub(r"\s+", " ", q.strip()),
            priority=p,
            family="question",
            provider_hint="tavily",
        )
    )
    p += 1

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
    serpapi_queries.append(
        SearchQuery(
            text=re.sub(r"\s+", " ", q).strip(),
            priority=p,
            family="core",
            provider_hint="serpapi",
        )
    )
    p += 1

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
def _people_queries(
topic: str,
task_spec: TaskSpec,
max_results: int = 25,
) -> Dict[str, List[SearchQuery]]:
"""
LinkedIn people search queries.
"""
from core.people_search import build_linkedin_queries

people_noise = {
    "profiles", "profile", "engineers", "managers", "manager", "engineer",
    "director", "hr", "executives", "professionals", "employees", "staff",
    "personnel", "linkedin", "accounts", "account",
}
clean_topic = " ".join(
    w for w in topic.lower().split()
    if w not in people_noise
).strip() or "oil gas service"

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
