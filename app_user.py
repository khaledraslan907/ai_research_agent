"""
app_user.py — AI Research Agent (updated)
========================================

Main fixes in this version:
- Preserves and displays company category clearly
- Preserves and displays solution keywords, domain keywords, and commercial intent
- Fixes prompt parsing call (uses prompt instead of undefined user_prompt)
- Saves domain keywords into task_meta so they appear in Search Details
- Keeps app-layer postprocessing minimal and aligned with backend parser
"""

from __future__ import annotations

import io
import os
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import CompanyRecord, ProviderSettings

try:
    from core.paper_summarizer import (
        summarize_papers,
        summaries_to_pdf_bytes,
        summaries_to_text,
    )
except Exception:
    from core.paper_summarizer import summarize_papers, summaries_to_text

    def summaries_to_pdf_bytes(summaries, topic):
        return None

from pipeline.orchestrator import SearchOrchestrator


# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "search_result": None,
    "task_meta": None,
    "paper_summaries": [],
    "summaries_topic": "",
    "prompt_value": "",
    "active_section": "results",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8eaf0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
.result-card {
    background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 12px;
}
.result-card:hover { border-color: #4f6ef7; }
.result-name {
    font-family: 'IBM Plex Mono', monospace; font-size: 15px; font-weight: 600;
    color: #ffffff; margin-bottom: 4px;
}
.result-meta { font-size: 13px; color: #7b8099; margin-bottom: 6px; }
.result-link { font-size: 13px; color: #4f6ef7; text-decoration: none; }
.tag {
    display: inline-block; background: #1e2236; border: 1px solid #363d5c;
    border-radius: 4px; padding: 2px 8px; font-size: 11px;
    font-family: 'IBM Plex Mono', monospace; color: #8892b0; margin-right: 6px;
}
.key-step {
    background: #151823; border-left: 3px solid #4f6ef7;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 13px;
}
.key-step a { color: #4f6ef7; }
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f6ef7; margin: 20px 0 8px 0;
    padding-bottom: 6px; border-bottom: 1px solid #2d3148;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


KEY_GUIDE = {
    "groq": {
        "label": "Groq API Key",
        "icon": "⚡",
        "cost": "100% Free · 14,400 requests/day",
        "steps": [
            ("Go to", "https://console.groq.com", "console.groq.com"),
            ("Click", None, '"Sign Up" (free, no credit card)'),
            ("Click", None, '"API Keys" → "Create API Key"'),
            ("Copy", None, "the key starting with gsk_... and paste above"),
        ],
        "why": "Powers smart query planning and result ranking. Most important key.",
    },
    "gemini": {
        "label": "Google Gemini Key",
        "icon": "🧠",
        "cost": "Free · 1,500 requests/day",
        "steps": [
            ("Go to", "https://aistudio.google.com", "aistudio.google.com"),
            ("Click", None, '"Get API Key" → "Create API Key"'),
            ("Copy", None, "the key starting with AIza... and paste above"),
        ],
        "why": "Backup LLM — used when Groq quota is exhausted.",
    },
    "exa": {
        "label": "Exa API Key",
        "icon": "🔭",
        "cost": "Free · 1,000 searches/month",
        "steps": [
            ("Go to", "https://exa.ai", "exa.ai"),
            ("Click", None, '"Sign Up" → confirm email'),
            ("Go to", None, 'Dashboard → "API Keys" → "New Key"'),
            ("Copy", None, "the key and paste above"),
        ],
        "why": "Best semantic search — finds things DDG misses. Needed for LinkedIn people search.",
    },
    "tavily": {
        "label": "Tavily API Key",
        "icon": "🌐",
        "cost": "Free · 1,000 searches/month",
        "steps": [
            ("Go to", "https://tavily.com", "tavily.com"),
            ("Click", None, '"Get Started Free"'),
            ("Copy", None, "your API key from the dashboard"),
        ],
        "why": "Adds question-style search diversity. Helps reach higher result counts.",
    },
    "serpapi": {
        "label": "SerpApi Key",
        "icon": "🎯",
        "cost": "Free · 100 searches/month",
        "steps": [
            ("Go to", "https://serpapi.com", "serpapi.com"),
            ("Click", None, '"Register" (free plan available)'),
            ("Copy", None, "your API key from the dashboard"),
        ],
        "why": "Best for LinkedIn people search via site:linkedin.com queries.",
    },
}

MODE_PRESETS = {
    "🚀 Fast": {"mode": "Fast", "max": 15, "desc": "~15 sec · DDG only · quick check"},
    "⚖️ Balanced": {"mode": "Balanced", "max": 40, "desc": "~1–2 min · best quality/speed ratio · recommended"},
    "🔬 Deep": {"mode": "Deep", "max": 100, "desc": "~5–10 min · all providers · maximum results"},
}

EXAMPLES = [
    ("🏢 Companies", "Find oilfield service companies in Egypt and Saudi Arabia with email and phone"),
    ("🏢 Companies", "Find digital oil and gas software companies outside USA and Egypt with email"),
    ("🏢 Companies", "Find renewable energy companies in Germany and Norway with contact details"),
    ("🏢 Companies", "Find AI and data analytics companies in the energy sector outside USA"),
    ("📄 Papers", "Find research papers about Electrical Submersible Pump with authors export as PDF"),
    ("📄 Papers", "Find papers about asphaltene effect on ESP performance with authors as PDF"),
    ("📄 Papers", "Find papers about carbon capture CCS techniques export as CSV"),
    ("👥 LinkedIn", "Find LinkedIn profiles of petroleum engineers in oil gas companies in Egypt"),
    ("👥 LinkedIn", "Find HR managers and engineers on LinkedIn working in oilfield service companies in Saudi Arabia"),
]


def _show_key_guide(key_id: str):
    g = KEY_GUIDE[key_id]
    st.markdown(f"**{g['icon']} {g['label']}** — {g['cost']}")
    for action, url, text in g["steps"]:
        link = f'<a href="{url}" target="_blank">{text}</a>' if url else text
        st.markdown(f'<div class="key-step">→ {action} {link}</div>', unsafe_allow_html=True)
    st.caption(f"💡 {g['why']}")


def _read_file(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    except Exception:
        return None


def _normalize_filename(name: str, fmt: str | None) -> str:
    ext_map = {"xlsx": ".xlsx", "csv": ".csv", "pdf": ".pdf", "json": ".json", None: ".xlsx"}
    desired_ext = ext_map.get(fmt, ".xlsx")
    name = (name or "results").strip()
    for ext in [".xlsx", ".csv", ".pdf", ".json"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name + desired_ext


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


def _is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _clean_text(value) -> str:
    return "" if _is_blank(value) else str(value).strip()


def _normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe_repeated_prompt(prompt: str) -> str:
    raw = (prompt or "").strip()
    if not raw:
        return raw

    half = len(raw) // 2
    left = raw[:half].strip()
    right = raw[half:].strip()

    if left and right and left == right:
        return left

    if len(raw) > 40:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        deduped = []
        prev = None
        for s in sentences:
            if prev is not None and s == prev:
                continue
            deduped.append(s)
            prev = s
        return " ".join(deduped)

    return raw


def _first_nonempty(record: dict, keys: list[str]) -> str:
    for key in keys:
        if key in record and not _is_blank(record.get(key)):
            return _clean_text(record.get(key))
    return ""


def _parse_year(value) -> str:
    if _is_blank(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if 1900 <= int(f) <= 2100:
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else ""


def _title_list(items: list[str]) -> list[str]:
    out = []
    for item in items or []:
        s = str(item).strip()
        if s:
            out.append(s.title())
    return out


def _humanize_task_type(task_type: str) -> str:
    return {
        "entity_discovery": "🏢 Companies",
        "document_research": "📄 Research Papers",
        "people_search": "👥 LinkedIn Profiles",
        "market_research": "📊 Market Research",
        "entity_enrichment": "🧩 Entity Enrichment",
        "similar_entity_expansion": "🧭 Similar Entities",
    }.get(task_type, "🔍 Entities")


def _humanize_entity_type(entity_type: str) -> str:
    return {
        "company": "🏢 Companies",
        "paper": "📄 Papers",
        "person": "👥 People",
        "tender": "📑 Tenders",
        "event": "🎪 Events",
        "product": "🧰 Products",
    }.get((entity_type or "company").strip(), "🔍 Entities")


def _humanize_target_category(category: str) -> str:
    return {
        "software_company": "Digital / software companies",
        "service_company": "Service / engineering companies",
        "general": "General companies",
    }.get((category or "").strip(), (category or "general").replace("_", " ").title())


def _humanize_commercial_intent(value: str) -> str:
    return {
        "general": "General",
        "agent_or_distributor": "Agent / distributor",
        "reseller": "Reseller",
        "partner": "Partner / channel partner",
    }.get((value or "general").strip(), (value or "general").replace("_", " ").title())


_DIGITAL_CATEGORY_HINTS = (
    "digital", "software", "saas", "platform", "analytics", "automation",
    "ai", "artificial intelligence", "machine learning", "data", "iot",
    "scada", "cloud", "tech company", "technology company", "technology vendor",
)


_PRESENCE_OUTSIDE_PATTERNS = [
    r"\boperate(?:s|d|ing)?\s+outside\b",
    r"\bwork(?:s|ed|ing)?\s+outside\b",
    r"\bserv(?:e|es|ed|ing)\s+outside\b",
    r"\bactive\s+outside\b",
    r"\bpresent\s+outside\b",
    r"\bno\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\bdo(?:es)?\s+not\s+have\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|presence|operations?|legal entities?)\s+in\b",
    r"\b(?:do(?:es)?\s+not|don't|doesn't|cannot|can't)\s+(?:operate|work|serve|have|be\s+active|be\s+present)\b.*?\b(?:in|inside|within)\b",
    r"\bnot\s+(?:operating|working|serving|active|present)\b.*?\b(?:in|inside|within)\b",
    r"\bexclude\b[^.\n;]{0,80}\bpresence\b",
    r"\bexcluding\b[^.\n;]{0,80}\bpresence\b",
]

_SOLUTION_KEYWORD_PATTERNS = {
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "artificial intelligence": [r"\bartificial intelligence\b"],
    "ai": [r"\bai\b"],
    "analytics": [r"\banalytics\b", r"\banalytic\b", r"\binsights\b"],
    "monitoring": [r"\bmonitoring\b", r"\bremote monitoring\b", r"\bsurveillance\b"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b", r"\boptimizer\b", r"\boptimiser\b"],
    "automation": [r"\bautomation\b", r"\bautomated\b", r"\bautonomous\b"],
    "iot": [r"\biot\b", r"\binternet of things\b"],
    "scada": [r"\bscada\b"],
    "digital twin": [r"\bdigital twin\b", r"\bdigital twins\b"],
    "predictive maintenance": [r"\bpredictive maintenance\b"],
}


_DOMAIN_KEYWORD_PATTERNS = {
    "wireline": [r"\bwireline\b", r"\bwire line\b"],
    "slickline": [r"\bslickline\b", r"\bslick line\b"],
    "e-line": [r"\be-line\b", r"\beline\b", r"\belectric line\b", r"\be-line services\b"],
    "well logging": [r"\bwell logging\b", r"\bwell log(?:ging)?\b"],
    "open hole logging": [r"\bopen hole logging\b", r"\bopen-hole logging\b"],
    "cased hole logging": [r"\bcased hole logging\b", r"\bcased-hole logging\b"],
    "perforation": [r"\bperforation\b", r"\bperforating\b"],
    "memory gauge": [r"\bmemory gauge\b", r"\bmemory gauges\b"],
    "well intervention": [r"\bwell intervention\b"],
    "completion": [r"\bcompletion services?\b", r"\bwell completion\b"],
    "coiled tubing": [r"\bcoiled tubing\b", r"\bcoiled-tubing\b"],
    "well testing": [r"\bwell testing\b", r"\bwell test\b"],
    "mud logging": [r"\bmud logging\b"],
    "drilling fluids": [r"\bdrilling fluids?\b", r"\bmud engineering\b"],
    "cementing": [r"\bcementing\b", r"\bcementation\b"],
    "fishing": [r"\bfishing services?\b", r"\bdownhole fishing\b"],
    "downhole tools": [r"\bdownhole tools?\b"],
    "well stimulation": [r"\bwell stimulation\b", r"\bstimulation services?\b"],
    "acidizing": [r"\bacidizing\b", r"\bacidising\b", r"\bacid stimulation\b"],
    "fracturing": [r"\bfracturing\b", r"\bfracking\b", r"\bhydraulic fracturing\b"],
    "pipeline inspection": [r"\bpipeline inspection\b"],
    "ndt": [r"\bndt\b", r"\bnon[- ]destructive testing\b"],
    "asset integrity": [r"\basset integrity\b", r"\bintegrity management\b"],
    "corrosion monitoring": [r"\bcorrosion monitoring\b"],
    "instrumentation": [r"\binstrumentation\b"],
    "automation": [r"\bindustrial automation\b", r"\bprocess automation\b", r"\bautomation\b"],
    "scada": [r"\bscada\b"],
    "process control": [r"\bprocess control\b"],
    "offshore": [r"\boffshore\b"],
    "marine services": [r"\bmarine services?\b"],
    "diving services": [r"\bdiving services?\b"],
    "rov": [r"\brov\b", r"\bremotely operated vehicle\b"],
    "oilfield chemicals": [r"\boilfield chemicals?\b", r"\bproduction chemicals?\b", r"\bdrilling chemicals?\b"],
    "water treatment": [r"\bwater treatment\b"],
    "hse": [r"\bhse\b", r"\bhealth safety\b", r"\bprocess safety\b"],
    "fire and gas": [r"\bfire and gas\b", r"\bgas detection\b"],
    "esp": [r"\besp\b", r"\belectrical submersible pump\b", r"\belectric submersible pump\b"],
    "rod pump": [r"\brod pump\b", r"\bsucker rod pump\b"],
    "gas lift": [r"\bgas lift\b"],
    "virtual flow metering": [r"\bvirtual flow metering\b", r"\bvirtual flow meter\b", r"\bvirtual meter\b"],
    "well performance": [r"\bwell performance\b"],
    "artificial lift": [r"\bartificial lift\b"],
    "production optimization": [r"\bproduction optimization\b", r"\bproduction optimisation\b"],
    "well surveillance": [r"\bwell surveillance\b"],
    "multiphase metering": [r"\bmultiphase metering\b", r"\bmultiphase meter\b"],
    "flow assurance": [r"\bflow assurance\b"],
    "production monitoring": [r"\bproduction monitoring\b"],
    "reservoir simulation": [r"\breservoir simulation\b"],
    "reservoir modeling": [r"\breservoir modeling\b", r"\breservoir modelling\b"],
    "drilling": [r"\bdrilling\b"],
    "drilling optimization": [r"\bdrilling optimization\b", r"\bdrilling optimisation\b"],
    "production engineering": [r"\bproduction engineering\b"],
}

_GEO_EVIDENCE_HINTS = {
    "egypt": [r"\begypt\b", r"\begyptian\b", r"\bcairo\b", r"\balexandria\b", r"\bsuez\b", r"\bport said\b", r"\b6th of october\b", r"\bnew cairo\b", r"\+20\b", r"\.eg\b"],
    "united arab emirates": [r"\buae\b", r"\bunited arab emirates\b", r"\bdubai\b", r"\babudhabi\b", r"\babu dhabi\b", r"\bsharjah\b", r"\.ae\b"],
    "united kingdom": [r"\buk\b", r"\bu\.k\.\b", r"\bunited kingdom\b", r"\bengland\b", r"\bscotland\b", r"\bwales\b", r"\blondon\b", r"\.uk\b"],
    "united states": [r"\busa\b", r"\bu\.s\.a\.\b", r"\bunited states\b", r"\bhouston\b", r"\.us\b"],
    "norway": [r"\bnorway\b", r"\bnorwegian\b", r"\boslo\b", r"\bstavanger\b", r"\.no\b"],
    "saudi arabia": [r"\bsaudi arabia\b", r"\bsaudi\b", r"\bdammam\b", r"\bdhahran\b", r"\bal khobar\b", r"\.sa\b"],
    "qatar": [r"\bqatar\b", r"\bdoha\b", r"\.qa\b"],
    "oman": [r"\boman\b", r"\bmuscat\b", r"\.om\b"],
    "kuwait": [r"\bkuwait\b", r"\.kw\b"],
    "bahrain": [r"\bbahrain\b", r"\.bh\b"],
}

_SERVICE_RESULT_HINTS = [
    "service", "services", "contractor", "engineering", "inspection", "maintenance",
    "wireline", "slickline", "logging", "well intervention", "coiled tubing",
    "testing", "instrumentation", "automation", "pipeline", "integrity",
    "oilfield", "petroleum services", "production services", "drilling services",
]


def _extract_solution_keywords_from_prompt(prompt_lower: str) -> list[str]:
    found = []
    for label, patterns in _SOLUTION_KEYWORD_PATTERNS.items():
        if any(re.search(p, prompt_lower) for p in patterns):
            found.append(label)
    return found


def _extract_domain_keywords_from_prompt(prompt_lower: str) -> list[str]:
    found = []
    for label, patterns in _DOMAIN_KEYWORD_PATTERNS.items():
        if any(re.search(p, prompt_lower) for p in patterns):
            found.append(label)
    return found


def _extract_commercial_intent_from_prompt(prompt_lower: str) -> str:
    if re.search(r"\b(agent|agency|distributor|distribution|local representation|representative|representation)\b", prompt_lower):
        return "agent_or_distributor"
    if re.search(r"\b(reseller|resellers)\b", prompt_lower):
        return "reseller"
    if re.search(r"\b(partner|partners|channel partner|channel partners)\b", prompt_lower):
        return "partner"
    return "general"



def _postprocess_task_spec_from_prompt(task_spec, prompt: str):
    prompt_lower = _normalize_prompt_text(prompt)

    if getattr(task_spec, "task_type", "") in {
        "entity_discovery", "market_research", "similar_entity_expansion", "entity_enrichment"
    }:
        if any(hint in prompt_lower for hint in _DIGITAL_CATEGORY_HINTS):
            task_spec.target_category = "software_company"

    if not list(getattr(task_spec, "solution_keywords", []) or []):
        task_spec.solution_keywords = _extract_solution_keywords_from_prompt(prompt_lower)

    if not list(getattr(task_spec, "domain_keywords", []) or []):
        task_spec.domain_keywords = _extract_domain_keywords_from_prompt(prompt_lower)

    if getattr(task_spec, "commercial_intent", "general") == "general":
        task_spec.commercial_intent = _extract_commercial_intent_from_prompt(prompt_lower)

    geo = getattr(task_spec, "geography", None)
    if geo is not None:
        has_presence_outside = any(re.search(pat, prompt_lower) for pat in _PRESENCE_OUTSIDE_PATTERNS)
        if has_presence_outside and getattr(geo, "exclude_countries", None) and not getattr(geo, "exclude_presence_countries", None):
            geo.exclude_presence_countries = sorted(set(list(geo.exclude_countries or [])))
            geo.exclude_countries = [c for c in (geo.exclude_countries or []) if c not in set(geo.exclude_presence_countries or [])]

        geo.strict_mode = bool(
            list(getattr(geo, "include_countries", []) or [])
            or list(getattr(geo, "exclude_countries", []) or [])
            or list(getattr(geo, "exclude_presence_countries", []) or [])
        )

    industry = _clean_text(getattr(task_spec, "industry", ""))
    domain_keywords = list(getattr(task_spec, "domain_keywords", []) or [])
    if domain_keywords and (not industry or industry.lower() in {"oil and gas", "oil & gas", "energy", "petroleum"}):
        task_spec.industry = f"{', '.join(domain_keywords[:4])} in oil and gas"

    return task_spec

def _normalize_paper_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "Title": _first_nonempty(r, ["company_name", "title", "paper_title", "document_title", "name"]),
                "Authors": _first_nonempty(r, ["authors", "author", "paper_authors", "creators", "creator"]),
                "Year": _parse_year(_first_nonempty(r, [
                    "publication_year", "year", "published_year", "publication_date", "published_date", "date"
                ])),
                "Abstract": _first_nonempty(r, ["description", "abstract", "summary", "snippet", "paper_abstract"]),
                "URL": _first_nonempty(r, ["website", "url", "paper_url", "source_url", "link"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Title", "Abstract", "URL"])

    keep_cols = []
    for c in df.columns:
        nonempty = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
        if c in {"Title", "Abstract", "URL"} or (nonempty != "").any():
            keep_cols.append(c)
    return df[keep_cols].copy()


_BAD_HOST_HINTS = [
    "wikipedia.org", "facebook.com", "instagram.com", "youtube.com", "x.com",
    "twitter.com", "companiesmarketcap.com", "worldpopulationreview.com",
    "alamy.com", "shutterstock.com",
]
_BAD_PATH_HINTS = ["/news/", "/blog/", "/article/", "/articles/", "/wiki/"]
_BAD_TEXT_HINTS = [
    "top oil and gas companies", "largest oil and gas companies", "market cap",
    "stock photo", "directory", "list of companies", "ranking",
]
_DIGITAL_RESULT_HINTS = [
    "software", "platform", "saas", "analytics", "automation", "digital",
    "ai", "iot", "scada", "monitoring", "optimization", "data platform", "cloud",
]
_COMPANYISH_HINTS = [
    "company", "vendor", "provider", "technology", "solutions", "platform", "software", "services"
]



def _record_text(record: dict) -> str:
    parts = [
        record.get("company_name"), record.get("description"), record.get("summary"),
        record.get("snippet"), record.get("website"), record.get("url"),
        record.get("linkedin_url"), record.get("hq_country"), record.get("presence_countries"),
        record.get("country"), record.get("location"), record.get("city"), record.get("title"),
    ]
    return _normalize_prompt_text(" ".join(str(p) for p in parts if not _is_blank(p)))


def _country_patterns(country: str) -> list[str]:
    c = _normalize_prompt_text(country)
    if c in _GEO_EVIDENCE_HINTS:
        return _GEO_EVIDENCE_HINTS[c]
    return [rf"\b{re.escape(c)}\b"] if c else []


def _record_matches_any_country(record: dict, countries: list[str]) -> bool:
    if not countries:
        return True
    text = _record_text(record)
    if not text:
        return False
    for country in countries:
        for pat in _country_patterns(country):
            if re.search(pat, text, re.IGNORECASE):
                return True
    return False


def _record_matches_excluded_country(record: dict, countries: list[str]) -> bool:
    if not countries:
        return False
    text = _record_text(record)
    if not text:
        return False
    for country in countries:
        for pat in _country_patterns(country):
            if re.search(pat, text, re.IGNORECASE):
                return True
    return False


def _requires_geo_evidence(task_meta: dict) -> bool:
    include_countries = list(task_meta.get("include_countries") or [])
    strict_mode = bool(task_meta.get("strict_mode"))
    if not strict_mode or not include_countries:
        return False
    normalized = [_normalize_prompt_text(c) for c in include_countries]
    if "egypt" in normalized:
        return True
    return len(include_countries) <= 2


def _is_valid_company_record(record: dict, task_meta: dict) -> bool:
    name = _clean_text(record.get("company_name"))
    website = _clean_text(record.get("website") or record.get("url"))
    text = _record_text(record)
    target_category = (task_meta.get("target_category") or "general").strip()

    if not name and not website:
        return False

    if website:
        parsed = urlparse(website if "://" in website else f"https://{website}")
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if any(bad in host for bad in _BAD_HOST_HINTS):
            return False
        if any(bad in path for bad in _BAD_PATH_HINTS):
            return False

    if any(bad in text for bad in _BAD_TEXT_HINTS):
        return False

    if len(name.split()) > 10 and any(bad in text for bad in ["top", "best", "largest", "ranking", "list"]):
        return False

    if target_category == "software_company":
        digital_ok = any(h in text for h in _DIGITAL_RESULT_HINTS)
        companyish_ok = bool(website) or any(h in text for h in _COMPANYISH_HINTS)
        if not (digital_ok and companyish_ok):
            return False

    if target_category == "service_company":
        serviceish_ok = any(h in text for h in _SERVICE_RESULT_HINTS)
        companyish_ok = bool(website) or any(h in text for h in _COMPANYISH_HINTS)
        if not (serviceish_ok and companyish_ok):
            return False

    include_countries = list(task_meta.get("include_countries") or [])
    exclude_countries = list(task_meta.get("exclude_countries") or [])
    exclude_presence_countries = list(task_meta.get("exclude_presence_countries") or [])

    if _requires_geo_evidence(task_meta):
        if not _record_matches_any_country(record, include_countries):
            return False

    if exclude_presence_countries and _record_matches_excluded_country(record, exclude_presence_countries):
        return False

    if exclude_countries and _record_matches_excluded_country(record, exclude_countries):
        return False

    return True

def _refine_company_records(records: list[dict], task_meta: dict) -> tuple[list[dict], int]:
    if task_meta.get("task_type") in {"document_research", "people_search"}:
        return records or [], 0

    refined, removed = [], 0
    for record in records or []:
        if _is_valid_company_record(record, task_meta):
            refined.append(record)
        else:
            removed += 1

    # Critical: do NOT fall back to the original records when strict geography
    # or strict exclusion filters are active. Falling back reintroduces generic
    # global hits and defeats the whole Egypt-only / exclude-country intent.
    geo_strict = bool(task_meta.get("strict_mode")) or bool(task_meta.get("include_countries")) or bool(task_meta.get("exclude_presence_countries")) or bool(task_meta.get("exclude_countries"))
    if geo_strict:
        return refined, removed

    # For fully global searches only, keep the old behavior so the UI does not
    # go blank when the lightweight app-side validator is too strict.
    if not refined:
        return records or [], 0
    return refined, removed


def _df_to_pdf_bytes(df: pd.DataFrame, title: str = "Results", is_papers: bool = False) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return None

    if df is None or df.empty:
        return None

    def _pdf_escape(text: str) -> str:
        return str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _clean_cell(value, max_len: int = 1200) -> str:
        text = _clean_text(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return text

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    meta_style = ParagraphStyle(
        "Meta", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=4,
        alignment=TA_LEFT, textColor=colors.HexColor("#333333")
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=8, alignment=TA_LEFT
    )
    link_style = ParagraphStyle(
        "Link", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=10,
        alignment=TA_LEFT, textColor=colors.HexColor("#1a4f7a")
    )

    if is_papers or {"Title", "Abstract"}.issubset(set(df.columns)):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = [Paragraph(_pdf_escape(title), title_style), Spacer(1, 10), Paragraph(f"{len(df)} results", meta_style), Spacer(1, 12)]

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            paper_title = _clean_cell(row.get("Title", ""), 300)
            authors = _clean_cell(row.get("Authors", ""), 250) if "Authors" in df.columns else ""
            year = _clean_cell(row.get("Year", ""), 20) if "Year" in df.columns else ""
            abstract = _clean_cell(row.get("Abstract", ""), 2500)
            url = _clean_cell(row.get("URL", ""), 400) if "URL" in df.columns else ""

            if not paper_title and not abstract and not url:
                continue

            story.append(Paragraph(f"{i}. {_pdf_escape(paper_title or 'Untitled')}", heading_style))
            meta_parts = []
            if authors:
                meta_parts.append(f"<b>Authors:</b> {_pdf_escape(authors)}")
            if year:
                meta_parts.append(f"<b>Year:</b> {_pdf_escape(year)}")
            if meta_parts:
                story.append(Paragraph(" &nbsp;&nbsp; ".join(meta_parts), meta_style))
            if abstract:
                story.append(Paragraph(f"<b>Abstract:</b> {_pdf_escape(abstract)}", body_style))
            if url:
                story.append(Paragraph(f"<b>URL:</b> {_pdf_escape(url)}", link_style))
            story.append(Spacer(1, 10))
            if i < len(df) and i % 3 == 0:
                story.append(PageBreak())

        doc.build(story)
        return buf.getvalue()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    story = [Paragraph(_pdf_escape(title), title_style), Spacer(1, 10), Paragraph(f"{len(df)} results", meta_style), Spacer(1, 10)]

    pdf_df = df.copy()
    table_header_style = ParagraphStyle("TableHeader", parent=styles["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT)
    table_cell_style = ParagraphStyle("TableCell", parent=styles["BodyText"], fontSize=7, leading=9, alignment=TA_LEFT)

    wrapped_header = [Paragraph(f"<b>{_pdf_escape(col)}</b>", table_header_style) for col in pdf_df.columns]
    wrapped_rows = []
    for _, row in pdf_df.iterrows():
        wrapped_rows.append([Paragraph(_pdf_escape(_clean_cell(row[col], 500)), table_cell_style) for col in pdf_df.columns])

    table_data = [wrapped_header] + wrapped_rows[:200]
    ncols = len(pdf_df.columns)
    if ncols <= 3:
        col_widths = [180, 280, 220][:ncols]
    elif ncols == 4:
        col_widths = [140, 180, 240, 160]
    else:
        usable_width = 780
        col_widths = [usable_width / ncols] * ncols

    table = Table(table_data, repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    if len(pdf_df) > 200:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Note: PDF includes first 200 rows only.", meta_style))
    doc.build(story)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">🔑 Your API Keys</p>', unsafe_allow_html=True)
    st.markdown(
        "The agent is free to use. It needs API keys to search and reason. "
        "**One free key is enough to start** — Groq takes 30 seconds to get."
    )

    with st.expander("⚡ Groq Key — Free, most important", expanded=True):
        groq_key = st.text_input("Groq API Key", value=_secret("GROQ_API_KEY"), type="password", placeholder="gsk_...", label_visibility="collapsed")
        st.success(f"✅ Groq connected  ({_mask(groq_key)})") if groq_key else _show_key_guide("groq")

    with st.expander("🧠 Gemini Key — Free backup LLM", expanded=False):
        gemini_key = st.text_input("Gemini Key", value=_secret("GEMINI_API_KEY"), type="password", placeholder="AIza...", label_visibility="collapsed")
        st.success(f"✅ Gemini connected  ({_mask(gemini_key)})") if gemini_key else _show_key_guide("gemini")

    st.markdown('<p class="section-header">🔭 Search Provider Keys</p>', unsafe_allow_html=True)
    st.caption("Optional but strongly recommended. Each is free with generous quotas.")

    with st.expander("🔭 Exa — semantic search + LinkedIn profiles", expanded=False):
        exa_key = st.text_input("Exa Key", value=_secret("EXA_API_KEY"), type="password", placeholder="your-exa-key", label_visibility="collapsed")
        st.success(f"✅ Exa connected  ({_mask(exa_key)})") if exa_key else _show_key_guide("exa")

    with st.expander("🌐 Tavily — question-style search", expanded=False):
        tavily_key = st.text_input("Tavily Key", value=_secret("TAVILY_API_KEY"), type="password", placeholder="tvly-...", label_visibility="collapsed")
        st.success(f"✅ Tavily connected  ({_mask(tavily_key)})") if tavily_key else _show_key_guide("tavily")

    with st.expander("🎯 SerpApi — LinkedIn + Google search", expanded=False):
        serpapi_key = st.text_input("SerpApi Key", value=_secret("SERPAPI_KEY"), type="password", placeholder="your-serpapi-key", label_visibility="collapsed")
        st.success(f"✅ SerpApi connected  ({_mask(serpapi_key)})") if serpapi_key else _show_key_guide("serpapi")

    anthropic_key = _secret("ANTHROPIC_API_KEY")
    openai_key = _secret("OPENAI_API_KEY")
    openrouter_key = _secret("OPENROUTER_API_KEY")
    firecrawl_key = _secret("FIRECRAWL_API_KEY")

    active_keys = []
    if groq_key:
        active_keys.append("Groq")
    if gemini_key:
        active_keys.append("Gemini")
    if exa_key:
        active_keys.append("Exa")
    if tavily_key:
        active_keys.append("Tavily")
    if serpapi_key:
        active_keys.append("SerpApi")
    if anthropic_key:
        active_keys.append("Claude")
    if openai_key:
        active_keys.append("GPT")

    if active_keys:
        st.success(f"**Active:** {' · '.join(active_keys)}")
    else:
        st.error("⚠️ No keys yet — add at least a free Groq key above.")

    st.divider()
    st.markdown('<p class="section-header">⚙️ Search Options</p>', unsafe_allow_html=True)
    mode_label = st.radio("Search depth", list(MODE_PRESETS.keys()), index=1, help="Balanced is recommended for most searches.")
    preset = MODE_PRESETS[mode_label]
    mode = preset["mode"]
    st.caption(preset["desc"])
    max_results = st.slider("Max results", 5, 150, value=preset["max"], step=5, help="How many results you want. More = slower search.")

    st.divider()
    st.markdown('<p class="section-header">📂 Avoid Duplicates</p>', unsafe_allow_html=True)
    st.caption("Upload a previous results file — new results won't repeat those.")
    uploaded_file = st.file_uploader("Previous results (CSV or Excel)", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded_file:
        st.success(f"✅ Dedup file loaded: {uploaded_file.name}")

    st.divider()
    with st.expander("🛠️ Advanced options", expanded=False):
        min_conf = st.slider("Min confidence score", 0, 100, 35, step=5, help="Lower = more results but less accurate.")
        export_fmt = st.selectbox("Export format", ["Auto-detect from prompt", "Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)"])
        export_filename = st.text_input("Export filename", value="results")
        st.caption("**Fields to collect:**")
        col1, col2 = st.columns(2)
        field_website = col1.checkbox("Website", value=True)
        field_email = col1.checkbox("Email", value=True)
        field_phone = col1.checkbox("Phone", value=True)
        field_linkedin = col2.checkbox("LinkedIn URL", value=False)
        field_hq = col2.checkbox("HQ Country", value=False)
        field_summary = col2.checkbox("Summary", value=False)


# ──────────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────────
llm_client = FreeLLMClient(
    groq_api_key=groq_key,
    gemini_api_key=gemini_key,
    openrouter_api_key=openrouter_key,
    anthropic_api_key=anthropic_key,
    openai_api_key=openai_key,
)
backends = llm_client.available_backends()


# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🔍 AI Research Agent</h1>"
    "<p style='color:#7b8099; font-size:15px; margin-top:4px; font-family:IBM Plex Sans'>"
    "Find companies, research papers, or LinkedIn profiles — just describe what you want in plain English."
    "</p>",
    unsafe_allow_html=True,
)

with st.expander("💡 Example searches", expanded=False):
    for cat, ex in EXAMPLES:
        c1, c2 = st.columns([1, 8])
        c1.markdown(f"<span class='tag'>{cat}</span>", unsafe_allow_html=True)
        if c2.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["prompt_value"] = ex
            st.rerun()

prompt = st.text_area(
    "What do you want to research?",
    value=st.session_state.get("prompt_value", ""),
    height=100,
    placeholder=(
        "• Find oilfield service companies in Egypt and Saudi Arabia with email and phone\n"
        "• Find LinkedIn profiles of petroleum engineers in oil gas companies in Egypt\n"
        "• Find research papers about ESP electrical submersible pump with authors export as PDF"
    ),
    key="main_prompt",
)
st.session_state["prompt_value"] = prompt

_has_llm = bool(groq_key or gemini_key or anthropic_key or openai_key or openrouter_key)
_has_search = bool(exa_key or tavily_key or serpapi_key)
if not _has_llm and not _has_search:
    st.warning("⚠️ No API keys set — results will be limited (DDG-only, no ranking). Add a free Groq key in the sidebar for much better results.")

run_btn = st.button("🚀 Run Search", type="primary", use_container_width=True, disabled=not prompt.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Run search
# ──────────────────────────────────────────────────────────────────────────────
if run_btn and prompt.strip():
    st.session_state["search_result"] = None
    st.session_state["task_meta"] = None
    st.session_state["paper_summaries"] = []
    st.session_state["summaries_topic"] = ""
    st.session_state["active_section"] = "results"

    clean_prompt = _dedupe_repeated_prompt(prompt)

    with st.spinner("🧠 Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(clean_prompt, llm=llm_client)

    task_spec = _postprocess_task_spec_from_prompt(task_spec, clean_prompt)
    task_spec.mode = mode
    task_spec.max_results = int(max_results)

    fmt_map = {
        "Auto-detect from prompt": None,
        "Excel (.xlsx)": "xlsx",
        "CSV (.csv)": "csv",
        "PDF (.pdf)": "pdf",
    }
    if fmt_map.get(export_fmt):
        task_spec.output.format = fmt_map[export_fmt]
    task_spec.output.filename = _normalize_filename(export_filename, task_spec.output.format)

    user_fields = []
    if field_website:
        user_fields.append("website")
    if field_email:
        user_fields.append("email")
    if field_phone:
        user_fields.append("phone")
    if field_linkedin:
        user_fields.append("linkedin")
    if field_hq:
        user_fields.append("hq_country")
    if field_summary:
        user_fields.append("summary")

    base_attrs = list(dict.fromkeys(list(getattr(task_spec, "target_attributes", []) or [])))

    entity_type = (getattr(task_spec, "target_entity_types", ["company"]) or ["company"])[0]
    if task_spec.task_type == "document_research":
        task_spec.target_attributes = sorted(set([
            "website", "summary", "author", "authors", "year",
            "publication_year", "published_date", "abstract"
        ] + user_fields + base_attrs))
    elif task_spec.task_type == "people_search":
        task_spec.target_attributes = list(dict.fromkeys(["linkedin", "website"] + user_fields + base_attrs))
    elif entity_type == "tender":
        task_spec.target_attributes = list(dict.fromkeys(base_attrs + (["website"] if not user_fields else []) + user_fields))
    else:
        task_spec.target_attributes = list(dict.fromkeys(base_attrs + (["website"] if not user_fields else []) + user_fields))

    if not task_spec.industry or len(task_spec.industry.strip()) < 2:
        st.error(
            "⚠️ Couldn't detect the topic. Try being more specific: "
            "'Find oil and gas service companies...' or 'Find research papers about ESP...'"
        )
        st.stop()

    geo = task_spec.geography
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"**Looking for:** {_humanize_entity_type((getattr(task_spec, 'target_entity_types', ['company']) or ['company'])[0])}")
        c2.info(f"**Category:** {_humanize_target_category(getattr(task_spec, 'target_category', 'general'))}")
        c3.info(f"**Industry:** {task_spec.industry}")
        c4.info(f"**Target:** {max_results} results in {mode} mode")

        if getattr(task_spec, "solution_keywords", None):
            st.info(f"🧩 Solution keywords: {', '.join(task_spec.solution_keywords)}")
        if getattr(task_spec, "domain_keywords", None):
            st.info(f"🛢️ Domain keywords: {', '.join(task_spec.domain_keywords)}")
        if getattr(task_spec, "commercial_intent", "general") != "general":
            st.info(f"🤝 Commercial intent: {_humanize_commercial_intent(task_spec.commercial_intent)}")

        if geo.include_countries:
            st.info(f"🌍 Search in: {', '.join(_title_list(geo.include_countries))}")
        if geo.exclude_presence_countries:
            st.warning(f"🚫 Excluding presence in: {', '.join(_title_list(geo.exclude_presence_countries))}")
        elif geo.exclude_countries:
            st.warning(f"🚫 Excluding countries: {', '.join(_title_list(geo.exclude_countries))}")
        if not geo.include_countries and not geo.exclude_countries and not geo.exclude_presence_countries:
            st.info("🌐 No geography filter — searching globally")

    provider_settings = ProviderSettings(
        use_ddg=True,
        use_exa=bool(exa_key),
        use_tavily=bool(tavily_key),
        use_serpapi=bool(serpapi_key),
        use_firecrawl=bool(firecrawl_key),
        use_llm_parser=bool(backends),
        use_uploaded_seed_dedupe=uploaded_file is not None,
    )

    user_keys = {
        "groq_api_key": groq_key,
        "gemini_api_key": gemini_key,
        "openrouter_api_key": openrouter_key,
        "exa_api_key": exa_key,
        "tavily_api_key": tavily_key,
        "serpapi_key": serpapi_key,
        "firecrawl_api_key": firecrawl_key,
        "anthropic_api_key": anthropic_key,
        "openai_api_key": openai_key,
    }

    progress_bar = st.progress(0, text="Starting...")
    status_box = st.empty()
    STAGE_PCT = {"DDG": 20, "Exa": 45, "Tavily": 65, "SerpApi": 75, "dedup": 85, "LLM re-rank": 92, "Accepted": 98}

    def _progress(msg: str):
        status_box.caption(f"⏳ {msg}")
        for kw, pct in STAGE_PCT.items():
            if kw.lower() in msg.lower():
                progress_bar.progress(pct, text=f"{kw}...")
                break

    orchestrator = SearchOrchestrator()
    uploaded_df = _read_file(uploaded_file)

    with st.spinner(f"Searching ({mode} mode)..."):
        result = orchestrator.run_task(
            task_spec=task_spec,
            provider_settings=provider_settings,
            uploaded_df=uploaded_df,
            budget_overrides={},
            min_confidence_score=int(min_conf),
            user_keys=user_keys,
            progress_callback=_progress,
        )

    progress_bar.progress(100, text="Done!")
    status_box.empty()

    st.session_state["search_result"] = result
    st.session_state["task_meta"] = {
        "task_type": task_spec.task_type,
        "raw_prompt": clean_prompt,
        "industry": task_spec.industry,
        "target_category": getattr(task_spec, "target_category", "general"),
        "solution_keywords": list(getattr(task_spec, "solution_keywords", []) or []),
        "domain_keywords": list(getattr(task_spec, "domain_keywords", []) or []),
        "commercial_intent": getattr(task_spec, "commercial_intent", "general"),
        "mode": mode,
        "max_results": int(max_results),
        "include_countries": list(task_spec.geography.include_countries or []),
        "exclude_countries": list(task_spec.geography.exclude_countries or []),
        "exclude_presence_countries": list(task_spec.geography.exclude_presence_countries or []),
        "strict_mode": bool(task_spec.geography.strict_mode),
        "target_attributes": list(task_spec.target_attributes or []),
    }
    st.session_state["summaries_topic"] = task_spec.industry or "research"

    if result.get("total_found", 0) == 0:
        st.error("No results found.")
        hints = []
        if not exa_key:
            hints.append("add a free Exa key (exa.ai) in the sidebar")
        if not tavily_key:
            hints.append("add a free Tavily key (tavily.com)")
        if mode == "Fast":
            hints.append("switch to Balanced or Deep mode")
        hints.append("try lowering the confidence score to 25 in Advanced options")
        hints.append("rephrase your search to include the specific industry or topic")
        if hints:
            st.info("**To get results:**\n" + "\n".join(f"- {h}" for h in hints))
        st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Render results from session state
# ──────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("search_result")
task_meta = st.session_state.get("task_meta")

if result and task_meta:
    raw = result.get("raw_search_results", 0)
    rej = len(result.get("rejected_records", []))
    b = result.get("budget", {})
    all_records = result.get("records", []) or []
    is_people = task_meta["task_type"] == "people_search"
    is_papers = task_meta["task_type"] == "document_research"

    refined_records, app_removed = _refine_company_records(all_records, task_meta)
    records = all_records if (is_people or is_papers) else refined_records
    total = len(records)

    if (not is_people and not is_papers) and total == 0 and len(all_records) > 0:
        if task_meta.get("include_countries") or task_meta.get("exclude_presence_countries") or task_meta.get("exclude_countries"):
            st.warning(
                "All raw candidates were removed by the geography/vendor filter. "
                "This is expected when the providers return generic global pages instead of Egypt-specific company evidence. "
                "Use Balanced or Deep mode and enable Exa/Tavily/SerpApi for better Egypt-specific coverage."
            )

    st.markdown(
        f'<div style="background:#0d2137;border:1px solid #1a4f7a;border-radius:10px;padding:16px 24px;margin:12px 0">'
        f'<span style="font-size:20px;font-weight:600;color:#4fc3f7">✅ {total} results shown</span>'
        f'<span style="color:#7b8099;font-size:13px;margin-left:16px">'
        f'from {len(all_records)} accepted · {raw} raw · {rej} filtered out by backend · {b.get("total_search_calls_used", 0)} searches</span></div>',
        unsafe_allow_html=True,
    )
    if app_removed:
        st.caption(f"App-side refinement removed {app_removed} obvious non-company / non-vendor records from display and exports.")

    df = pd.DataFrame(records)

    if is_people:
        show_cols = ["company_name", "job_title", "employer_name", "city", "linkedin_url", "linkedin_profile"]
        col_rename = {
            "company_name": "Name",
            "job_title": "Job Title",
            "employer_name": "Company",
            "city": "Location",
            "linkedin_url": "LinkedIn URL",
            "linkedin_profile": "Profile Link",
        }
        show = [c for c in show_cols if c in df.columns]
        df_display = df[show].copy() if show else pd.DataFrame()
        df_display.columns = [col_rename.get(c, c) for c in show]
    elif is_papers:
        df_display = _normalize_paper_records(records)
    else:
        show_cols = ["company_name", "website"]
        if field_email:
            show_cols.append("email")
        if field_phone:
            show_cols.append("phone")
        if field_linkedin:
            show_cols.append("linkedin_url")
        if field_hq:
            show_cols.append("hq_country")
        if field_summary:
            show_cols.append("description")
        show_cols.append("confidence_score")

        col_rename = {
            "company_name": "Company",
            "website": "Website",
            "email": "Email",
            "phone": "Phone",
            "linkedin_url": "LinkedIn",
            "hq_country": "HQ Country",
            "description": "Description",
            "confidence_score": "Score",
        }
        show = [c for c in show_cols if c in df.columns]
        df_display = df[show].copy() if show else pd.DataFrame()
        df_display.columns = [col_rename.get(c, c) for c in show]
        if "Score" in df_display.columns:
            df_display = df_display.sort_values("Score", ascending=False)

    if is_papers:
        section_labels = {"results": f"📊 Results ({total})", "summaries": "📝 AI Summaries", "download": "⬇️ Download", "details": "🔍 Search Details"}
    else:
        section_labels = {"results": f"📊 Results ({total})", "download": "⬇️ Download", "details": "🔍 Search Details"}

    section_options = list(section_labels.keys())
    if st.session_state.get("active_section") not in section_options:
        st.session_state["active_section"] = section_options[0]

    selected_section = st.radio("View", options=section_options, horizontal=True, key="active_section", format_func=lambda x: section_labels[x])

    if selected_section == "results":
        if is_people:
            st.markdown(f"**{total} LinkedIn profiles found**")
            for _, row in df_display.head(total).iterrows():
                name = row.get("Name", "")
                title = row.get("Job Title", "")
                company = row.get("Company", "")
                loc = row.get("Location", "")
                url = row.get("LinkedIn URL", "") or row.get("Profile Link", "")
                meta = " · ".join([p for p in [title, company, loc] if p])
                link_html = f'<a href="{url}" target="_blank" class="result-link">🔗 View Profile</a>' if url else ""
                st.markdown(
                    f'<div class="result-card"><div class="result-name">👤 {name}</div><div class="result-meta">{meta}</div>{link_html}</div>',
                    unsafe_allow_html=True,
                )
        elif is_papers:
            st.markdown(f"**{total} papers found**")
            for _, row in df_display.iterrows():
                title = row.get("Title", "")
                authors = row.get("Authors", "")
                year = row.get("Year", "")
                url = row.get("URL", "")
                abstract = str(row.get("Abstract", ""))[:250]
                meta_parts = [p for p in [str(authors)[:120], str(year)] if p and p != "nan"]
                meta = " · ".join(meta_parts)
                link_html = f'<a href="{url}" target="_blank" class="result-link">🔗 View Paper</a>' if url else ""
                st.markdown(
                    f'<div class="result-card"><div class="result-name">📄 {title}</div><div class="result-meta">{meta}</div><div style="font-size:12px;color:#5a6080;margin:6px 0">{abstract}{"..." if len(abstract) == 250 else ""}</div>{link_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.dataframe(
                df_display,
                use_container_width=True,
                height=min(600, 80 + len(df_display) * 40),
                column_config={
                    "Website": st.column_config.LinkColumn("Website"),
                    "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
                    "Email": st.column_config.TextColumn("Email"),
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                },
            )

    elif selected_section == "summaries":
        st.markdown("### 📝 Paper Summaries")
        st.caption("One plain-English summary per paper — what it found and why it matters.")
        if not backends:
            st.warning("⚠️ Add a free Groq or Gemini key in the sidebar to enable summaries.")
        else:
            summary_candidates: list[CompanyRecord] = []
            for r in records:
                description = _first_nonempty(r, ["description", "abstract", "summary", "snippet", "paper_abstract"])
                if not description.strip():
                    continue
                summary_candidates.append(
                    CompanyRecord(
                        company_name=_first_nonempty(r, ["company_name", "title", "paper_title", "document_title", "name"]) or "Untitled",
                        authors=_first_nonempty(r, ["authors", "author", "paper_authors", "creators", "creator"]),
                        description=description,
                        doi=_first_nonempty(r, ["doi"]),
                        website=_first_nonempty(r, ["website", "url", "paper_url", "source_url", "link"]),
                    )
                )
            summary_topic = task_meta.get("industry") or "the topic"
            st.info(f"**{len(summary_candidates)} papers** ready — click to generate summaries.")
            if st.button("🤖 Generate Summaries", key="gen_summaries", type="primary"):
                status_holder = st.empty()

                def _summary_progress(msg: str):
                    status_holder.caption(f"⏳ {msg}")

                summaries = summarize_papers(
                    papers=summary_candidates,
                    topic=summary_topic,
                    llm=llm_client,
                    max_papers=20,
                    progress_callback=_summary_progress,
                )
                st.session_state["paper_summaries"] = summaries
                st.session_state["summaries_topic"] = summary_topic
                status_holder.empty()
                st.success(f"✅ {len(summaries)} summaries ready — see below and download.")
                st.rerun()

        if st.session_state.get("paper_summaries"):
            summaries = st.session_state["paper_summaries"]
            topic = st.session_state.get("summaries_topic", "research")
            for i, s in enumerate(summaries, 1):
                with st.expander(f"📄 {i}. {s['title'][:75]}", expanded=(i == 1)):
                    meta = []
                    if s.get("authors"):
                        meta.append(f"👤 {s['authors']}")
                    if s.get("doi"):
                        meta.append(f"🔗 {s['doi'][:80]}")
                    if meta:
                        st.caption("  ·  ".join(meta))
                    st.markdown(s["summary"])

            txt_bytes = summaries_to_text(summaries, topic).encode("utf-8")
            pdf_bytes = summaries_to_pdf_bytes(summaries, topic)
            fn = topic[:30].replace(" ", "_")
            st.divider()
            st.markdown("**⬇️ Download all summaries:**")
            c1, c2 = st.columns(2)
            if pdf_bytes:
                c1.download_button("📑 PDF", data=pdf_bytes, file_name=f"summaries_{fn}.pdf", mime="application/pdf", use_container_width=True)
            else:
                c1.warning("Install reportlab to enable PDF summaries.")
            c2.download_button("📄 Text (.txt)", data=txt_bytes, file_name=f"summaries_{fn}.txt", mime="text/plain", use_container_width=True)

    elif selected_section == "download":
        st.markdown("### ⬇️ Download your results")
        st.caption("Downloads are generated from the cleaned table shown in the app.")
        try:
            xl_buf = io.BytesIO()
            df_display.to_excel(xl_buf, index=False, engine="openpyxl")
            xl_bytes = xl_buf.getvalue()
        except Exception:
            xl_bytes = None
        csv_bytes = df_display.to_csv(index=False).encode("utf-8")
        pdf_bytes = _df_to_pdf_bytes(df_display, title=f"Search Results - {task_meta.get('industry', 'Results')}", is_papers=is_papers)
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        if xl_bytes:
            dl_col1.download_button(
                "📊 Excel (.xlsx)",
                data=xl_bytes,
                file_name=_normalize_filename(export_filename, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        dl_col2.download_button("📄 CSV (.csv)", data=csv_bytes, file_name=_normalize_filename(export_filename, "csv"), mime="text/csv", use_container_width=True)
        if pdf_bytes:
            dl_col3.download_button("📑 PDF", data=pdf_bytes, file_name=_normalize_filename(export_filename, "pdf"), mime="application/pdf", use_container_width=True)
        else:
            dl_col3.warning("Install reportlab to enable PDF export.")
        st.divider()
        st.dataframe(df_display.head(5), use_container_width=True)
        st.caption(f"Preview: first 5 of {len(df_display)} rows.")

    elif selected_section == "details":
        st.markdown("### 🔍 Search details")
        st.caption("Technical details about how the search ran.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Results shown", total)
        c2.metric("Accepted before app refine", len(all_records))
        c3.metric("Raw candidates", raw)
        c4.metric("Searches made", b.get("total_search_calls_used", 0))

        if backends:
            st.info(f"🤖 AI backends used: {', '.join(backends)}")
        else:
            st.warning("No LLM backend — running without AI ranking. Add a Groq or Gemini key for better results.")

        with st.expander("What the agent understood from your prompt", expanded=True):
            st.markdown(f"- **Task type:** {_humanize_task_type(task_meta.get('task_type', ''))}")
            st.markdown(f"- **Category:** {_humanize_target_category(task_meta.get('target_category', 'general'))}")
            st.markdown(f"- **Topic / industry:** {task_meta.get('industry', '')}")
            if task_meta.get("solution_keywords"):
                st.markdown(f"- **Solution keywords:** {', '.join(task_meta['solution_keywords'])}")
            if task_meta.get("domain_keywords"):
                st.markdown(f"- **Domain keywords:** {', '.join(task_meta['domain_keywords'])}")
            st.markdown(f"- **Commercial intent:** {_humanize_commercial_intent(task_meta.get('commercial_intent', 'general'))}")
            st.markdown(f"- **Mode:** {task_meta.get('mode', '')}")
            if task_meta.get("include_countries"):
                st.markdown(f"- **Search in:** {', '.join(_title_list(task_meta['include_countries']))}")
            if task_meta.get("exclude_countries"):
                st.markdown(f"- **Excluded countries:** {', '.join(_title_list(task_meta['exclude_countries']))}")
            if task_meta.get("exclude_presence_countries"):
                st.markdown(f"- **Excluded presence countries:** {', '.join(_title_list(task_meta['exclude_presence_countries']))}")
            if task_meta.get("target_attributes"):
                st.markdown(f"- **Requested fields:** {', '.join(task_meta['target_attributes'])}")
            if task_meta.get("raw_prompt"):
                st.code(task_meta["raw_prompt"], language="text")
            if app_removed:
                st.markdown(f"- **App-side refinement removed:** {app_removed} obvious non-company / non-vendor records")

        with st.expander("Providers used"):
            provider_used = []
            if b.get("ddg_calls_used", 0) > 0:
                provider_used.append(f"DuckDuckGo ({b['ddg_calls_used']} calls)")
            if b.get("exa_calls_used", 0) > 0:
                provider_used.append(f"Exa ({b['exa_calls_used']} calls)")
            if b.get("tavily_calls_used", 0) > 0:
                provider_used.append(f"Tavily ({b['tavily_calls_used']} calls)")
            if b.get("serpapi_calls_used", 0) > 0:
                provider_used.append(f"SerpApi ({b['serpapi_calls_used']} calls)")
            for p in provider_used:
                st.markdown(f"- {p}")
            if not provider_used:
                st.caption("No providers ran.")


st.divider()
st.markdown(
    '<p style="text-align:center;color:#3d4363;font-size:12px;font-family:IBM Plex Mono">'
    'AI Research Agent · Powered by free APIs · No data stored'
    '</p>',
    unsafe_allow_html=True,
)