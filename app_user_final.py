from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings
from pipeline.orchestrator import SearchOrchestrator


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Navigator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(180deg, #070b14 0%, #0b1020 100%);
    color: #e7ecf5;
}
.block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1180px;}
.hero-panel { background: transparent; border: none; padding: 0.25rem 0 0.55rem 0; margin-bottom: 0.25rem; }
.hero-title {
    font-size: 2.35rem; font-weight: 800; letter-spacing: -0.035em;
    color: #f8fafc; margin-bottom: 0.2rem; line-height: 1.08;
}
.hero-subtitle {
    color: #b3bfd4; font-size: 1.04rem; margin-bottom: 0.1rem; line-height: 1.5;
}
.search-shell {
    background: rgba(15, 22, 38, 0.92);
    border: 1px solid rgba(124, 143, 179, 0.18);
    border-radius: 20px;
    padding: 1.05rem 1.05rem 0.9rem 1.05rem;
    box-shadow: 0 18px 44px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}
.summary-card {
    background: rgba(15, 22, 38, 0.9);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.metric-chip {
    display: inline-block; padding: 0.28rem 0.62rem; border-radius: 999px;
    background: rgba(83, 113, 255, 0.14); border: 1px solid rgba(83, 113, 255, 0.22);
    color: #dbe6ff; font-size: 0.82rem; margin: 0 0.45rem 0.45rem 0;
}
.result-card {
    background: rgba(15, 22, 38, 0.88);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.85rem;
}
.result-card:hover { border-color: rgba(106, 154, 255, 0.42); }
.result-title { color: #f7f9fc; font-size: 1.02rem; font-weight: 600; margin-bottom: 0.35rem; }
.result-meta { color: #a8b3c7; font-size: 0.88rem; margin-bottom: 0.45rem; }
.result-desc { color: #d8dfeb; font-size: 0.92rem; line-height: 1.5; }
.badge {
    display: inline-block; padding: 0.2rem 0.52rem; border-radius: 999px; font-size: 0.74rem;
    border: 1px solid rgba(124, 143, 179, 0.2); color: #d6ddea; background: rgba(255,255,255,0.03);
    margin-right: 0.35rem; margin-bottom: 0.35rem;
}
.empty-box {
    background: rgba(15, 22, 38, 0.88);
    border: 1px dashed rgba(124, 143, 179, 0.22);
    border-radius: 16px; padding: 1.25rem; color: #c4cede;
}
.compact-section { margin-top: 0.75rem; }
.info-strip {
    background: linear-gradient(135deg, rgba(25,34,58,0.92) 0%, rgba(15,22,38,0.92) 100%);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
}
.info-strip-title { color: #f8fafc; font-weight: 600; margin-bottom: 0.18rem; }
.info-strip-text { color: #aeb9cb; font-size: 0.9rem; line-height: 1.45; }
.key-status {
    color: #d7eadc;
    background: rgba(41, 89, 63, 0.22);
    border: 1px solid rgba(91, 166, 116, 0.18);
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
    font-size: 0.88rem;
    margin-top: 0.6rem;
}
.small-caption { color: #8f9bb0; font-size: 0.84rem; }
.mode-note { color: #aeb9cb; font-size: 0.85rem; margin-top: 0.45rem; }
.suggest-line {
    color: #aeb9cb;
    font-size: 0.89rem;
    margin: 0.55rem 0 0.1rem 0;
    line-height: 1.45;
}
/* better sidebar mode buttons */
div[data-testid="stSidebar"] button[kind="secondary"] {
    white-space: nowrap !important;
}
@media (max-width: 900px) {
    .hero-title { font-size: 2rem; }
}
</style>
<script>
window.addEventListener('load', function() {
  const setSpell = () => {
    const areas = window.parent.document.querySelectorAll('textarea');
    areas.forEach(a => a.setAttribute('spellcheck', 'true'));
  };
  setSpell();
  setTimeout(setSpell, 800);
});
</script>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
SUGGEST_TERMS = [
    "software", "companies", "service", "engineering", "academic", "papers",
    "linkedin", "accounts", "profiles", "tenders", "exhibitors", "products",
    "germany", "egypt", "saudi arabia", "norway", "food manufacturing",
    "electrical submersible pump", "wireline", "slickline", "well logging",
]


def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


def _resolve_key(entered: str, secret_name: str) -> str:
    return (entered or "").strip() or _secret(secret_name)


def _normalize_filename(name: str, fmt: str) -> str:
    ext_map = {"xlsx": ".xlsx", "csv": ".csv", "json": ".json", "pdf": ".pdf"}
    base = (name or "results").strip() or "results"
    for ext in ext_map.values():
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base + ext_map.get(fmt, ".xlsx")


def _human_task(task_type: str) -> str:
    return {
        "entity_discovery": "Companies",
        "document_research": "Academic Papers",
        "people_search": "LinkedIn Profiles",
        "market_research": "Market Research",
        "entity_enrichment": "Enrichment",
        "similar_entity_expansion": "Similar Entities",
    }.get(task_type or "", "Results")


def _human_category(cat: str) -> str:
    return {
        "software_company": "Digital / software companies",
        "service_company": "Service / engineering companies",
        "general": "General results",
    }.get((cat or "general").strip(), (cat or "general").replace("_", " ").title())


def _fit_label(score: Any) -> str:
    try:
        s = float(score)
    except Exception:
        return "Review"
    if s >= 75:
        return "Strong match"
    if s >= 50:
        return "Good match"
    if s >= 30:
        return "Possible match"
    return "Review"


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _read_file(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        return None


def _render_search_summary(task_spec: dict):
    geo = task_spec.get("geography", {}) or {}
    inc = [str(x).title() for x in (geo.get("include_countries") or []) if str(x).strip()]
    exc = [str(x).title() for x in (geo.get("exclude_countries") or []) if str(x).strip()]
    excp = [str(x).title() for x in (geo.get("exclude_presence_countries") or []) if str(x).strip()]

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### Search summary")
    st.markdown(f'<span class="metric-chip">Type: {_human_task(task_spec.get("task_type", ""))}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="metric-chip">Category: {_human_category(task_spec.get("target_category", ""))}</span>', unsafe_allow_html=True)
    if _clean(task_spec.get("industry")):
        st.markdown(f'<span class="metric-chip">Focus: {_clean(task_spec.get("industry"))}</span>', unsafe_allow_html=True)
    if inc:
        st.markdown(f'<span class="metric-chip">Region: {", ".join(inc)}</span>', unsafe_allow_html=True)
    if exc:
        st.markdown(f'<span class="metric-chip">Exclude HQ: {", ".join(exc)}</span>', unsafe_allow_html=True)
    if excp:
        st.markdown(f'<span class="metric-chip">Exclude presence: {", ".join(excp)}</span>', unsafe_allow_html=True)

    requested = [str(x) for x in (task_spec.get("target_attributes") or []) if str(x).strip()]
    if requested:
        st.caption("Requested info: " + ", ".join(requested))
    st.markdown('</div>', unsafe_allow_html=True)


def _result_description(r: dict) -> str:
    for key in ["description", "summary", "snippet", "notes"]:
        val = _clean(r.get(key))
        if val:
            return val[:400]
    return ""


def _render_company_cards(records: list[dict]):
    for r in records:
        name = _clean(r.get("company_name")) or _clean(r.get("title")) or "Untitled"
        website = _clean(r.get("website")) or _clean(r.get("source_url"))
        email = _clean(r.get("email"))
        phone = _clean(r.get("phone"))
        country = _clean(r.get("hq_country")) or _clean(r.get("country"))
        provider = _clean(r.get("source_provider"))
        fit = _fit_label(r.get("confidence_score"))
        desc = _result_description(r)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{name}</div>', unsafe_allow_html=True)
        meta = []
        if country:
            meta.append(country.title())
        if provider:
            meta.append(provider.upper())
        if meta:
            st.markdown(f'<div class="result-meta">{" • ".join(meta)}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge">{fit}</span>', unsafe_allow_html=True)
        if email:
            st.markdown('<span class="badge">Email available</span>', unsafe_allow_html=True)
        if phone:
            st.markdown('<span class="badge">Phone available</span>', unsafe_allow_html=True)
        if website:
            st.markdown(f"<div class='compact-section'><a href='{website}' target='_blank'>Open source</a></div>", unsafe_allow_html=True)
        if desc:
            st.markdown(f'<div class="result-desc">{desc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_paper_table(records: list[dict]):
    rows = []
    for r in records:
        rows.append({
            "Title": _clean(r.get("company_name")) or _clean(r.get("title")),
            "Authors": _clean(r.get("authors")),
            "Year": _clean(r.get("publication_year")),
            "DOI": _clean(r.get("doi")),
            "Abstract": _clean(r.get("description"))[:500],
            "Source": _clean(r.get("website")) or _clean(r.get("source_url")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=520)


def _render_people_cards(records: list[dict]):
    for r in records:
        name = _clean(r.get("company_name")) or "Unnamed profile"
        title = _clean(r.get("job_title"))
        employer = _clean(r.get("employer_name"))
        location = _clean(r.get("city")) or _clean(r.get("country"))
        profile = _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url")) or _clean(r.get("website"))
        desc = _result_description(r)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{name}</div>', unsafe_allow_html=True)
        meta = [x for x in [title, employer, location] if x]
        if meta:
            st.markdown(f'<div class="result-meta">{" • ".join(meta)}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge">{_fit_label(r.get("confidence_score"))}</span>', unsafe_allow_html=True)
        if profile:
            st.markdown(f"<div class='compact-section'><a href='{profile}' target='_blank'>Open profile</a></div>", unsafe_allow_html=True)
        if desc:
            st.markdown(f'<div class="result-desc">{desc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _connected_integrations(keys: dict[str, str]) -> list[str]:
    labels = {
        "groq": "Groq",
        "gemini": "Gemini",
        "exa": "Exa",
        "tavily": "Tavily",
        "serpapi": "SerpApi",
    }
    return [labels[k] for k, v in keys.items() if str(v).strip()]


def _typing_hints(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if len(cleaned) < 4:
        return []
    tokens = re.findall(r"[A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF\- ]+", cleaned.lower())
    hints: list[str] = []
    for token in tokens[-6:]:
        tok = token.strip()
        if len(tok) < 4:
            continue
        match = process.extractOne(tok, SUGGEST_TERMS, scorer=fuzz.WRatio)
        if match and match[1] >= 88 and match[0].lower() != tok.lower():
            hints.append(f"{tok} → {match[0]}")
    # unique preserve
    out = []
    seen = set()
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:3]


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
mode_defaults = {"Fast": 15, "Balanced": 25, "Deep": 40}
mode_help = {
    "Fast": "Quick search",
    "Balanced": "Best overall",
    "Deep": "Broader search",
}

if "rn_mode" not in st.session_state:
    st.session_state["rn_mode"] = "Balanced"

with st.sidebar:
    st.markdown("## Search settings")
    st.caption("Choose a search mode and optional integrations.")

    st.markdown("**Search mode**")
    c1, c2, c3 = st.columns([1, 1.55, 1])
    if c1.button("Fast ?", help=mode_help["Fast"], use_container_width=True):
        st.session_state["rn_mode"] = "Fast"
    if c2.button("Balanced ?", help=mode_help["Balanced"], use_container_width=True):
        st.session_state["rn_mode"] = "Balanced"
    if c3.button("Deep ?", help=mode_help["Deep"], use_container_width=True):
        st.session_state["rn_mode"] = "Deep"

    mode = st.session_state["rn_mode"]
    max_results = mode_defaults[mode]
    st.markdown(f"<div class='mode-note'>Selected mode: <strong>{mode}</strong></div>", unsafe_allow_html=True)

    with st.expander("Optional integrations", expanded=False):
        st.markdown("**Optional keys for stronger search quality**")
        st.caption("Recommended setup: Groq or Gemini for interpretation, plus Exa or Tavily for broader coverage.")

        groq_key = st.text_input("Groq API key", value="", type="password", placeholder="Paste key (optional)")
        gemini_key = st.text_input("Gemini API key", value="", type="password", placeholder="Paste key (optional)")
        exa_key = st.text_input("Exa API key", value="", type="password", placeholder="Paste key (optional)")
        tavily_key = st.text_input("Tavily API key", value="", type="password", placeholder="Paste key (optional)")
        serpapi_key = st.text_input("SerpApi key", value="", type="password", placeholder="Paste key (optional)")

        connected = _connected_integrations({
            "groq": groq_key,
            "gemini": gemini_key,
            "exa": exa_key,
            "tavily": tavily_key,
            "serpapi": serpapi_key,
        })
        if connected:
            st.markdown(
                f"<div class='key-status'>Connected: {', '.join(connected)}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("How to get keys", expanded=False):
            st.markdown(
                """
**[Groq — API Keys](https://console.groq.com/keys)**  
1. Create a free account or sign in.  
2. Open the **API Keys** page.  
3. Click **Create API Key**.  
4. Copy the key and paste it here.

**[Gemini — Google AI Studio API Keys](https://aistudio.google.com/app/apikey)**  
1. Sign in with your Google account.  
2. Open the **API key** page in Google AI Studio.  
3. Create a key.  
4. Copy the key and paste it here.

**[Exa — Dashboard API Keys](https://dashboard.exa.ai/api-keys)**  
1. Create an account or sign in.  
2. Open your **API Keys** page.  
3. Click **Create Key**.  
4. Copy the key and paste it here.

**[Tavily — Platform](https://app.tavily.com/home)**  
1. Create a free account or sign in.  
2. Open your dashboard.  
3. Copy one of your API keys.  
4. Paste it here.

**[SerpApi — Sign in / Dashboard](https://serpapi.com/users/sign_in)**  
1. Create an account or sign in.  
2. Open your dashboard.  
3. Copy your API key.  
4. Paste it here.
                """
            )

    with st.expander("Advanced settings", expanded=False):
        export_format = st.selectbox("Export format", ["xlsx", "csv", "json", "pdf"], index=0)
        export_filename = st.text_input("Export filename", value="results")
        min_confidence = st.slider("Minimum relevance", 0, 100, 25 if mode == "Fast" else 35, 5)
        uploaded_file = st.file_uploader("Existing list for deduplication", type=["csv", "xlsx"])
        use_seed_dedupe = st.checkbox("Use uploaded file for deduplication", value=True)


# -----------------------------------------------------------------------------
# Header / Hero
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Research Navigator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Search companies, academic papers, LinkedIn accounts, tenders, exhibitors, and products in English or Arabic.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="info-strip">
  <div class="info-strip-title">Optional integrations can improve search quality</div>
  <div class="info-strip-text">For broader coverage and stronger interpretation, add Groq, Gemini, Exa, Tavily, or SerpApi from the <strong>Optional integrations</strong> section in the sidebar.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="search-shell">', unsafe_allow_html=True)
prompt = st.text_area(
    "What would you like to research?",
    value="",
    height=235,
    placeholder=(
        "Describe what you want to find in English or Arabic.\n\n"
        "Examples:\n"
        "• Find software companies in food manufacturing in Germany with website and email.\n"
        "• Find academic papers about electrical submersible pumps with authors and abstract.\n"
        "• Find LinkedIn accounts of petroleum engineers in Saudi Arabia.\n"
        "• ابحث عن شركات خدمات البترول في مصر مع الموقع الإلكتروني والإيميل."
    ),
)

hints = _typing_hints(prompt)
if hints:
    st.markdown(
        "<div class='suggest-line'><strong>Did you mean:</strong> " + " • ".join(hints) + "</div>",
        unsafe_allow_html=True,
    )

run_btn = st.button("Start search", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Run search
# -----------------------------------------------------------------------------
if run_btn:
    if not prompt.strip():
        st.stop()

    resolved_groq = _resolve_key(groq_key, "GROQ_API_KEY")
    resolved_gemini = _resolve_key(gemini_key, "GEMINI_API_KEY")
    resolved_exa = _resolve_key(exa_key, "EXA_API_KEY")
    resolved_tavily = _resolve_key(tavily_key, "TAVILY_API_KEY")
    resolved_serpapi = _resolve_key(serpapi_key, "SERPAPI_KEY")

    user_keys = {
        "groq_api_key": resolved_groq,
        "gemini_api_key": resolved_gemini,
        "exa_api_key": resolved_exa,
        "tavily_api_key": resolved_tavily,
        "serpapi_key": resolved_serpapi,
    }

    llm_client = FreeLLMClient(
        groq_api_key=resolved_groq,
        gemini_api_key=resolved_gemini,
    )

    with st.spinner("Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    task_spec.mode = mode
    task_spec.max_results = int(max_results)
    task_spec.output.format = export_format
    task_spec.output.filename = _normalize_filename(export_filename, export_format)
    if not task_spec.target_attributes:
        task_spec.target_attributes = ["website"]

    provider_settings = ProviderSettings(
        use_ddg=True,
        use_exa=bool(resolved_exa),
        use_tavily=bool(resolved_tavily),
        use_serpapi=bool(resolved_serpapi),
        use_firecrawl=False,
        use_llm_parser=bool(llm_client.available_backends()),
        use_uploaded_seed_dedupe=use_seed_dedupe,
    )
    uploaded_df = _read_file(uploaded_file)

    progress = st.empty()

    def _progress(msg: str):
        progress.caption(msg)

    with st.spinner("Searching..."):
        result = SearchOrchestrator().run_task(
            task_spec=task_spec,
            provider_settings=provider_settings,
            uploaded_df=uploaded_df,
            budget_overrides={},
            min_confidence_score=int(min_confidence),
            user_keys=user_keys,
            progress_callback=_progress,
        )
    progress.empty()

    task_meta = result.get("task_spec", {}) or {}
    records = result.get("records", []) or []
    raw_total = int(result.get("raw_search_results", 0) or 0)
    export_path = result.get("export_path", "")

    _render_search_summary(task_meta)

    top1, top2, top3 = st.columns(3)
    top1.metric("Results", len(records))
    top2.metric("Raw matches", raw_total)
    top3.metric("Mode", mode)

    if records:
        result_tabs = st.tabs(["Results", "Download", "Search details"])
        with result_tabs[0]:
            entity_type = (task_meta.get("target_entity_types") or ["company"])[0]
            if task_meta.get("task_type") == "document_research" or entity_type == "paper":
                _render_paper_table(records)
            elif task_meta.get("task_type") == "people_search" or entity_type == "person":
                _render_people_cards(records)
            else:
                _render_company_cards(records)

        with result_tabs[1]:
            if export_path and Path(export_path).exists():
                data = Path(export_path).read_bytes()
                st.download_button(
                    "Download results",
                    data=data,
                    file_name=Path(export_path).name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.caption("No downloadable file is available for this run.")

        with result_tabs[2]:
            with st.expander("Search interpretation", expanded=True):
                st.json(task_meta)
            with st.expander("Search logs", expanded=False):
                logs = result.get("logs", []) or []
                st.code("\n".join(logs[-120:]) if logs else "No logs")
            with st.expander("Planned queries", expanded=False):
                st.json(result.get("queries", {}))
    else:
        st.markdown('<div class="empty-box">', unsafe_allow_html=True)
        st.markdown("### No strong matches found yet")
        st.markdown(
            "Try switching to **Balanced** mode, broadening the request slightly, or adding optional integrations for wider coverage."
        )
        st.markdown('</div>', unsafe_allow_html=True)
