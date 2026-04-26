from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(180deg, #0b1020 0%, #0e1426 100%);
    color: #e7ecf5;
}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px;}
.hero-title {
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em;
    color: #f7f9fc; margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: #a8b3c7; font-size: 1rem; margin-bottom: 1.2rem;
}
.search-shell {
    background: rgba(18, 25, 45, 0.82);
    border: 1px solid rgba(124, 143, 179, 0.18);
    border-radius: 18px;
    padding: 1rem 1rem 0.8rem 1rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}
.summary-card {
    background: rgba(18, 25, 45, 0.84);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.metric-chip {
    display: inline-block; padding: 0.28rem 0.62rem; border-radius: 999px;
    background: rgba(85, 124, 255, 0.14); border: 1px solid rgba(85, 124, 255, 0.25);
    color: #d7e4ff; font-size: 0.82rem; margin: 0 0.4rem 0.4rem 0;
}
.result-card {
    background: rgba(18, 25, 45, 0.78);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.85rem;
}
.result-card:hover { border-color: rgba(106, 154, 255, 0.42); }
.result-title { color: #f7f9fc; font-size: 1.02rem; font-weight: 600; margin-bottom: 0.35rem; }
.result-meta { color: #a8b3c7; font-size: 0.88rem; margin-bottom: 0.45rem; }
.result-desc { color: #d8dfeb; font-size: 0.92rem; line-height: 1.45; }
.badge {
    display: inline-block; padding: 0.2rem 0.52rem; border-radius: 999px; font-size: 0.74rem;
    border: 1px solid rgba(124, 143, 179, 0.2); color: #d6ddea; background: rgba(255,255,255,0.03);
    margin-right: 0.35rem; margin-bottom: 0.35rem;
}
.empty-box {
    background: rgba(18, 25, 45, 0.78);
    border: 1px dashed rgba(124, 143, 179, 0.22);
    border-radius: 16px; padding: 1.25rem; color: #c4cede;
}
.small-muted { color: #8f9bb0; font-size: 0.86rem; }
.compact-section { margin-top: 0.75rem; }
hr { border-color: rgba(124, 143, 179, 0.14); }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


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
        "document_research": "Research Papers",
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
            st.markdown(f'<span class="badge">Email available</span>', unsafe_allow_html=True)
        if phone:
            st.markdown(f'<span class="badge">Phone available</span>', unsafe_allow_html=True)
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
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=520)


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


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Research Navigator")
    st.caption("Professional search workspace")

    mode = st.radio("Search mode", ["Fast", "Balanced", "Deep"], index=1, horizontal=True)
    max_results = st.slider("Results to return", 5, 100, 25, 5)

    with st.expander("Optional integrations", expanded=False):
        groq_key = st.text_input("Groq", value=_secret("GROQ_API_KEY"), type="password")
        gemini_key = st.text_input("Gemini", value=_secret("GEMINI_API_KEY"), type="password")
        exa_key = st.text_input("Exa", value=_secret("EXA_API_KEY"), type="password")
        tavily_key = st.text_input("Tavily", value=_secret("TAVILY_API_KEY"), type="password")
        serpapi_key = st.text_input("SerpApi", value=_secret("SERPAPI_KEY"), type="password")
        firecrawl_key = st.text_input("Firecrawl", value=_secret("FIRECRAWL_API_KEY"), type="password")
        openrouter_key = st.text_input("OpenRouter", value=_secret("OPENROUTER_API_KEY"), type="password")
        anthropic_key = st.text_input("Anthropic", value=_secret("ANTHROPIC_API_KEY"), type="password")
        openai_key = st.text_input("OpenAI", value=_secret("OPENAI_API_KEY"), type="password")
        st.caption("All integrations are optional.")

    with st.expander("Advanced settings", expanded=False):
        export_format = st.selectbox("Export format", ["xlsx", "csv", "json", "pdf"], index=0)
        export_filename = st.text_input("Export filename", value="results")
        min_confidence = st.slider("Minimum relevance", 0, 100, 25 if mode == "Fast" else 35, 5)
        uploaded_file = st.file_uploader("Existing list for deduplication", type=["csv", "xlsx"])
        use_seed_dedupe = st.checkbox("Use uploaded file for deduplication", value=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Research Navigator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Search companies, papers, people, tenders, exhibitors, and products in English or Arabic.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="search-shell">', unsafe_allow_html=True)
prompt = st.text_area(
    "What would you like to search?",
    value="",
    height=165,
    placeholder="Describe what you want to find in English or Arabic.\n\nمثال: ابحث عن شركات خدمات البترول في مصر مع الموقع الإلكتروني والإيميل\n\nExample: Find software companies in food manufacturing in Germany with website and email.",
)
run_btn = st.button("Start search", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Run search
# -----------------------------------------------------------------------------
if run_btn:
    if not prompt.strip():
        st.stop()

    user_keys = {
        "groq_api_key": groq_key,
        "gemini_api_key": gemini_key,
        "exa_api_key": exa_key,
        "tavily_api_key": tavily_key,
        "serpapi_key": serpapi_key,
        "firecrawl_api_key": firecrawl_key,
        "openrouter_api_key": openrouter_key,
        "anthropic_api_key": anthropic_key,
        "openai_api_key": openai_key,
    }

    llm_client = FreeLLMClient(
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
    )

    with st.spinner("Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    task_spec.mode = mode
    task_spec.max_results = int(max_results)
    task_spec.output.format = export_format
    task_spec.output.filename = _normalize_filename(export_filename, export_format)
    # Keep requested fields closer to parser output.
    if not task_spec.target_attributes:
        task_spec.target_attributes = ["website"]

    provider_settings = ProviderSettings(
        use_ddg=True,
        use_exa=bool(exa_key),
        use_tavily=bool(tavily_key),
        use_serpapi=bool(serpapi_key),
        use_firecrawl=bool(firecrawl_key),
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
    llm_backends = result.get("llm_backends", []) or []

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
            if llm_backends:
                st.caption("Enhanced search enabled")
    else:
        st.markdown('<div class="empty-box">', unsafe_allow_html=True)
        st.markdown("### No strong matches found yet")
        st.markdown(
            "Try switching to **Balanced** mode, broadening the search slightly, or adding optional integrations for wider coverage."
        )
        st.markdown('</div>', unsafe_allow_html=True)
