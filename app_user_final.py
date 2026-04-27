from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings
from pipeline.orchestrator import SearchOrchestrator


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
.stApp { background: linear-gradient(180deg, #070b14 0%, #0b1020 100%); color: #e7ecf5; }
.block-container { padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1180px; }
.hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.035em; color: #f8fafc; margin-bottom: 0.18rem; line-height: 1.1; }
.hero-subtitle { color: #b3bfd4; font-size: 0.98rem; margin-bottom: 0.95rem; line-height: 1.45; max-width: 860px; }
.search-shell {
    background: rgba(15, 22, 38, 0.92);
    border: 1px solid rgba(124, 143, 179, 0.18);
    border-radius: 20px;
    padding: 1.05rem 1.05rem 0.95rem 1.05rem;
    box-shadow: 0 18px 44px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}
.info-strip {
    background: linear-gradient(135deg, rgba(25,34,58,0.92) 0%, rgba(15,22,38,0.92) 100%);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
}
.info-strip-title { color: #f8fafc; font-weight: 600; margin-bottom: 0.18rem; }
.info-strip-text { color: #aeb9cb; font-size: 0.9rem; line-height: 1.45; }
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
.empty-box {
    background: rgba(15, 22, 38, 0.88);
    border: 1px dashed rgba(124, 143, 179, 0.22);
    border-radius: 16px; padding: 1.25rem; color: #c4cede;
}
.key-status {
    color: #d7eadc; background: rgba(41, 89, 63, 0.22); border: 1px solid rgba(91, 166, 116, 0.18);
    border-radius: 12px; padding: 0.55rem 0.75rem; font-size: 0.88rem; margin-top: 0.6rem;
}
.summary-box {
    background: rgba(15, 22, 38, 0.88);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.85rem;
}
.summary-title { color: #f8fafc; font-weight: 700; margin-bottom: 0.45rem; }
.summary-text { color: #d8dfeb; line-height: 1.6; }
.stButton > button { border-radius: 12px; }
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


def _normalize_filename(name: str) -> str:
    base = (name or "results").strip() or "results"
    for ext in [".xlsx", ".csv", ".json", ".pdf", ".docx", ".doc"]:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


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
    st.markdown('</div>', unsafe_allow_html=True)


def _long_summary(record: dict) -> str:
    for key in ["summary", "description", "snippet", "notes"]:
        text = _clean(record.get(key))
        if text:
            return re.sub(r"\s+", " ", text).strip()
    return ""


def _paper_short_summary(record: dict) -> str:
    text = _long_summary(record)
    if not text:
        return "No summary available."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    brief = " ".join(sentences[:2]).strip()
    if len(brief) < 80:
        brief = text[:500].strip()
    return brief


def _records_to_display_df(records: list[dict], task_meta: dict) -> pd.DataFrame:
    task_type = task_meta.get("task_type", "")
    entity_type = (task_meta.get("target_entity_types") or [""])[0]

    if task_type == "document_research" or entity_type == "paper":
        rows = []
        for r in records:
            rows.append(
                {
                    "Title": _clean(r.get("company_name")) or _clean(r.get("title")),
                    "Authors": _clean(r.get("authors")),
                    "Year": _clean(r.get("publication_year")),
                    "DOI": _clean(r.get("doi")),
                    "Link": _clean(r.get("website")) or _clean(r.get("source_url")),
                    "Summary": _paper_short_summary(r),
                }
            )
        return pd.DataFrame(rows)

    if task_type == "people_search" or entity_type == "person":
        rows = []
        for r in records:
            rows.append(
                {
                    "Name": _clean(r.get("company_name")) or _clean(r.get("title")),
                    "Link": _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url")) or _clean(r.get("website")),
                    "Email": _clean(r.get("email")),
                    "Phone": _clean(r.get("phone")),
                    "LinkedIn": _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url")),
                    "Summary": _long_summary(r)[:350],
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for r in records:
        rows.append(
            {
                "Name": _clean(r.get("company_name")) or _clean(r.get("title")),
                "Link": _clean(r.get("website")) or _clean(r.get("source_url")),
                "Email": _clean(r.get("email")),
                "Phone": _clean(r.get("phone")),
                "LinkedIn": _clean(r.get("linkedin_url")) or _clean(r.get("linkedin_profile")),
                "Summary": _long_summary(r)[:350],
            }
        )
    return pd.DataFrame(rows)


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def _to_word_bytes(df: pd.DataFrame, title: str) -> tuple[bytes, str, str]:
    try:
        from docx import Document  # type: ignore

        doc = Document()
        doc.add_heading(title, level=1)
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = "Table Grid"
        for i, col in enumerate(df.columns):
            table.rows[0].cells[i].text = str(col)
        for _, row in df.fillna("").iterrows():
            cells = table.add_row().cells
            for i, col in enumerate(df.columns):
                cells[i].text = str(row[col])[:1200]
        out = BytesIO()
        doc.save(out)
        return out.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
    except Exception:
        html = [f"<html><body><h1>{title}</h1><table border='1' cellspacing='0' cellpadding='4'>"]
        html.append("<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>")
        for _, row in df.fillna("").iterrows():
            html.append("<tr>" + "".join(f"<td>{str(row[c])}</td>" for c in df.columns) + "</tr>")
        html.append("</table></body></html>")
        return "".join(html).encode("utf-8"), "application/msword", ".doc"


def _to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=0.6 * cm, leftMargin=0.6 * cm, topMargin=0.6 * cm, bottomMargin=0.6 * cm)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * cm)]

    pdf_df = df.fillna("").copy()
    for col in pdf_df.columns:
        pdf_df[col] = pdf_df[col].astype(str).map(lambda x: (x[:180] + "…") if len(x) > 180 else x)

    data = [list(pdf_df.columns)] + pdf_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3f73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b9c4d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()


def _render_downloads(df: pd.DataFrame, title: str, base_name: str):
    st.markdown("### Downloads")
    c1, c2, c3 = st.columns(3)

    excel_bytes = _to_excel_bytes(df)
    word_bytes, word_mime, word_ext = _to_word_bytes(df, title)
    pdf_bytes = _to_pdf_bytes(df, title)

    with c1:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Word",
            data=word_bytes,
            file_name=f"{base_name}{word_ext}",
            mime=word_mime,
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{base_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
mode_defaults = {"Fast": 15, "Balanced": 25, "Deep": 40}

with st.sidebar:
    st.markdown("## Search settings")
    st.caption("Choose a search mode and optional integrations.")

    mode = st.radio(
        "Search mode",
        ["Fast", "Balanced", "Deep"],
        index=1,
        horizontal=True,
        help="Fast is quickest, Balanced is recommended, and Deep is best for more difficult or niche searches.",
    )
    search_coverage = st.slider(
        "Search coverage",
        5,
        100,
        mode_defaults[mode],
        5,
        help="Increase this to widen the search.",
    )

    with st.expander("Optional integrations", expanded=False):
        st.markdown("**Optional keys for stronger search quality**")
        st.caption("Recommended setup: Groq or Gemini for interpretation, plus Exa or Tavily for broader coverage.")

        groq_key = st.text_input("Groq API key", value="", type="password")
        gemini_key = st.text_input("Gemini API key", value="", type="password")
        exa_key = st.text_input("Exa API key", value="", type="password")
        tavily_key = st.text_input("Tavily API key", value="", type="password")
        serpapi_key = st.text_input("SerpApi key", value="", type="password")

        connected = [
            label
            for label, val in {
                "Groq": groq_key or _secret("GROQ_API_KEY"),
                "Gemini": gemini_key or _secret("GEMINI_API_KEY"),
                "Exa": exa_key or _secret("EXA_API_KEY"),
                "Tavily": tavily_key or _secret("TAVILY_API_KEY"),
                "SerpApi": serpapi_key or _secret("SERPAPI_KEY"),
            }.items()
            if str(val).strip()
        ]
        if connected:
            st.markdown(f"<div class='key-status'>Connected: {', '.join(connected)}</div>", unsafe_allow_html=True)

        with st.expander("How to get keys", expanded=False):
            st.markdown(
                """
**[Groq](https://console.groq.com/keys)**  
Create a free account → open **API Keys** → create a key → paste it here.

**[Gemini](https://aistudio.google.com/app/apikey)**  
Open **Google AI Studio** → create an API key → paste it here.

**[Exa](https://dashboard.exa.ai/api-keys)**  
Create an account → open your dashboard → generate an API key → paste it here.

**[Tavily](https://docs.tavily.com/documentation/quickstart)**  
Create a free account → open **API Keys** → copy your key → paste it here.

**[SerpApi](https://serpapi.com/users/sign_in)**  
Create an account → open your dashboard → copy the API key → paste it here.
                """
            )

    with st.expander("Advanced settings", expanded=False):
        export_filename = st.text_input("Base filename", value="results")
        min_confidence = st.slider("Minimum relevance", 0, 100, 25 if mode == "Fast" else 35, 5)
        uploaded_file = st.file_uploader("Existing list for deduplication", type=["csv", "xlsx"])
        use_seed_dedupe = st.checkbox("Use uploaded file for deduplication", value=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Research Navigator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Search companies, academic papers, LinkedIn accounts, tenders, and exhibitors in English or Arabic.</div>', unsafe_allow_html=True)

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
    value=st.session_state.get("prompt_value", ""),
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
run_btn = st.button("Start search", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Run search
# -----------------------------------------------------------------------------
if run_btn:
    if not prompt.strip():
        st.stop()

    st.session_state["prompt_value"] = prompt

    user_keys = {
        "groq_api_key": groq_key or _secret("GROQ_API_KEY"),
        "gemini_api_key": gemini_key or _secret("GEMINI_API_KEY"),
        "exa_api_key": exa_key or _secret("EXA_API_KEY"),
        "tavily_api_key": tavily_key or _secret("TAVILY_API_KEY"),
        "serpapi_key": serpapi_key or _secret("SERPAPI_KEY"),
    }

    llm_client = FreeLLMClient(
        groq_api_key=user_keys["groq_api_key"],
        gemini_api_key=user_keys["gemini_api_key"],
    )

    with st.spinner("Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    task_spec.mode = mode
    task_spec.max_results = int(search_coverage)
    task_spec.output.filename = _normalize_filename(export_filename)
    if not task_spec.target_attributes:
        task_spec.target_attributes = ["website"]

    provider_settings = ProviderSettings(
        use_ddg=True,
        use_exa=bool(user_keys["exa_api_key"]),
        use_tavily=bool(user_keys["tavily_api_key"]),
        use_serpapi=bool(user_keys["serpapi_key"]),
        use_firecrawl=False,
        use_llm_parser=bool(llm_client.available_backends()),
        use_uploaded_seed_dedupe=use_seed_dedupe,
    )
    uploaded_df = _read_file(uploaded_file)

    progress_box = st.empty()
    with progress_box.container():
        st.info("Searching...")
        bar = st.progress(5)

    def _progress(msg: str):
        nonlocal_bar = bar
        try:
            val = min(nonlocal_bar._value + 10, 90)  # type: ignore[attr-defined]
        except Exception:
            val = 35
        nonlocal_bar.progress(val, text=msg)

    result = SearchOrchestrator().run_task(
        task_spec=task_spec,
        provider_settings=provider_settings,
        uploaded_df=uploaded_df,
        budget_overrides={},
        min_confidence_score=int(min_confidence),
        user_keys=user_keys,
        progress_callback=_progress,
    )
    bar.progress(100, text="Search complete")
    progress_box.empty()

    task_meta = result.get("task_spec", {}) or {}
    records = result.get("records", []) or []
    raw_total = int(result.get("raw_search_results", 0) or 0)

    _render_search_summary(task_meta)

    top1, top2, top3 = st.columns(3)
    top1.metric("Results", len(records))
    top2.metric("Raw matches", raw_total)
    top3.metric("Mode", mode)

    if records:
        display_df = _records_to_display_df(records, task_meta)
        entity_type = (task_meta.get("target_entity_types") or ["company"])[0]
        is_papers = task_meta.get("task_type") == "document_research" or entity_type == "paper"
        base_name = _normalize_filename(export_filename)
        title = f"Research Navigator - {_human_task(task_meta.get('task_type', ''))}"

        if is_papers:
            tab_results, tab_summaries = st.tabs(["Results", "Paper summaries"])
            with tab_results:
                st.dataframe(display_df, use_container_width=True, height=520)
                _render_downloads(display_df, title, base_name)
            with tab_summaries:
                for _, row in display_df.iterrows():
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown(f"<div class='summary-title'>{_clean(row.get('Title')) or 'Untitled paper'}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='summary-text'>{_clean(row.get('Summary')) or 'No summary available.'}</div>", unsafe_allow_html=True)
                    link = _clean(row.get("Link"))
                    if link:
                        st.markdown(f"<div style='margin-top:0.5rem;'><a href='{link}' target='_blank'>Open source</a></div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.dataframe(display_df, use_container_width=True, height=520)
            _render_downloads(display_df, title, base_name)
    else:
        st.markdown('<div class="empty-box">', unsafe_allow_html=True)
        st.markdown("### No strong matches found yet")
        st.markdown("Try switching to **Balanced** mode, broadening the request slightly, or increasing **Search coverage**.")
        st.markdown('</div>', unsafe_allow_html=True)
