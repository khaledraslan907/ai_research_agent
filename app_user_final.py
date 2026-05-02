from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from openpyxl.styles import Font
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings
from pipeline.orchestrator import SearchOrchestrator

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None


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
.block-container { padding-top: 1.75rem; padding-bottom: 2rem; max-width: 1180px; }
.hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; color: #f8fafc; margin-bottom: 0.2rem; }
.hero-subtitle { color: #b3bfd4; font-size: 1.02rem; margin-bottom: 0.9rem; line-height: 1.45; }
.search-shell {
    background: rgba(15, 22, 38, 0.92);
    border: 1px solid rgba(124, 143, 179, 0.18);
    border-radius: 20px;
    padding: 1rem 1rem 0.95rem 1rem;
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
.key-status {
    color: #d7eadc; background: rgba(41, 89, 63, 0.22); border: 1px solid rgba(91, 166, 116, 0.18);
    border-radius: 12px; padding: 0.55rem 0.75rem; font-size: 0.88rem; margin-top: 0.6rem;
}
.empty-box {
    background: rgba(15, 22, 38, 0.88);
    border: 1px dashed rgba(124, 143, 179, 0.22);
    border-radius: 16px; padding: 1.25rem; color: #c4cede;
}
.download-box {
    background: rgba(15, 22, 38, 0.9);
    border: 1px solid rgba(124, 143, 179, 0.16);
    border-radius: 16px;
    padding: 1rem;
    margin-top: 1rem;
}
.stButton > button, .stDownloadButton > button { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _normalize_filename(name: str, ext: str) -> str:
    base = (name or "results").strip() or "results"
    for e in [".xlsx", ".csv", ".json", ".pdf", ".docx", ".doc"]:
        if base.lower().endswith(e):
            base = base[: -len(e)]
            break
    return f"{base}{ext}"


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


def _connected_integrations(keys: dict[str, str]) -> list[str]:
    labels = {"groq": "Groq", "gemini": "Gemini", "exa": "Exa", "tavily": "Tavily", "serpapi": "SerpApi"}
    return [labels[k] for k, v in keys.items() if str(v).strip()]


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


def _summary_from_record(r: dict) -> str:
    for key in ["summary", "description", "snippet", "notes"]:
        val = _clean(r.get(key))
        if val:
            return val
    return ""


def _build_display_df(records: list[dict], task_meta: dict) -> pd.DataFrame:
    task_type = task_meta.get("task_type", "")
    entity_type = (task_meta.get("target_entity_types") or [""])[0]
    rows = []
    for r in records:
        if task_type == "document_research" or entity_type == "paper":
            rows.append({
                "Title": _clean(r.get("company_name")) or _clean(r.get("title")),
                "Link": _clean(r.get("website")) or _clean(r.get("source_url")),
                "Authors": _clean(r.get("authors")),
                "Year": _clean(r.get("publication_year")),
                "Summary": _summary_from_record(r)[:320],
            })
        elif task_type == "people_search" or entity_type == "person":
            rows.append({
                "Name": _clean(r.get("company_name")) or _clean(r.get("title")),
                "Link": _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url")) or _clean(r.get("website")),
                "Employer": _clean(r.get("employer_name")),
                "Job title": _clean(r.get("job_title")),
                "Location": _clean(r.get("city")) or _clean(r.get("country")),
                "Summary": _summary_from_record(r)[:260],
            })
        else:
            rows.append({
                "Name": _clean(r.get("company_name")) or _clean(r.get("title")),
                "Link": _clean(r.get("website")) or _clean(r.get("source_url")),
                "Email": _clean(r.get("email")),
                "Phone": _clean(r.get("phone")),
                "LinkedIn": _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url")),
                "Summary": _summary_from_record(r)[:240],
            })
    return pd.DataFrame(rows)


def _build_export_df(records: list[dict], task_meta: dict) -> pd.DataFrame:
    rows = []
    task_type = task_meta.get("task_type", "")
    entity_type = (task_meta.get("target_entity_types") or [""])[0]
    for r in records:
        link = _clean(r.get("website")) or _clean(r.get("source_url"))
        linkedin = _clean(r.get("linkedin_profile")) or _clean(r.get("linkedin_url"))
        if task_type == "people_search" or entity_type == "person":
            link = linkedin or link
        rows.append({
            "name": _clean(r.get("company_name")) or _clean(r.get("title")),
            "link": link,
            "email": _clean(r.get("email")),
            "phone": _clean(r.get("phone")),
            "linkedin": linkedin,
            "summary": _summary_from_record(r),
        })
    return pd.DataFrame(rows)


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        ws = writer.book["Results"]
        headers = [c.value for c in ws[1]]
        link_cols = [i + 1 for i, h in enumerate(headers) if str(h).lower() in {"link", "linkedin"}]
        for col_idx in link_cols:
            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value:
                    cell.hyperlink = str(cell.value)
                    cell.font = Font(color="0563C1", underline="single")
    return out.getvalue()


def _to_pdf_bytes_vertical(df: pd.DataFrame, title: str) -> bytes:
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for idx, row in df.iterrows():
        story.append(Paragraph(f"<b>Result {idx + 1}</b>", styles["Heading3"]))
        for col, val in row.items():
            text = _clean(val)
            if text:
                safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(f"<b>{col.title()}:</b> {safe}", styles["BodyText"]))
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 10))
    doc.build(story)
    return out.getvalue()


def _to_word_bytes(df: pd.DataFrame, title: str) -> tuple[bytes, str]:
    if Document is not None:
        doc = Document()
        doc.add_heading(title, level=1)
        for idx, row in df.iterrows():
            doc.add_heading(f"Result {idx + 1}", level=2)
            for col, val in row.items():
                text = _clean(val)
                if text:
                    p = doc.add_paragraph()
                    p.add_run(f"{col.title()}: ").bold = True
                    p.add_run(text)
        out = io.BytesIO()
        doc.save(out)
        return out.getvalue(), ".docx"

    lines = [title, ""]
    for idx, row in df.iterrows():
        lines.append(f"Result {idx + 1}")
        for col, val in row.items():
            text = _clean(val)
            if text:
                lines.append(f"{col.title()}: {text}")
        lines.append("")
    return "\n".join(lines).encode("utf-8"), ".doc"


mode_defaults = {"Fast": 15, "Balanced": 25, "Deep": 40}
if "coverage_value" not in st.session_state:
    st.session_state.coverage_value = mode_defaults["Balanced"]
if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = "Balanced"

with st.sidebar:
    st.markdown("## Search settings")
    st.caption("Choose a search mode and optional integrations.")

    mode = st.radio(
        "Search mode",
        ["Fast", "Balanced", "Deep"],
        index=["Fast", "Balanced", "Deep"].index(st.session_state.mode_selected),
        horizontal=True,
        help="Fast is quickest, Balanced is recommended, and Deep is best for more difficult or niche searches.",
    )
    if mode != st.session_state.mode_selected:
        st.session_state.mode_selected = mode
        st.session_state.coverage_value = mode_defaults[mode]

    search_coverage = st.slider(
        "Search coverage",
        min_value=5,
        max_value=80,
        value=int(st.session_state.coverage_value),
        step=5,
        help="Increase this when you want to widen the search and retrieve more candidates.",
    )
    st.session_state.coverage_value = search_coverage

    with st.expander("Optional integrations", expanded=False):
        st.markdown("**Optional keys for stronger search quality**")
        st.caption("Recommended setup: Groq or Gemini for interpretation, plus Exa or Tavily for broader coverage.")

        groq_key = st.text_input("Groq API key", value="", type="password")
        gemini_key = st.text_input("Gemini API key", value="", type="password")
        exa_key = st.text_input("Exa API key", value="", type="password")
        tavily_key = st.text_input("Tavily API key", value="", type="password")
        serpapi_key = st.text_input("SerpApi key", value="", type="password")

        connected = _connected_integrations({
            "groq": groq_key or _secret("GROQ_API_KEY"),
            "gemini": gemini_key or _secret("GEMINI_API_KEY"),
            "exa": exa_key or _secret("EXA_API_KEY"),
            "tavily": tavily_key or _secret("TAVILY_API_KEY"),
            "serpapi": serpapi_key or _secret("SERPAPI_KEY"),
        })
        if connected:
            st.markdown(f"<div class='key-status'>Connected: {', '.join(connected)}</div>", unsafe_allow_html=True)

    with st.expander("Advanced settings", expanded=False):
        requested_fields = st.multiselect(
            "Requested fields",
            ["website", "email", "phone", "linkedin", "summary", "hq_country", "presence_countries", "author", "doi"],
            default=["website", "email", "phone"],
            help="Choose which details you want the search to prioritize when possible.",
        )
        min_confidence = st.slider(
            "Minimum relevance",
            0,
            100,
            25 if mode == "Fast" else 35,
            5,
            help="Higher values keep only stronger matches. Lower values widen the search.",
        )
        uploaded_file = st.file_uploader("Existing list for deduplication", type=["csv", "xlsx"])
        use_seed_dedupe = st.checkbox("Use uploaded file for deduplication", value=True)

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

if run_btn:
    st.session_state["prompt_value"] = prompt
    if prompt.strip():
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

        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)
        task_spec.mode = mode
        task_spec.max_results = int(search_coverage)
        if requested_fields:
            task_spec.target_attributes = list(dict.fromkeys(requested_fields))
        elif not task_spec.target_attributes:
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

        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text="Searching... 0%")
        progress_state = {"pct": 5}

        def _progress(msg: str):
            progress_state["pct"] = min(92, progress_state["pct"] + 8)
            progress_bar.progress(progress_state["pct"], text=f"Searching... {progress_state['pct']}%")

        result = SearchOrchestrator().run_task(
            task_spec=task_spec,
            provider_settings=provider_settings,
            uploaded_df=uploaded_df,
            budget_overrides={},
            min_confidence_score=int(min_confidence),
            user_keys=user_keys,
            progress_callback=_progress,
        )
        progress_bar.progress(100, text="Searching... 100%")
        progress_container.empty()

        st.session_state["last_result"] = result
        st.session_state["last_mode"] = mode
        st.session_state["last_coverage"] = search_coverage

result = st.session_state.get("last_result")
if result:
    task_meta = result.get("task_spec", {}) or {}
    records = result.get("records", []) or []

    _render_search_summary(task_meta)

    m1, m2 = st.columns(2)
    m1.metric("Results", len(records))
    m2.metric("Mode", st.session_state.get("last_mode", mode))

    if records:
        display_df = _build_display_df(records, task_meta)
        export_df = _build_export_df(records, task_meta)

        is_papers = task_meta.get("task_type") == "document_research" or (task_meta.get("target_entity_types") or [""])[0] == "paper"
        if is_papers:
            results_tab, summaries_tab = st.tabs(["Results", "Paper summaries"])
        else:
            results_tab = st.container()
            summaries_tab = None

        with results_tab:
            column_config = {}
            if "Link" in display_df.columns:
                column_config["Link"] = st.column_config.LinkColumn("Link")
            if "LinkedIn" in display_df.columns:
                column_config["LinkedIn"] = st.column_config.LinkColumn("LinkedIn")
            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

            st.markdown('<div class="download-box">', unsafe_allow_html=True)
            st.markdown("### Download results")
            st.caption("Exports include only the main highlights: name, link, email, phone, LinkedIn, and summary.")
            c1, c2, c3 = st.columns(3)
            excel_bytes = _to_excel_bytes(export_df)
            pdf_bytes = _to_pdf_bytes_vertical(export_df, f"Research Navigator - {_human_task(task_meta.get('task_type', ''))}")
            word_bytes, word_ext = _to_word_bytes(export_df, f"Research Navigator - {_human_task(task_meta.get('task_type', ''))}")
            with c1:
                st.download_button(
                    "Download Excel",
                    data=excel_bytes,
                    file_name=_normalize_filename("results", ".xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    on_click="ignore",
                )
            with c2:
                st.download_button(
                    "Download Word",
                    data=word_bytes,
                    file_name=_normalize_filename("results", word_ext),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    on_click="ignore",
                )
            with c3:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=_normalize_filename("results", ".pdf"),
                    mime="application/pdf",
                    use_container_width=True,
                    on_click="ignore",
                )
            st.markdown('</div>', unsafe_allow_html=True)

        if summaries_tab is not None:
            with summaries_tab:
                for _, row in export_df.iterrows():
                    title = _clean(row.get("name")) or "Untitled"
                    summary = _clean(row.get("summary"))
                    st.markdown(f"#### {title}")
                    st.write(summary or "No summary available.")
                    st.divider()
    else:
        st.markdown('<div class="empty-box">', unsafe_allow_html=True)
        st.markdown("### No strong matches found yet")
        st.markdown("Try switching to **Balanced** mode, broadening the request slightly, or adding optional integrations for wider coverage.")
        st.markdown('</div>', unsafe_allow_html=True)
