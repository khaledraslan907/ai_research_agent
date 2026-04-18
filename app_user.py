"""
app_user.py  —  AI Research Agent  (user-facing / deployable version)
=====================================================================
Designed for Streamlit Cloud deployment.

Key fixes in this version:
- Search results, task metadata, and summaries are stored in st.session_state
- Summary generation no longer makes the results disappear
- Summary page stays selected after rerun (radio-based sections instead of st.tabs)
- Paper results download uses Excel / CSV / PDF
- Summary download uses TXT / PDF
- Paper results table removes DOI and tries harder to recover Authors / Year
"""

import io
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings, CompanyRecord

try:
    from core.paper_summarizer import (
        summarize_papers,
        summaries_to_text,
        summaries_to_pdf_bytes,
    )
except Exception:
    # Safe fallback so the whole app does not crash on import
    from core.paper_summarizer import summarize_papers, summaries_to_text

    def summaries_to_pdf_bytes(summaries, topic):
        return None

from pipeline.orchestrator import SearchOrchestrator


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ────────────────────────────────────────────────────────────
if "search_result" not in st.session_state:
    st.session_state["search_result"] = None

if "task_meta" not in st.session_state:
    st.session_state["task_meta"] = None

if "paper_summaries" not in st.session_state:
    st.session_state["paper_summaries"] = []

if "summaries_topic" not in st.session_state:
    st.session_state["summaries_topic"] = ""

if "prompt_value" not in st.session_state:
    st.session_state["prompt_value"] = ""

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "results"


# ── Custom CSS — refined dark-industrial aesthetic ────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background: #0f1117; color: #e8eaf0; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }

.result-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #4f6ef7; }

.result-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 4px;
}
.result-meta {
    font-size: 13px;
    color: #7b8099;
    margin-bottom: 6px;
}
.result-link {
    font-size: 13px;
    color: #4f6ef7;
    text-decoration: none;
}
.tag {
    display: inline-block;
    background: #1e2236;
    border: 1px solid #363d5c;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #8892b0;
    margin-right: 6px;
}
.key-step {
    background: #151823;
    border-left: 3px solid #4f6ef7;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}
.key-step a { color: #4f6ef7; }
.metric-row { display: flex; gap: 16px; margin: 8px 0; }
.metric-box {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
}
.metric-num { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 600; }
.metric-lbl { font-size: 11px; color: #7b8099; text-transform: uppercase; letter-spacing: 0.05em; }
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f6ef7;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #2d3148;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Secrets helpers — read from st.secrets then fall back to env
# ──────────────────────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, then env, then default."""
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _read_file(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    except Exception:
        return None


def _normalize_filename(name: str, fmt: str) -> str:
    ext_map = {"xlsx": ".xlsx", "csv": ".csv", "pdf": ".pdf", "json": ".json"}
    desired_ext = ext_map.get(fmt, ".xlsx")
    name = (name or "results").strip()

    for ext in ext_map.values():
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


def _normalize_paper_records(records: list[dict]) -> pd.DataFrame:
    """
    Build a robust paper dataframe from many possible field names.
    DOI removed intentionally.
    """
    rows = []

    for r in records:
        title = _first_nonempty(r, [
            "company_name", "title", "paper_title", "document_title", "name"
        ])

        authors = _first_nonempty(r, [
            "authors", "author", "paper_authors", "creators", "creator"
        ])

        year = _parse_year(_first_nonempty(r, [
            "publication_year", "year", "published_year",
            "publication_date", "published_date", "date"
        ]))

        abstract = _first_nonempty(r, [
            "description", "abstract", "summary", "snippet", "paper_abstract"
        ])

        url = _first_nonempty(r, [
            "website", "url", "paper_url", "source_url", "link"
        ])

        rows.append(
            {
                "Title": title,
                "Authors": authors,
                "Year": year,
                "Abstract": abstract,
                "URL": url,
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


def _safe_pdf_text(value, max_len: int = 350) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _df_to_pdf_bytes(df: pd.DataFrame, title: str = "Results") -> bytes | None:
    """
    Returns PDF bytes using reportlab.
    Install once if needed:
        pip install reportlab
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None

    if df is None or df.empty:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 10),
    ]

    pdf_df = df.copy()
    for col in pdf_df.columns:
        pdf_df[col] = pdf_df[col].map(lambda x: _safe_pdf_text(x, 220))

    max_rows = min(len(pdf_df), 200)
    table_data = [list(pdf_df.columns)] + pdf_df.head(max_rows).values.tolist()

    usable_width = 780
    col_width = max(90, usable_width / max(1, len(pdf_df.columns)))
    col_widths = [col_width] * len(pdf_df.columns)

    table = Table(table_data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ]
        )
    )

    story.append(table)
    if len(pdf_df) > max_rows:
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Note: PDF preview includes first {max_rows} rows only.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Key guidance content
# ──────────────────────────────────────────────────────────────────────────────
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


def _show_key_guide(key_id: str):
    g = KEY_GUIDE[key_id]
    st.markdown(f"**{g['icon']} {g['label']}** — {g['cost']}")
    for action, url, text in g["steps"]:
        link = f'<a href="{url}" target="_blank">{text}</a>' if url else text
        st.markdown(f'<div class="key-step">→ {action} {link}</div>', unsafe_allow_html=True)
    st.caption(f"💡 {g['why']}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode presets
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">🔑 Your API Keys</p>', unsafe_allow_html=True)

    st.markdown(
        "The agent is free to use. It needs API keys to search and reason. "
        "**One free key is enough to start** — Groq takes 30 seconds to get.",
        unsafe_allow_html=False,
    )

    # ── LLM KEYS ─────────────────────────────────────────────────────────────
    with st.expander("⚡ Groq Key — Free, most important", expanded=True):
        _default_groq = _secret("GROQ_API_KEY")
        groq_key = st.text_input(
            "Groq API Key",
            value=_default_groq,
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if not groq_key:
            _show_key_guide("groq")
        else:
            st.success(f"✅ Groq connected  ({_mask(groq_key)})")

    with st.expander("🧠 Gemini Key — Free backup LLM", expanded=False):
        _default_gemini = _secret("GEMINI_API_KEY")
        gemini_key = st.text_input(
            "Gemini Key",
            value=_default_gemini,
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if not gemini_key:
            _show_key_guide("gemini")
        else:
            st.success(f"✅ Gemini connected  ({_mask(gemini_key)})")

    st.markdown('<p class="section-header">🔭 Search Provider Keys</p>', unsafe_allow_html=True)
    st.caption("Optional but strongly recommended. Each is free with generous quotas.")

    with st.expander("🔭 Exa — semantic search + LinkedIn profiles", expanded=False):
        _default_exa = _secret("EXA_API_KEY")
        exa_key = st.text_input(
            "Exa Key",
            value=_default_exa,
            type="password",
            placeholder="your-exa-key",
            label_visibility="collapsed",
        )
        if not exa_key:
            _show_key_guide("exa")
        else:
            st.success(f"✅ Exa connected  ({_mask(exa_key)})")

    with st.expander("🌐 Tavily — question-style search", expanded=False):
        _default_tavily = _secret("TAVILY_API_KEY")
        tavily_key = st.text_input(
            "Tavily Key",
            value=_default_tavily,
            type="password",
            placeholder="tvly-...",
            label_visibility="collapsed",
        )
        if not tavily_key:
            _show_key_guide("tavily")
        else:
            st.success(f"✅ Tavily connected  ({_mask(tavily_key)})")

    with st.expander("🎯 SerpApi — LinkedIn + Google search", expanded=False):
        _default_serp = _secret("SERPAPI_KEY")
        serpapi_key = st.text_input(
            "SerpApi Key",
            value=_default_serp,
            type="password",
            placeholder="your-serpapi-key",
            label_visibility="collapsed",
        )
        if not serpapi_key:
            _show_key_guide("serpapi")
        else:
            st.success(f"✅ SerpApi connected  ({_mask(serpapi_key)})")

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

    # ── SEARCH OPTIONS ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">⚙️ Search Options</p>', unsafe_allow_html=True)

    mode_label = st.radio(
        "Search depth",
        list(MODE_PRESETS.keys()),
        index=1,
        help="Balanced is recommended for most searches.",
    )
    preset = MODE_PRESETS[mode_label]
    mode = preset["mode"]
    st.caption(preset["desc"])

    max_results = st.slider(
        "Max results",
        5,
        150,
        value=preset["max"],
        step=5,
        help="How many results you want. More = slower search.",
    )

    st.divider()

    # ── SEED FILE (dedup) ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📂 Avoid Duplicates</p>', unsafe_allow_html=True)
    st.caption("Upload a previous results file — new results won't repeat those.")
    uploaded_file = st.file_uploader(
        "Previous results (CSV or Excel)",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ Dedup file loaded: {uploaded_file.name}")

    st.divider()

    # ── ADVANCED ──────────────────────────────────────────────────────────────
    with st.expander("🛠️ Advanced options", expanded=False):
        min_conf = st.slider(
            "Min confidence score",
            0,
            100,
            35,
            step=5,
            help="Lower = more results but less accurate. 35 is a good default.",
        )
        export_fmt = st.selectbox(
            "Export format",
            ["Auto-detect from prompt", "Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)"],
        )
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
# Build LLM client on every run
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
# MAIN AREA
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
    st.warning(
        "⚠️ No API keys set — results will be limited (DDG-only, no ranking). "
        "Add a free Groq key in the sidebar for 10× better results."
    )

run_btn = st.button(
    "🚀 Run Search",
    type="primary",
    use_container_width=True,
    disabled=not prompt.strip(),
)


# ──────────────────────────────────────────────────────────────────────────────
# RUN SEARCH
# ──────────────────────────────────────────────────────────────────────────────
if run_btn and prompt.strip():
    st.session_state["search_result"] = None
    st.session_state["task_meta"] = None
    st.session_state["paper_summaries"] = []
    st.session_state["summaries_topic"] = ""
    st.session_state["active_section"] = "results"

    with st.spinner("🧠 Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    task_spec.mode = mode
    task_spec.max_results = int(max_results)

    has_geo = bool(
        task_spec.geography.include_countries
        or task_spec.geography.exclude_countries
        or task_spec.geography.exclude_presence_countries
    )
    task_spec.geography.strict_mode = has_geo

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

    if task_spec.task_type == "document_research":
        task_spec.target_attributes = sorted(
            set(
                [
                    "website",
                    "summary",
                    "author",
                    "authors",
                    "year",
                    "publication_year",
                    "published_date",
                    "abstract",
                ] + user_fields
            )
        )
    elif task_spec.task_type == "people_search":
        task_spec.target_attributes = ["linkedin", "website"]
    else:
        task_spec.target_attributes = user_fields or ["website", "email", "phone"]

    if not task_spec.industry or len(task_spec.industry.strip()) < 2:
        st.error(
            "⚠️ Couldn't detect the topic. Try being more specific: "
            "'Find oil and gas service companies...' or "
            "'Find research papers about ESP...'"
        )
        st.stop()

    geo = task_spec.geography
    with st.container():
        c1, c2, c3 = st.columns(3)
        task_labels = {
            "entity_discovery": "🏢 Companies",
            "document_research": "📄 Research Papers",
            "people_search": "👥 LinkedIn Profiles",
            "market_research": "📊 Market Research",
        }
        c1.info(f"**Looking for:** {task_labels.get(task_spec.task_type, '🔍 Entities')}")
        c2.info(f"**Topic:** {task_spec.industry}")
        c3.info(f"**Target:** {max_results} results in {mode} mode")
        if geo.include_countries:
            st.info(f"🌍 Searching IN: {', '.join(c.title() for c in geo.include_countries)}")
        if geo.exclude_countries:
            st.warning(f"🚫 Excluding: {', '.join(c.title() for c in geo.exclude_countries)}")
        if not geo.include_countries and not geo.exclude_countries:
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
    STAGE_PCT = {
        "DDG": 20,
        "Exa": 45,
        "Tavily": 65,
        "SerpApi": 75,
        "dedup": 85,
        "LLM re-rank": 92,
        "Accepted": 98,
    }

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
        "industry": task_spec.industry,
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
# RENDER RESULTS FROM SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("search_result")
task_meta = st.session_state.get("task_meta")

if result and task_meta:
    total = result["total_found"]
    raw = result["raw_search_results"]
    rej = len(result.get("rejected_records", []))
    b = result["budget"]

    st.markdown(
        f'<div style="background:#0d2137;border:1px solid #1a4f7a;border-radius:10px;'
        f'padding:16px 24px;margin:12px 0">'
        f'<span style="font-size:20px;font-weight:600;color:#4fc3f7">✅ {total} results found</span>'
        f'<span style="color:#7b8099;font-size:13px;margin-left:16px">'
        f'from {raw} raw · {rej} filtered out · {b["total_search_calls_used"]} searches</span></div>',
        unsafe_allow_html=True,
    )

    records = result.get("records", []) or []
    df = pd.DataFrame(records)

    is_people = task_meta["task_type"] == "people_search"
    is_papers = task_meta["task_type"] == "document_research"

    if is_people:
        SHOW_COLS = ["company_name", "job_title", "employer_name", "city", "linkedin_url", "linkedin_profile"]
        COL_RENAME = {
            "company_name": "Name",
            "job_title": "Job Title",
            "employer_name": "Company",
            "city": "Location",
            "linkedin_url": "LinkedIn URL",
            "linkedin_profile": "Profile Link",
        }
        show = [c for c in SHOW_COLS if c in df.columns]
        df_display = df[show].copy() if show else pd.DataFrame()
        df_display.columns = [COL_RENAME.get(c, c) for c in show]

    elif is_papers:
        df_display = _normalize_paper_records(records)

    else:
        SHOW_COLS = ["company_name", "website"]
        if field_email:
            SHOW_COLS.append("email")
        if field_phone:
            SHOW_COLS.append("phone")
        if field_linkedin:
            SHOW_COLS.append("linkedin_url")
        if field_hq:
            SHOW_COLS.append("hq_country")
        if field_summary:
            SHOW_COLS.append("description")
        SHOW_COLS.append("confidence_score")

        COL_RENAME = {
            "company_name": "Company",
            "website": "Website",
            "email": "Email",
            "phone": "Phone",
            "linkedin_url": "LinkedIn",
            "hq_country": "HQ Country",
            "description": "Description",
            "confidence_score": "Score",
        }

        show = [c for c in SHOW_COLS if c in df.columns]
        df_display = df[show].copy() if show else pd.DataFrame()
        df_display.columns = [COL_RENAME.get(c, c) for c in show]

        if "Score" in df_display.columns:
            df_display = df_display.sort_values("Score", ascending=False)

    if is_papers:
        section_labels = {
            "results": f"📊 Results ({total})",
            "summaries": "📝 AI Summaries",
            "download": "⬇️ Download",
            "details": "🔍 Search Details",
        }
        section_options = list(section_labels.keys())
    else:
        section_labels = {
            "results": f"📊 Results ({total})",
            "download": "⬇️ Download",
            "details": "🔍 Search Details",
        }
        section_options = list(section_labels.keys())

    if st.session_state.get("active_section") not in section_options:
        st.session_state["active_section"] = section_options[0]

    selected_section = st.radio(
        "View",
        options=section_options,
        horizontal=True,
        key="active_section",
        format_func=lambda x: section_labels[x],
    )

    if selected_section == "results":
        if is_people:
            st.markdown(f"**{total} LinkedIn profiles found**")
            for _, row in df_display.head(total).iterrows():
                name = row.get("Name", "")
                title = row.get("Job Title", "")
                company = row.get("Company", "")
                loc = row.get("Location", "")
                url = row.get("LinkedIn URL", "") or row.get("Profile Link", "")

                meta_parts = [p for p in [title, company, loc] if p]
                meta = " · ".join(meta_parts)
                link_html = f'<a href="{url}" target="_blank" class="result-link">🔗 View Profile</a>' if url else ""

                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-name">👤 {name}</div>'
                    f'<div class="result-meta">{meta}</div>'
                    f'{link_html}'
                    f'</div>',
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
                    f'<div class="result-card">'
                    f'<div class="result-name">📄 {title}</div>'
                    f'<div class="result-meta">{meta}</div>'
                    f'<div style="font-size:12px;color:#5a6080;margin:6px 0">{abstract}{"..." if len(abstract) == 250 else ""}</div>'
                    f'{link_html}'
                    f'</div>',
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
                description = _first_nonempty(r, [
                    "description", "abstract", "summary", "snippet", "paper_abstract"
                ])
                if not description.strip():
                    continue

                summary_candidates.append(
                    CompanyRecord(
                        company_name=_first_nonempty(r, [
                            "company_name", "title", "paper_title", "document_title", "name"
                        ]) or "Untitled",
                        authors=_first_nonempty(r, [
                            "authors", "author", "paper_authors", "creators", "creator"
                        ]),
                        description=description,
                        doi=_first_nonempty(r, ["doi"]),
                        website=_first_nonempty(r, [
                            "website", "url", "paper_url", "source_url", "link"
                        ]),
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
                c1.download_button(
                    "📑 PDF",
                    data=pdf_bytes,
                    file_name=f"summaries_{fn}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                c1.warning("Install reportlab to enable PDF summaries.")

            c2.download_button(
                "📄 Text (.txt)",
                data=txt_bytes,
                file_name=f"summaries_{fn}.txt",
                mime="text/plain",
                use_container_width=True,
            )

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
        pdf_bytes = _df_to_pdf_bytes(
            df_display,
            title=f"Search Results - {task_meta.get('industry', 'Results')}",
        )

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        if xl_bytes:
            dl_col1.download_button(
                "📊 Excel (.xlsx)",
                data=xl_bytes,
                file_name=_normalize_filename(export_filename, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        dl_col2.download_button(
            "📄 CSV (.csv)",
            data=csv_bytes,
            file_name=_normalize_filename(export_filename, "csv"),
            mime="text/csv",
            use_container_width=True,
        )

        if pdf_bytes:
            dl_col3.download_button(
                "📑 PDF",
                data=pdf_bytes,
                file_name=_normalize_filename(export_filename, "pdf"),
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            dl_col3.warning("Install reportlab to enable PDF export.")

        st.divider()
        st.dataframe(df_display.head(5), use_container_width=True)
        st.caption(f"Preview: first 5 of {len(df_display)} rows.")

    elif selected_section == "details":
        st.markdown("### 🔍 Search details")
        st.caption("Technical details about how the search ran.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Results found", total)
        c2.metric("Raw candidates", raw)
        c3.metric("Filtered out", rej)
        c4.metric("Searches made", b["total_search_calls_used"])

        if backends:
            st.info(f"🤖 AI backends used: {', '.join(backends)}")
        else:
            st.warning("No LLM backend — running without AI ranking. Add a Groq or Gemini key for better results.")

        with st.expander("What the agent understood from your prompt"):
            st.markdown(f"- **Task type:** {task_meta.get('task_type', '')}")
            st.markdown(f"- **Topic:** {task_meta.get('industry', '')}")
            st.markdown(f"- **Mode:** {task_meta.get('mode', '')}")
            if task_meta.get("include_countries"):
                st.markdown(f"- **Search in:** {', '.join(c.title() for c in task_meta['include_countries'])}")
            if task_meta.get("exclude_countries"):
                st.markdown(f"- **Excluded:** {', '.join(c.title() for c in task_meta['exclude_countries'])}")

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


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#3d4363;font-size:12px;font-family:IBM Plex Mono">'
    "AI Research Agent · Powered by free APIs · No data stored"
    "</p>",
    unsafe_allow_html=True,
)
