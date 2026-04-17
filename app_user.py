"""
app_user.py  —  AI Research Agent  (user-facing / deployable version)
======================================================================
Designed for Streamlit Cloud deployment.
- Users enter their own API keys (with step-by-step guidance)
- Works with just ONE key (free Groq or Gemini is enough to start)
- Only shows results, not rejected/logs/budget noise
- Clean, guided UX for non-technical users
- Reads keys from st.secrets when deployed, input fields as fallback

STREAMLIT SECRETS (when deployed on Streamlit Cloud):
  [defaults]               ← Optional pre-filled keys (your own, as defaults)
  GROQ_API_KEY = "..."
  GEMINI_API_KEY = "..."
  EXA_API_KEY = "..."
  TAVILY_API_KEY = "..."
  SERPAPI_KEY = "..."

Users can override any key via the UI.  If secrets are set, users still
see the key slots but they show as already-filled (masked).
"""
from __future__ import annotations

import sys, os
from pathlib import Path

# ── allow running from project root ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings, CompanyRecord
from pipeline.orchestrator import SearchOrchestrator

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — refined dark-industrial aesthetic ────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


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
    ext = {"xlsx": ".xlsx", "csv": ".csv", "json": ".json", "pdf": ".pdf"}.get(fmt, ".xlsx")
    name = (name or "results").strip()
    return name if name.lower().endswith(ext) else name + ext


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


def _paper_attr(paper, *names: str) -> str:
    for name in names:
        try:
            value = getattr(paper, name, "")
        except Exception:
            value = ""
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value)
    return ""


def _paper_rows_for_display(papers: list[CompanyRecord], source_rows: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    source_rows = source_rows or []
    for idx, paper in enumerate(papers):
        src = source_rows[idx] if idx < len(source_rows) else {}
        rows.append({
            "Title": _paper_attr(paper, "company_name", "title"),
            "Authors": _paper_attr(paper, "authors"),
            "Year": _paper_attr(paper, "publication_year", "year"),
            "DOI": _paper_attr(paper, "doi"),
            "URL": _paper_attr(paper, "website", "source_url"),
            "AI Summary": _paper_attr(paper, "notes"),
            "Abstract": _paper_attr(paper, "description", "abstract", "summary"),
            "Confidence": src.get("confidence_score", ""),
            "Source Provider": src.get("source_provider", ""),
        })
    return pd.DataFrame(rows)


def _file_bytes(path_str: str) -> bytes | None:
    if not path_str:
        return None
    try:
        path = Path(path_str)
        if path.exists() and path.is_file():
            return path.read_bytes()
    except Exception:
        return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Key guidance content
# ──────────────────────────────────────────────────────────────────────────────

KEY_GUIDE = {
    "groq": {
        "label":   "Groq API Key",
        "icon":    "⚡",
        "cost":    "100% Free · 14,400 requests/day",
        "steps":   [
            ('Go to', 'https://console.groq.com', 'console.groq.com'),
            ('Click', None, '"Sign Up" (free, no credit card)'),
            ('Click', None, '"API Keys" → "Create API Key"'),
            ('Copy', None, 'the key starting with gsk_... and paste above'),
        ],
        "why":     "Powers smart query planning and result ranking. Most important key.",
    },
    "gemini": {
        "label":   "Google Gemini Key",
        "icon":    "🧠",
        "cost":    "Free · 1,500 requests/day",
        "steps":   [
            ('Go to', 'https://aistudio.google.com', 'aistudio.google.com'),
            ('Click', None, '"Get API Key" → "Create API Key"'),
            ('Copy', None, 'the key starting with AIza... and paste above'),
        ],
        "why":     "Backup LLM — used when Groq quota is exhausted.",
    },
    "exa": {
        "label":   "Exa API Key",
        "icon":    "🔭",
        "cost":    "Free · 1,000 searches/month",
        "steps":   [
            ('Go to', 'https://exa.ai', 'exa.ai'),
            ('Click', None, '"Sign Up" → confirm email'),
            ('Go to', None, 'Dashboard → "API Keys" → "New Key"'),
            ('Copy', None, 'the key and paste above'),
        ],
        "why":     "Best semantic search — finds things DDG misses. Needed for LinkedIn people search.",
    },
    "tavily": {
        "label":   "Tavily API Key",
        "icon":    "🌐",
        "cost":    "Free · 1,000 searches/month",
        "steps":   [
            ('Go to', 'https://tavily.com', 'tavily.com'),
            ('Click', None, '"Get Started Free"'),
            ('Copy', None, 'your API key from the dashboard'),
        ],
        "why":     "Adds question-style search diversity. Helps reach higher result counts.",
    },
    "serpapi": {
        "label":   "SerpApi Key",
        "icon":    "🎯",
        "cost":    "Free · 100 searches/month",
        "steps":   [
            ('Go to', 'https://serpapi.com', 'serpapi.com'),
            ('Click', None, '"Register" (free plan available)'),
            ('Copy', None, 'your API key from the dashboard'),
        ],
        "why":     "Best for LinkedIn people search via site:linkedin.com queries.",
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
    "🚀 Fast":      {"mode": "Fast",     "max": 15,  "desc": "~15 sec · DDG only · quick check"},
    "⚖️ Balanced":  {"mode": "Balanced", "max": 40,  "desc": "~1–2 min · best quality/speed ratio · recommended"},
    "🔬 Deep":      {"mode": "Deep",     "max": 100, "desc": "~5–10 min · all providers · maximum results"},
}

EXAMPLES = [
    ("🏢 Companies",  "Find oilfield service companies in Egypt and Saudi Arabia with email and phone"),
    ("📄 Papers",     "Find research papers about Electrical Submersible Pump with authors export as PDF"),
    ("👥 LinkedIn",   "Find LinkedIn profiles of petroleum engineers in oil gas companies in Egypt")
]


# ──────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="section-header">🔑 Your API Keys</p>', unsafe_allow_html=True)

    st.markdown(
        "The agent is free to use. It needs API keys to search and reason. "
        "**One free key is enough to start** — Groq takes 30 seconds to get.",
        unsafe_allow_html=False,
    )

    # ── LLM KEYS (required for smart search) ─────────────────────────────────
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

    # Hidden paid keys — only shown if already set via secrets
    _anthropic = _secret("ANTHROPIC_API_KEY")
    _openai    = _secret("OPENAI_API_KEY")
    anthropic_key = _anthropic
    openai_key    = _openai
    openrouter_key = _secret("OPENROUTER_API_KEY")
    firecrawl_key  = _secret("FIRECRAWL_API_KEY")

    # Show active key summary
    active_keys = []
    if groq_key:    active_keys.append("Groq")
    if gemini_key:  active_keys.append("Gemini")
    if exa_key:     active_keys.append("Exa")
    if tavily_key:  active_keys.append("Tavily")
    if serpapi_key: active_keys.append("SerpApi")
    if _anthropic:  active_keys.append("Claude")
    if _openai:     active_keys.append("GPT")

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
    preset     = MODE_PRESETS[mode_label]
    mode       = preset["mode"]
    st.caption(preset["desc"])

    max_results = st.slider(
        "Max results", 5, 150,
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

    # ── ADVANCED (collapsed for regular users) ────────────────────────────────
    with st.expander("🛠️ Advanced options", expanded=False):
        min_conf = st.slider(
            "Min confidence score", 0, 100, 35, step=5,
            help="Lower = more results but less accurate. 35 is a good default.",
        )
        export_fmt = st.selectbox(
            "Export format",
            ["Auto-detect from prompt", "Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)", "JSON (.json)"],
        )
        export_filename = st.text_input("Export filename", value="results")
        paper_summary_limit = st.slider("Higher-quality Feynman summaries for top papers", 0, 10, 5, help="All papers will get a quick review. This controls how many top papers get the richer Feynman pass when available.")
        paper_synthesis_label = st.selectbox(
            "Paper summary depth",
            ["Literature review (faster)", "Deep research (slower)"],
            index=0,
        )
        export_summary_pdf = st.checkbox("Create PDF research brief for paper searches", value=True)

        st.caption("**Fields to collect:**")
        col1, col2 = st.columns(2)
        field_website  = col1.checkbox("Website",     value=True)
        field_email    = col1.checkbox("Email",        value=True)
        field_phone    = col1.checkbox("Phone",        value=True)
        field_linkedin = col2.checkbox("LinkedIn URL", value=False)
        field_hq       = col2.checkbox("HQ Country",   value=False)
        field_summary  = col2.checkbox("Summary",      value=False)


# ──────────────────────────────────────────────────────────────────────────────
# ── MAIN AREA ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>🔍 AI Research Agent</h1>"
    "<p style='color:#7b8099; font-size:15px; margin-top:4px; font-family:IBM Plex Sans'>"
    "Find companies, research papers, or LinkedIn profiles — just describe what you want in plain English."
    "</p>",
    unsafe_allow_html=True,
)

# Example prompts
with st.expander("💡 Example searches", expanded=False):
    for cat, ex in EXAMPLES:
        c1, c2 = st.columns([1, 8])
        c1.markdown(f"<span class='tag'>{cat}</span>", unsafe_allow_html=True)
        if c2.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["prompt_value"] = ex
            st.rerun()

# Main prompt
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

# Key check warning before running
_has_llm   = bool(groq_key or gemini_key or _anthropic or _openai or openrouter_key)
_has_search = bool(exa_key or tavily_key or serpapi_key)

if not _has_llm and not _has_search:
    st.warning(
        "⚠️ No API keys set — results will be limited (DDG-only, no ranking). "
        "Add a free Groq key in the sidebar for 10× better results."
    )

# Run button
run_btn = st.button(
    "🚀 Run Search",
    type="primary",
    use_container_width=True,
    disabled=not prompt.strip(),
)

# ──────────────────────────────────────────────────────────────────────────────
# ── RUN ───────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if run_btn and prompt.strip():

    # Build LLM client
    llm_client = FreeLLMClient(
        groq_api_key       = groq_key,
        gemini_api_key     = gemini_key,
        openrouter_api_key = openrouter_key,
        anthropic_api_key  = anthropic_key,
        openai_api_key     = openai_key,
    )
    backends = llm_client.available_backends()

    # Parse intent
    with st.spinner("🧠 Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    # Apply settings
    task_spec.mode        = mode
    task_spec.max_results = int(max_results)
    has_geo = bool(
        task_spec.geography.include_countries
        or task_spec.geography.exclude_countries
        or task_spec.geography.exclude_presence_countries
    )
    task_spec.geography.strict_mode = has_geo

    # Format override
    fmt_map = {
        "Auto-detect from prompt": None,
        "Excel (.xlsx)": "xlsx",
        "CSV (.csv)":    "csv",
        "PDF (.pdf)":    "pdf",
        "JSON (.json)":  "json",
    }
    if fmt_map.get(export_fmt):
        task_spec.output.format = fmt_map[export_fmt]
    task_spec.output.filename = _normalize_filename(export_filename, task_spec.output.format)

    # Fields
    user_fields = []
    if field_website:  user_fields.append("website")
    if field_email:    user_fields.append("email")
    if field_phone:    user_fields.append("phone")
    if field_linkedin: user_fields.append("linkedin")
    if field_hq:       user_fields.append("hq_country")
    if field_summary:  user_fields.append("summary")

    if task_spec.task_type == "document_research":
        task_spec.target_attributes = sorted(set(["website","summary","author"] + user_fields))
    elif task_spec.task_type == "people_search":
        task_spec.target_attributes = ["linkedin", "website"]
    else:
        task_spec.target_attributes = user_fields or ["website","email","phone"]

    # Topic check
    if not task_spec.industry or len(task_spec.industry.strip()) < 2:
        st.error(
            "⚠️ Couldn't detect the topic. Try being more specific: "
            "'Find oil and gas service companies...' or "
            "'Find research papers about ESP...'"
        )
        st.stop()

    # Show understanding banner
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

    # Provider settings (auto from keys + mode)
    provider_settings = ProviderSettings(
        use_ddg      = True,
        use_exa      = bool(exa_key),
        use_tavily   = bool(tavily_key),
        use_serpapi  = bool(serpapi_key),
        use_firecrawl = bool(firecrawl_key),
        use_llm_parser = bool(backends),
        use_uploaded_seed_dedupe = uploaded_file is not None,
    )

    user_keys = {
        "groq_api_key":       groq_key,
        "gemini_api_key":     gemini_key,
        "openrouter_api_key": openrouter_key,
        "exa_api_key":        exa_key,
        "tavily_api_key":     tavily_key,
        "serpapi_key":        serpapi_key,
        "firecrawl_api_key":  firecrawl_key,
        "anthropic_api_key":  anthropic_key,
        "openai_api_key":     openai_key,
    }

    # Progress
    progress_bar = st.progress(0, text="Starting...")
    status_box   = st.empty()
    STAGE_PCT = {
        "DDG": 20, "Exa": 45, "Tavily": 65, "SerpApi": 75,
        "dedup": 85, "LLM re-rank": 92, "Accepted": 98,
    }

    def _progress(msg: str):
        status_box.caption(f"⏳ {msg}")
        for kw, pct in STAGE_PCT.items():
            if kw.lower() in msg.lower():
                progress_bar.progress(pct, text=f"{kw}...")
                break

    orchestrator = SearchOrchestrator()
    uploaded_df  = _read_file(uploaded_file)

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

    total = result["total_found"]
    raw   = result["raw_search_results"]
    rej   = len(result.get("rejected_records", []))
    b     = result["budget"]

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if total == 0:
        st.error("No results found.")
        hints = []
        if not exa_key:   hints.append("add a **free Exa key** (exa.ai) in the sidebar")
        if not tavily_key: hints.append("add a **free Tavily key** (tavily.com)")
        if mode == "Fast": hints.append("switch to **Balanced** or **Deep** mode")
        hints.append("try **lowering the confidence score** to 25 in Advanced options")
        hints.append("rephrase your search to include the specific industry or topic")
        if hints:
            st.info("**To get results:**\n" + "\n".join(f"- {h}" for h in hints))
        st.stop()

    # Success summary
    st.markdown(
        f'<div style="background:#0d2137;border:1px solid #1a4f7a;border-radius:10px;'
        f'padding:16px 24px;margin:12px 0">'
        f'<span style="font-size:20px;font-weight:600;color:#4fc3f7">✅ {total} results found</span>'
        f'<span style="color:#7b8099;font-size:13px;margin-left:16px">'
        f'from {raw} raw · {rej} filtered out · {b["total_search_calls_used"]} searches</span></div>',
        unsafe_allow_html=True,
    )

    # ── Build DataFrame ────────────────────────────────────────────────────────
    records = result.get("records", [])
    df = pd.DataFrame(records)

    # ── Determine columns to show ──────────────────────────────────────────────
    is_people = task_spec.task_type == "people_search"
    is_papers = task_spec.task_type == "document_research"

    if is_people:
        SHOW_COLS = ["company_name", "job_title", "employer_name", "city",
                     "linkedin_url", "linkedin_profile"]
        COL_RENAME = {
            "company_name":   "Name",
            "job_title":      "Job Title",
            "employer_name":  "Company",
            "city":           "Location",
            "linkedin_url":   "LinkedIn URL",
            "linkedin_profile": "Profile Link",
        }
    elif is_papers:
        SHOW_COLS = ["company_name", "authors", "publication_year", "doi",
                     "description", "website"]
        COL_RENAME = {
            "company_name":     "Title",
            "authors":          "Authors",
            "publication_year": "Year",
            "doi":              "DOI",
            "description":      "Abstract",
            "website":          "URL",
        }
    else:
        SHOW_COLS = ["company_name", "website"]
        if field_email:    SHOW_COLS.append("email")
        if field_phone:    SHOW_COLS.append("phone")
        if field_linkedin: SHOW_COLS.append("linkedin_url")
        if field_hq:       SHOW_COLS.append("hq_country")
        if field_summary:  SHOW_COLS.append("description")
        SHOW_COLS.append("confidence_score")
        COL_RENAME = {
            "company_name":   "Company",
            "website":        "Website",
            "email":          "Email",
            "phone":          "Phone",
            "linkedin_url":   "LinkedIn",
            "hq_country":     "HQ Country",
            "description":    "Description",
            "confidence_score": "Score",
        }

    # Build clean display df
    show = [c for c in SHOW_COLS if c in df.columns]
    df_display = df[show].copy()
    df_display.columns = [COL_RENAME.get(c, c) for c in show]
    if "Score" in df_display.columns:
        df_display = df_display.sort_values("Score", ascending=False)

    # ── TABS + AUTO PAPER SUMMARY ───────────────────────────────────────────
    is_papers = task_spec.task_type == "document_research"
    paper_records: list[CompanyRecord] = []
    paper_display_df = pd.DataFrame()
    paper_export_paths: dict[str, str] = {}
    paper_summary_report = ""
    paper_summary_error = ""
    paper_synthesis_mode = "deepresearch" if paper_synthesis_label.startswith("Deep") else "lit"

    if is_papers:
        paper_records = [CompanyRecord(**r) for r in result.get("records", [])]
        try:
            from core.feynman_bridge import auto_summarize_and_export

            summary_status = st.empty()
            with st.spinner("🔬 Preparing automatic paper summaries and PDF brief..."):
                def _paper_progress(msg: str):
                    summary_status.caption(f"🔬 {msg}")

                paper_records, paper_summary_report, paper_export_paths = auto_summarize_and_export(
                    papers=paper_records,
                    topic=task_spec.industry or prompt,
                    export_dir="outputs",
                    export_pdf=export_summary_pdf,
                    per_paper_limit=int(paper_summary_limit),
                    synthesis_mode=paper_synthesis_mode,
                    progress_callback=_paper_progress,
                )
            summary_status.empty()
            summary_engine = paper_export_paths.get("summary_engine", "built-in")
            if summary_engine == "feynman":
                st.success(
                    f"🧠 Paper summary ready · engine: Feynman · "
                    f"{sum(1 for p in paper_records if getattr(p, "notes", ""))} quick reviews generated"
                )
            else:
                st.success(
                    f"🧠 Paper summary ready · engine: built-in fallback · "
                    f"{sum(1 for p in paper_records if getattr(p, "notes", ""))} quick reviews generated"
                )
            if paper_export_paths.get("pdf_error"):
                st.warning(f"PDF note: {paper_export_paths['pdf_error']}")
        except Exception as exc:
            paper_summary_error = f"Automatic paper summarization failed: {exc}"
            st.warning(paper_summary_error)

        paper_display_df = _paper_rows_for_display(paper_records, result.get("records", []))

    # ── TABS ─────────────────────────────────────────────────────────────────
    if is_papers:
        tab_results, tab_summary, tab_download, tab_details = st.tabs([
            f"📊 Results ({total})",
            "🧠 Summary",
            "⬇️ Download",
            "🔍 Search Details",
        ])
    else:
        tab_results, tab_download, tab_details = st.tabs([
            f"📊 Results ({total})",
            "⬇️ Download",
            "🔍 Search Details",
        ])
        tab_summary = None

    # ── Results tab ────────────────────────────────────────────────────────────
    with tab_results:
        if is_people:
            st.markdown(f"**{total} LinkedIn profiles found**")
            for _, row in df_display.head(total).iterrows():
                name    = row.get("Name", "")
                title   = row.get("Job Title", "")
                company = row.get("Company", "")
                loc     = row.get("Location", "")
                url     = row.get("LinkedIn URL", "") or row.get("Profile Link", "")

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
            st.markdown(f"**{len(paper_records)} papers found**")
            if paper_summary_error:
                st.caption("Showing paper results without automatic summaries.")

            for _, row in paper_display_df.iterrows():
                title   = row.get("Title", "")
                authors = row.get("Authors", "")
                year    = row.get("Year", "")
                doi     = row.get("DOI", "")
                url     = row.get("URL", "")
                summary = str(row.get("AI Summary", "") or "")[:420]
                abstract = str(row.get("Abstract", "") or "")[:220]

                meta_parts = [p for p in [str(authors)[:80], str(year)] if p and p != "nan"]
                meta = " · ".join(meta_parts)
                link_ref = doi or url or ""
                link_html = f'<a href="{link_ref}" target="_blank" class="result-link">🔗 View Paper</a>' if link_ref else ""
                summary_html = (
                    f'<div style="font-size:13px;color:#d7d9e2;margin:8px 0"><strong>AI Summary:</strong> {summary}{"..." if len(summary) == 420 else ""}</div>'
                    if summary else ""
                )
                abstract_html = (
                    f'<div style="font-size:12px;color:#5a6080;margin:6px 0">{abstract}{"..." if len(abstract) == 220 else ""}</div>'
                    if abstract else ""
                )

                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-name">📄 {title}</div>'
                    f'<div class="result-meta">{meta}</div>'
                    f'{summary_html}'
                    f'{abstract_html}'
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
                    "Website":  st.column_config.LinkColumn("Website"),
                    "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
                    "Email":    st.column_config.TextColumn("Email"),
                    "Score":    st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                },
            )

    if tab_summary is not None:
        with tab_summary:
            st.markdown("### 🧠 Automatic paper summary")
            if paper_summary_error:
                st.warning(paper_summary_error)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Papers found", len(paper_records))
                c2.metric("Quick reviews ready", sum(1 for p in paper_records if getattr(p, "notes", "")))
                c3.metric("PDF ready", "Yes" if paper_export_paths.get("pdf") else "No")

                st.info("Quick review summaries are attached to each paper below.")

                if not paper_display_df.empty:
                    st.dataframe(
                        paper_display_df[[c for c in ["Title", "Authors", "Year", "AI Summary"] if c in paper_display_df.columns]],
                        use_container_width=True,
                        height=460,
                    )

                if paper_summary_report:
                    with st.expander("Topic overview (optional combined synthesis)", expanded=False):
                        st.markdown(paper_summary_report)

    # ── Download tab ───────────────────────────────────────────────────────────
    with tab_download:
        st.markdown("### ⬇️ Download your results")
        st.caption("Choose the format you want:")

        dl_col1, dl_col2, dl_col3, dl_col4, dl_col5 = st.columns(5)

        export_df = paper_display_df if is_papers and not paper_display_df.empty else df_display

        try:
            import io
            import openpyxl  # noqa: F401
            xl_buf = io.BytesIO()
            export_df.to_excel(xl_buf, index=False, engine="openpyxl")
            xl_buf.seek(0)
            dl_col1.download_button(
                "📊 Excel (.xlsx)",
                data=xl_buf,
                file_name=_normalize_filename(export_filename, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            dl_col1.caption("Excel export unavailable (openpyxl missing)")

        dl_col2.download_button(
            "📄 CSV (.csv)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=_normalize_filename(export_filename, "csv"),
            mime="text/csv",
            use_container_width=True,
        )

        dl_col3.download_button(
            "🔧 JSON (.json)",
            data=export_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=_normalize_filename(export_filename, "json"),
            mime="application/json",
            use_container_width=True,
        )

        md_bytes = _file_bytes(paper_export_paths.get("markdown", "")) if is_papers else None
        if md_bytes:
            dl_col4.download_button(
                "📝 Summary (.md)",
                data=md_bytes,
                file_name=Path(paper_export_paths["markdown"]).name,
                mime="text/markdown",
                use_container_width=True,
            )
        elif not is_papers:
            dl_col4.caption("Markdown summary only for paper searches")
        else:
            dl_col4.caption("Summary markdown not ready")

        pdf_bytes = None
        if is_papers:
            pdf_bytes = _file_bytes(paper_export_paths.get("pdf", ""))
        else:
            ep = result.get("export_path", "")
            if ep and Path(ep).exists() and ep.endswith(".pdf"):
                pdf_bytes = Path(ep).read_bytes()
        if pdf_bytes:
            file_name = Path(paper_export_paths["pdf"]).name if is_papers and paper_export_paths.get("pdf") else _normalize_filename(export_filename, "pdf")
            dl_col5.download_button(
                "📑 PDF",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            dl_col5.caption("PDF not ready")

        if is_papers and paper_export_paths.get("pdf_error"):
            st.warning(f"PDF export note: {paper_export_paths['pdf_error']}")

        st.divider()
        st.markdown("**Preview of what you'll download:**")
        st.dataframe(export_df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(export_df)} rows.")

    # ── Search details tab ────────────────────────────────────────────────────
    with tab_details:
        st.markdown("### 🔍 Search details")
        st.caption("Technical details about how the search ran.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Results found",    total)
        c2.metric("Raw candidates",   raw)
        c3.metric("Filtered out",     rej)
        c4.metric("Searches made",    b["total_search_calls_used"])

        if backends:
            st.info(f"🤖 AI backends used: {', '.join(backends)}")
        else:
            st.warning("No LLM backend — running without AI ranking. Add a Groq or Gemini key for better results.")

        with st.expander("What the agent understood from your prompt"):
            geo = task_spec.geography
            st.markdown(f"- **Task type:** {task_spec.task_type}")
            st.markdown(f"- **Topic:** {task_spec.industry}")
            st.markdown(f"- **Mode:** {mode}")
            if geo.include_countries:
                st.markdown(f"- **Search in:** {', '.join(c.title() for c in geo.include_countries)}")
            if geo.exclude_countries:
                st.markdown(f"- **Excluded:** {', '.join(c.title() for c in geo.exclude_countries)}")

        with st.expander("Providers used"):
            provider_used = []
            if b.get("ddg_calls_used", 0) > 0:    provider_used.append(f"DuckDuckGo ({b['ddg_calls_used']} calls)")
            if b.get("exa_calls_used", 0) > 0:     provider_used.append(f"Exa ({b['exa_calls_used']} calls)")
            if b.get("tavily_calls_used", 0) > 0:  provider_used.append(f"Tavily ({b['tavily_calls_used']} calls)")
            if b.get("serpapi_calls_used", 0) > 0: provider_used.append(f"SerpApi ({b['serpapi_calls_used']} calls)")
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
    'AI Research Agent · Powered by free APIs · No data stored'
    '</p>',
    unsafe_allow_html=True,
)
