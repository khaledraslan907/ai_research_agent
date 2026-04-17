from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.free_llm_client import FreeLLMClient
from core.llm_task_parser import parse_task_prompt_llm_first
from core.models import ProviderSettings, CompanyRecord
from core.keyword_expander import expand_keywords, build_expanded_queries
from pipeline.orchestrator import SearchOrchestrator

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Mode presets — changing mode auto-fills all provider checkboxes + budget sliders
# Numbers are intentionally larger than before to return more results
# ─────────────────────────────────────────────────────────────────────────────
MODE_PRESETS = {
    "Fast": {
        "description": "DDG only · no LLM re-ranking · ~10–20 sec · quick check. Max ~20 results.",
        "providers": {"ddg": True,  "exa": False, "tavily": False, "serpapi": False, "firecrawl": False},
        "budget":    {"total": 10,  "ddg": 4, "exa": 0, "tavily": 0, "serpapi": 0, "pages": 30},
    },
    "Balanced": {
        "description": "DDG + Exa + Tavily · LLM re-ranking · ~1–2 min · recommended. Up to ~60 results.",
        "providers": {"ddg": True,  "exa": True,  "tavily": True,  "serpapi": False, "firecrawl": False},
        "budget":    {"total": 30,  "ddg": 7, "exa": 5, "tavily": 4, "serpapi": 0, "pages": 90},
    },
    "Deep": {
        "description": "All providers · max diversity · ~3–8 min · maximum results. Up to 150+ results.",
        "providers": {"ddg": True,  "exa": True,  "tavily": True,  "serpapi": True,  "firecrawl": True},
        "budget":    {"total": 60,  "ddg": 12, "exa": 8, "tavily": 6, "serpapi": 4, "pages": 300},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_uploaded_file(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        if file.name.lower().endswith(".xlsx"):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
    return None


def _normalize_filename(filename: str, fmt: str) -> str:
    filename = (filename or "results").strip()
    ext_map  = {"xlsx": ".xlsx", "csv": ".csv", "json": ".json", "pdf": ".pdf"}
    ext = ext_map.get(fmt, ".xlsx")
    return filename if filename.lower().endswith(ext) else filename + ext


def _show_task_banner(task_spec: dict):
    geo  = task_spec.get("geography", {}) or {}
    inc  = geo.get("include_countries", []) or []
    exc  = geo.get("exclude_countries", []) or []
    excp = geo.get("exclude_presence_countries", []) or []

    TYPE_LABELS = {
        "entity_discovery":         "🔍 Find companies / entities",
        "document_research":        "📄 Find papers / documents",
        "entity_enrichment":        "✏️ Enrich existing list",
        "similar_entity_expansion": "🔁 Find similar entities",
        "market_research":          "📊 Market research",
    }
    CAT_LABELS = {
        "service_company":  "⚙️ Service company",
        "software_company": "💻 Software / digital company",
        "general":          "🏢 General entity",
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Task",     TYPE_LABELS.get(task_spec.get("task_type",""), task_spec.get("task_type","")))
    c2.metric("Topic",    task_spec.get("industry", "—") or "—")
    c3.metric("Max",      f"{task_spec.get('max_results', 25)} results")

    c4, c5 = st.columns(2)
    c4.metric("Category", CAT_LABELS.get(task_spec.get("target_category",""), "—"))
    c5.metric("Collecting", ", ".join(task_spec.get("target_attributes") or ["website"]))

    if inc:  st.info(f"🌍 Search IN: {', '.join(c.title() for c in inc)}")
    if exc:  st.warning(f"🚫 Exclude HQ in: {', '.join(c.title() for c in exc)}")
    if excp: st.warning(f"🚫 Exclude presence in: {', '.join(c.title() for c in excp)}")
    if not inc and not exc and not excp:
        st.caption("🌐 No geography filter — searching globally")


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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # 1 — FREE LLM KEYS ───────────────────────────────────────────────────────
    st.header("🆓 Free LLM Keys")
    st.caption("No credit card needed. Add at least one for smart search + re-ranking.")
    with st.expander("🔑 Add free LLM keys", expanded=True):
        groq_key = st.text_input(
            "Groq API Key", value="", type="password",
            help="FREE · 14 400 req/day · console.groq.com (30 sec, no card)")
        gemini_key = st.text_input(
            "Google Gemini Key", value="", type="password",
            help="FREE · 1 500 req/day · aistudio.google.com")
        openrouter_key = st.text_input(
            "OpenRouter Key", value="", type="password",
            help="FREE tier · openrouter.ai")

    # 2 — SEARCH PROVIDER KEYS ────────────────────────────────────────────────
    with st.expander("🔑 Search provider keys (optional free tiers)", expanded=False):
        exa_key       = st.text_input("Exa API Key",       value="", type="password",
                                       help="1 000 free/month · exa.ai")
        tavily_key    = st.text_input("Tavily API Key",    value="", type="password",
                                       help="1 000 free/month · tavily.com")
        serpapi_key   = st.text_input("SerpApi Key",       value="", type="password",
                                       help="100 free/month · serpapi.com")
        firecrawl_key = st.text_input("Firecrawl API Key", value="", type="password",
                                       help="JS-heavy sites · free trial · firecrawl.dev")

    # 3 — PAID KEYS ───────────────────────────────────────────────────────────
    with st.expander("💳 Paid LLM keys (optional fallback)", expanded=False):
        anthropic_key = st.text_input("Anthropic Claude", value="", type="password")
        openai_key    = st.text_input("OpenAI GPT",       value="", type="password")

    st.divider()

    # 4 — MODE ────────────────────────────────────────────────────────────────
    st.header("⚙️ Search Mode")
    mode = st.radio(
        "Mode", ["Fast", "Balanced", "Deep"],
        index=1, horizontal=True,
        help="Mode auto-sets providers and budget below.",
    )
    preset = MODE_PRESETS[mode]
    st.caption(f"ℹ️ {preset['description']}")

    st.divider()

    # 5 — RESULTS ─────────────────────────────────────────────────────────────
    st.header("🎯 Results")
    max_results = st.number_input(
        "Max results to return", 1, 500, 25, step=5,
        help="Agent keeps searching until this many accepted results are found or budget runs out.")
    min_conf = st.slider(
        "Min confidence score", 0, 100, 35, step=5,
        help="Lower = more results, some noise. 35 is a good starting point.")

    st.header("📋 Fields to collect")
    field_website  = st.checkbox("Website",           value=True)
    field_email    = st.checkbox("Email",              value=True)
    field_phone    = st.checkbox("Phone / contact",    value=True)
    field_linkedin = st.checkbox("LinkedIn",           value=False)
    field_hq       = st.checkbox("HQ country",         value=False)
    field_presence = st.checkbox("Branch presence",    value=False)
    field_summary  = st.checkbox("Summary / abstract", value=False)

    st.divider()

    # 6 — PROVIDERS (auto-set by mode) ────────────────────────────────────────
    st.header("🔌 Providers")
    st.caption("Auto-set by mode. Override if needed.")
    p = preset["providers"]
    use_ddg       = st.checkbox("DuckDuckGo (always free)",       value=p["ddg"])
    use_exa       = st.checkbox("Exa (1 000 free/month)",          value=p["exa"])
    use_tavily    = st.checkbox("Tavily (1 000 free/month)",       value=p["tavily"])
    use_serpapi   = st.checkbox("SerpApi (100 free/month)",        value=p["serpapi"])
    use_firecrawl = st.checkbox(
        "Firecrawl (JS-heavy sites fallback)",
        value=p["firecrawl"],
        help=(
            "Firecrawl is used automatically when normal scraping returns 403/timeout. "
            "Requires a Firecrawl API key. Free trial at firecrawl.dev."
        ),
    )

    # 7 — BUDGET (auto-set by mode) ───────────────────────────────────────────
    st.header("💰 Budget caps")
    st.caption("Auto-set by mode. Increase for more results; decrease to save API credits.")
    b = preset["budget"]
    max_total_calls   = st.number_input("Max total search calls", 1, 200, b["total"],   step=1)
    max_ddg_calls     = st.number_input("Max DDG calls",          0,  50, b["ddg"],     step=1)
    max_exa_calls     = st.number_input("Max Exa calls",          0,  50, b["exa"],     step=1)
    max_tavily_calls  = st.number_input("Max Tavily calls",       0,  50, b["tavily"],  step=1)
    max_serpapi_calls = st.number_input("Max SerpApi calls",      0,  50, b["serpapi"], step=1)
    max_pages         = st.number_input("Max pages to scrape",    1, 500, b["pages"],   step=10)

    st.divider()

    # 8 — ADVANCED ────────────────────────────────────────────────────────────
    with st.expander("🎛️ Advanced overrides (auto-detected, edit if wrong)", expanded=False):
        st.caption("The agent detects these from your prompt. Only change if the detection is wrong.")
        task_override   = st.selectbox("Task type",
            ["auto","entity_discovery","entity_enrichment",
             "similar_entity_expansion","market_research","document_research"], index=0)
        entity_override = st.selectbox("Entity type",
            ["auto","company","person","paper","organization","event","product"], index=0)
        format_override = st.selectbox("Output format",
            ["auto","xlsx","csv","json","pdf"], index=0)
        export_filename = st.text_input("Export filename", value="results")
        auto_summarize_papers = st.checkbox("Auto-summarize papers after search", value=True)
        paper_summary_limit = st.number_input("Top N papers for richer Feynman summaries", 0, 20, 5, step=1, help="All papers will still get a quick review. This controls how many top papers receive the richer Feynman pass when available.")
        paper_synthesis_mode = st.selectbox(
            "Combined synthesis mode",
            ["lit", "deepresearch"],
            index=0,
            help="lit is faster. deepresearch is slower but more detailed.",
        )
        export_summary_pdf = st.checkbox("Create PDF research summary", value=True)

    # 9 — SEED FILE ───────────────────────────────────────────────────────────
    with st.expander("📂 Seed file — deduplication", expanded=False):
        st.caption("Upload your existing company list. New results won't duplicate them.")
        uploaded_file   = st.file_uploader("CSV or Excel file", type=["csv","xlsx"])
        use_seed_dedupe = st.checkbox("Deduplicate against seed file", value=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main prompt area
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ("Companies", "Find oilfield service companies in Saudi Arabia and UAE with email and phone"),
    ("Companies", "Find digital oil and gas software companies outside USA and Egypt with email"),
    ("Companies", "Find renewable energy software companies in Germany and Norway with email"),
    ("Companies", "Find AI and data analytics companies in the energy sector outside USA with LinkedIn"),
    ("Companies", "Find upstream drilling services contractors in Middle East with contact details"),
    ("Papers",    "Find research papers about Electrical Submersible Pump with title, authors and export as PDF"),
    ("Papers",    "Find papers about efficacy of TMS in management of stuttering with authors, export in pdf file"),
    ("Papers",    "Find papers about carbon capture CCS techniques, export to CSV"),
]

st.title("🔍 AI Research Agent")
st.caption(
    "Find companies, research papers, or any entity — just describe what you want in plain language. "
    "The agent searches multiple sources, filters noise, and exports clean structured results."
)

with st.expander("💡 Example prompts", expanded=False):
    for cat, ex in EXAMPLES:
        c1, c2 = st.columns([1, 9])
        c1.caption(f"`{cat}`")
        if c2.button(ex, key=f"ex_{hash(ex)}"):
            st.session_state["prompt_value"] = ex
            st.rerun()

prompt = st.text_area(
    "What do you want to research?",
    value=st.session_state.get("prompt_value", ""),
    height=90,
    placeholder=(
        "Examples:\n"
        "• Find oilfield service companies in Saudi Arabia and UAE with email and phone\n"
        "• Find research papers about ESP Electrical Submersible Pump with title and authors, export as PDF\n"
        "• Find digital oil and gas software companies outside USA and Egypt with email"
    ),
    key="main_prompt",
)

# ── PEOPLE SEARCH OPTIONS (shown when LinkedIn is detected) ─────────────
if st.session_state.get("prompt_value", "") or prompt.strip():
    _check_prompt = (st.session_state.get("prompt_value", "") or prompt or "").lower()
    _is_people = any(w in _check_prompt for w in [
        "linkedin", "people", "engineers", "managers", "hr", "professionals",
        "employees", "staff", "personnel",
    ])
    if _is_people:
        with st.expander("👥 LinkedIn People Search Options", expanded=True):
            st.info(
                "🔍 **LinkedIn People Search mode detected.** The agent will use "
                "`site:linkedin.com/in` X-ray searches via Google/DDG/SerpApi to find "
                "public LinkedIn profiles. No LinkedIn account or API needed."
            )
            c1, c2 = st.columns(2)
            job_levels_sel = c1.multiselect(
                "Job levels to search",
                options=["Engineer / Specialist", "Manager / Supervisor",
                         "Director / VP", "HR / Talent Acquisition", "Executive / C-suite"],
                default=["Engineer / Specialist", "Manager / Supervisor",
                         "HR / Talent Acquisition"],
                help="Select which seniority levels to target.",
            )
            _level_map = {
                "Engineer / Specialist":    "engineer",
                "Manager / Supervisor":     "manager",
                "Director / VP":            "director",
                "HR / Talent Acquisition":  "hr",
                "Executive / C-suite":      "executive",
            }
            st.session_state["job_levels"] = [_level_map[l] for l in job_levels_sel]

            extra_titles = c2.text_input(
                "Extra job titles (optional, comma-separated)",
                value="",
                placeholder="e.g. Wellsite Geologist, Reservoir Specialist",
                help="Add specific job titles beyond the levels above.",
            )
            st.session_state["extra_job_titles"] = [
                t.strip() for t in extra_titles.split(",") if t.strip()
            ]

            st.caption(
                "💡 **Tips for best results:**\n"
                "- Use **SerpApi** key for much better site:linkedin.com results\n"
                "- Use **Deep mode** for 50+ profiles\n"
                "- Results are LinkedIn profile URLs — click to view profiles\n"
                "- Refine with more specific job titles in the box above"
            )

# ── KEYWORD EXPANSION ───────────────────────────────────────────────────────
_current_prompt = prompt.strip() or st.session_state.get("prompt_value", "")
if _current_prompt and len(_current_prompt) > 10:
    with st.expander("🔍 Keyword Expansion — widen your search", expanded=False):
        st.caption(
            "The agent suggests related terms to find more results. "
            "Select the ones you want to add to your search."
        )
        if st.button("💡 Suggest keywords", key="expand_btn"):
            st.session_state["show_expansions"] = True

        if st.session_state.get("show_expansions"):
            with st.spinner("Generating keyword suggestions..."):
                from core.task_parser import parse_task_prompt as _parse_tmp
                _tmp_task = _parse_tmp(_current_prompt)
                _llm_tmp  = FreeLLMClient(
                    groq_api_key=st.session_state.get("groq_input",""),
                    gemini_api_key=st.session_state.get("gemini_input",""),
                )
                _expansions = expand_keywords(
                    topic=_tmp_task.industry or _current_prompt,
                    entity_type=(_tmp_task.target_entity_types or ["company"])[0],
                    task_type=_tmp_task.task_type,
                    geography=_tmp_task.geography.include_countries or [],
                    llm=_llm_tmp,
                )

            _selected_extra: list = []
            _all_cats = {
                "🔄 Synonyms":        _expansions.get("synonyms", []),
                "⚙️ Sub-sectors":     _expansions.get("sub_sectors", []),
                "🏢 Company types":   _expansions.get("company_types", []),
                "👤 Job titles":      _expansions.get("job_variants", []),
                "🌍 Geo variants":    _expansions.get("geo_variants", []),
                "📋 Industry codes":  _expansions.get("industry_codes", []),
            }
            for cat_idx, (cat_label, terms) in enumerate(_all_cats.items()):
                if terms:
                    st.write(f"**{cat_label}**")
                    cols = st.columns(min(4, len(terms)))
                    for i, term in enumerate(terms):
                        # Key = category_index + term_index → always unique
                        _key = f"kw_c{cat_idx}_t{i}"
                        if cols[i % len(cols)].checkbox(term, key=_key):
                            _selected_extra.append(term)

            st.session_state["selected_expansions"] = _selected_extra

            if _selected_extra:
                st.success(
                    f"✅ {len(_selected_extra)} keywords selected — "
                    f"will be added as extra search queries: "
                    f"{', '.join(_selected_extra[:5])}{'...' if len(_selected_extra) > 5 else ''}"
                )
            else:
                st.caption("☝️ Select keywords above to add them to your search.")


run_btn = st.button("🚀 Run Research Agent", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not prompt.strip():
        st.warning("Please enter a research request.")
        st.stop()

    # Build LLM client (free first)
    llm_client = FreeLLMClient(
        groq_api_key       = groq_key,
        gemini_api_key     = gemini_key,
        openrouter_api_key = openrouter_key,
        anthropic_api_key  = anthropic_key,
        openai_api_key     = openai_key,
    )
    backends = llm_client.available_backends()
    if backends:
        st.success(f"🤖 LLM active: {', '.join(backends)}")
    else:
        st.warning(
            "⚠️ No LLM key provided — running in keyword-only mode. "
            "Add a free Groq or Gemini key in the sidebar for much better results."
        )

    # Parse intent
    with st.spinner("🧠 Understanding your request..."):
        task_spec = parse_task_prompt_llm_first(prompt, llm=llm_client)

    # Apply UI overrides
    task_spec.mode        = mode
    task_spec.max_results = int(max_results)

    # Wire people search job levels from session state
    task_spec.job_levels = st.session_state.get("job_levels", ["engineer", "manager", "hr"])

    # Wire keyword expansions — append to prompt so query planner picks them up
    selected_expansions = st.session_state.get("selected_expansions", [])
    if selected_expansions:
        expansion_hint = " ".join(selected_expansions[:8])
        task_spec.raw_prompt = f"{task_spec.raw_prompt} [{expansion_hint}]"
        st.info(f"🔍 Adding {len(selected_expansions)} keyword(s): {', '.join(selected_expansions[:5])}")

    # Strict geo: only when constraints exist
    has_geo = bool(
        task_spec.geography.include_countries
        or task_spec.geography.exclude_countries
        or task_spec.geography.exclude_presence_countries
    )
    task_spec.geography.strict_mode = has_geo

    if task_override != "auto":
        task_spec.task_type = task_override
    if entity_override != "auto":
        task_spec.target_entity_types = [entity_override]
    if format_override != "auto":
        task_spec.output.format = format_override
    task_spec.output.filename = _normalize_filename(export_filename, task_spec.output.format)

    # Fields
    user_fields = []
    if field_website:  user_fields.append("website")
    if field_email:    user_fields.append("email")
    if field_phone:    user_fields.append("phone")
    if field_linkedin: user_fields.append("linkedin")
    if field_hq:       user_fields.append("hq_country")
    if field_presence: user_fields.append("presence_countries")
    if field_summary:  user_fields.append("summary")

    if task_spec.task_type == "document_research":
        task_spec.target_attributes = sorted(set(
            ["website", "summary", "author"] + user_fields
        ))
    else:
        task_spec.target_attributes = user_fields or ["website", "email", "phone"]

    # Auto-mode guidance
    effective_mode = mode
    if int(max_results) > 60 and mode == "Balanced":
        effective_mode = "Deep"
        st.info(
            f"ℹ️ You requested {int(max_results)} results — agent automatically using **Deep** mode "
            f"for maximum coverage (Balanced can only return ~30–60)."
        )
    elif int(max_results) > 30 and mode == "Fast":
        effective_mode = "Balanced"
        st.info(
            f"ℹ️ You requested {int(max_results)} results — agent automatically using **Balanced** mode "
            f"(Fast can only return ~15–20)."
        )

    # Verify task before running
    with st.expander("📋 What the agent understood — verify before running", expanded=True):
        _show_task_banner(task_spec.to_dict())
        if not task_spec.industry or len(task_spec.industry.strip()) < 2:
            st.error(
                "⚠️ Topic not detected. Rephrase to include the industry or subject, "
                "e.g. 'oil and gas software', 'renewable energy', 'electrical submersible pump'."
            )
            st.stop()

    provider_settings = ProviderSettings(
        use_ddg=use_ddg, use_exa=use_exa, use_tavily=use_tavily,
        use_serpapi=use_serpapi, use_firecrawl=use_firecrawl,
        use_llm_parser=bool(backends),
        use_uploaded_seed_dedupe=use_seed_dedupe,
    )

    budget_overrides = {
        "max_total_search_calls": max_total_calls,
        "max_ddg_calls":          max_ddg_calls,
        "max_exa_calls":          max_exa_calls,
        "max_tavily_calls":       max_tavily_calls,
        "max_serpapi_calls":      max_serpapi_calls,
        "max_pages_to_scrape":    max_pages,
    }

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
    log_box      = st.empty()
    STAGE_PCT    = {"DDG": 20, "Exa": 45, "Tavily": 65, "SerpApi": 75,
                    "dedup": 85, "LLM re-rank": 92, "Accepted": 98}

    def _progress(msg: str):
        log_box.caption(f"⏳ {msg}")
        for kw, pct in STAGE_PCT.items():
            if kw.lower() in msg.lower():
                progress_bar.progress(pct, text=f"{kw}...")
                break

    orchestrator = SearchOrchestrator()
    uploaded_df  = _read_uploaded_file(uploaded_file)

    with st.spinner(f"Running {mode} search..."):
        result = orchestrator.run_task(
            task_spec=task_spec,
            provider_settings=provider_settings,
            uploaded_df=uploaded_df,
            budget_overrides=budget_overrides,
            min_confidence_score=int(min_conf),
            user_keys=user_keys,
            progress_callback=_progress,
        )

    progress_bar.progress(100, text="Done!")
    log_box.empty()

    # Summary banner
    total = result["total_found"]
    raw   = result["raw_search_results"]
    rej   = len(result.get("rejected_records", []))
    b_used = result["budget"]

    if total > 0:
        if total < int(max_results) * 0.3 and int(max_results) > 20:
            st.warning(
                f"⚠️ {total} results accepted from {raw} raw. Target was {int(max_results)}. "
                f"Try: **Deep** mode, more provider keys (Exa + Tavily + SerpApi), "
                f"or lower the **Min confidence score**."
            )
        else:
            st.success(f"✅ {total} results accepted out of {raw} raw results.")
    else:
        st.error(
            f"❌ No results accepted ({raw} raw, {rej} rejected).\n\n"
            "**To get more results:**\n"
            "- Switch to **Deep** mode (handles 100+ result requests)\n"
            "- Add **Exa + Tavily** keys — together they add 3× more unique results\n"
            "- Add **SerpApi** key for additional diversity\n"
            "- Lower **Min confidence score** to 25\n"
            "- Rephrase to be more specific about the industry/sub-sector"
        )

    cols = st.columns(5)
    cols[0].metric("✅ Accepted",     total)
    cols[1].metric("❌ Rejected",     rej)
    cols[2].metric("🔍 Raw results",  raw)
    cols[3].metric("📞 Search calls", b_used["total_search_calls_used"])
    cols[4].metric("📄 Pages scraped", b_used["pages_scraped_used"])

    if result.get("llm_backends"):
        st.caption(f"🤖 LLM used: {', '.join(result['llm_backends'])}")

    # ── TABS + AUTO PAPER SUMMARY ───────────────────────────────────────────
    _is_paper_search = task_spec.task_type == "document_research"
    paper_records: list[CompanyRecord] = []
    paper_display_df = pd.DataFrame()
    paper_export_paths: dict[str, str] = {}
    paper_summary_report = ""
    paper_summary_error = ""
    paper_summary_enabled = _is_paper_search and auto_summarize_papers

    if _is_paper_search:
        paper_records = [CompanyRecord(**r) for r in result.get("records", [])]
        if paper_summary_enabled:
            try:
                from core.feynman_bridge import auto_summarize_and_export

                summary_status = st.empty()
                with st.spinner("🔬 Auto-summarizing papers and preparing research brief..."):
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
                        f"🧠 Automatic paper summaries ready · engine: Feynman · "
                        f"{sum(1 for p in paper_records if getattr(p, "notes", ""))} paper quick reviews generated"
                    )
                else:
                    st.success(
                        f"🧠 Automatic paper summaries ready · engine: built-in fallback · "
                        f"{sum(1 for p in paper_records if getattr(p, "notes", ""))} paper quick reviews generated"
                    )
                if paper_export_paths.get("pdf_error"):
                    st.warning(f"PDF note: {paper_export_paths['pdf_error']}")
            except Exception as exc:
                paper_summary_error = f"Automatic paper summarization failed: {exc}"
                st.warning(paper_summary_error)

        paper_display_df = _paper_rows_for_display(paper_records, result.get("records", []))
        if "Confidence" in paper_display_df.columns:
            paper_display_df = paper_display_df.sort_values("Confidence", ascending=False, na_position="last")

    if _is_paper_search:
        tab_accept, tab_summary, tab_download, tab_reject, tab_task, tab_log, tab_budget = st.tabs([
            "✅ Results", "🧠 Auto Summary", "⬇️ Downloads", "❌ Rejected", "📋 Task", "📜 Log", "💰 Budget",
        ])
    else:
        tab_accept, tab_reject, tab_task, tab_log, tab_budget = st.tabs([
            "✅ Results", "❌ Rejected", "📋 Task", "📜 Log", "💰 Budget",
        ])
        tab_summary = None
        tab_download = None

    with tab_accept:
        df = pd.DataFrame(result["records"])
        if _is_paper_search:
            if not paper_display_df.empty:
                st.caption("For paper searches, the results below include the automatic AI summary for each paper.")
                st.dataframe(paper_display_df, use_container_width=True, height=520)
            else:
                st.info("No paper records to display.")
        elif not df.empty:
            # Build display columns from what was requested
            if task_spec.task_type == "people_search":
                base = ["company_name", "job_title", "employer_name", "city",
                        "linkedin_url", "linkedin_profile", "description",
                        "confidence_score", "source_provider"]
            else:
                base = ["company_name", "website"]
                if "email"    in (task_spec.target_attributes or []): base.append("email")
                if "phone"    in (task_spec.target_attributes or []): base.append("phone")
                if "linkedin" in (task_spec.target_attributes or []): base.append("linkedin_url")
                if "hq_country" in (task_spec.target_attributes or []): base.append("hq_country")
                if "author" in (task_spec.target_attributes or []):   base += ["authors", "doi"]
                if "summary" in (task_spec.target_attributes or []):  base.append("description")
                base += ["confidence_score", "source_provider"]

            seen_cols: set = set()
            show_cols = []
            for c in base:
                if c in df.columns and c not in seen_cols:
                    seen_cols.add(c)
                    show_cols.append(c)

            sort_col = "confidence_score" if "confidence_score" in show_cols else None
            display_df = df[show_cols]
            if sort_col:
                display_df = display_df.sort_values(sort_col, ascending=False)
            st.dataframe(display_df, use_container_width=True, height=520)
        else:
            st.info("No accepted results. Check the Rejected tab.")

        ep = result.get("export_path", "")
        if (not _is_paper_search) and ep and Path(ep).exists():
            with open(ep, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {Path(ep).name}",
                    data=f, file_name=Path(ep).name,
                    mime="application/octet-stream",
                )

    if tab_summary is not None:
        with tab_summary:
            st.markdown("### 🧠 Automatic Paper Summary")
            if paper_summary_error:
                st.warning(paper_summary_error)
            elif not paper_summary_enabled:
                st.info("Automatic paper summarization is disabled in Advanced overrides.")
            else:
                if paper_export_paths.get("pdf"):
                    st.success("PDF research brief is ready to download.")
                elif paper_export_paths.get("markdown"):
                    st.info("Markdown research brief is ready. PDF was not created.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Papers found", len(paper_records))
                c2.metric("Quick reviews ready", sum(1 for p in paper_records if getattr(p, "notes", "")))
                c3.metric("PDF ready", "Yes" if paper_export_paths.get("pdf") else "No")

                if paper_summary_report:
                    st.markdown("#### Combined topic synthesis")
                    st.markdown(paper_summary_report)
                elif not paper_records:
                    st.info("No papers found to summarize.")
                else:
                    st.info("Per-paper summaries are attached to the results table. Combined synthesis was not generated.")

                if not paper_display_df.empty:
                    with st.expander("Paper summaries table", expanded=False):
                        st.dataframe(
                            paper_display_df[[c for c in ["Title", "Authors", "Year", "AI Summary"] if c in paper_display_df.columns]],
                            use_container_width=True,
                            height=420,
                        )

    if tab_download is not None:
        with tab_download:
            st.markdown("### ⬇️ Download paper outputs")
            st.caption("For paper searches, downloads include the AI-enriched paper table and the combined research brief.")

            dl_col1, dl_col2, dl_col3, dl_col4, dl_col5 = st.columns(5)
            export_df = paper_display_df.copy() if not paper_display_df.empty else pd.DataFrame()

            if not export_df.empty:
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
                    dl_col1.caption("Excel export unavailable")

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
            else:
                dl_col1.caption("No tabular paper results")
                dl_col2.caption("")
                dl_col3.caption("")

            md_bytes = _file_bytes(paper_export_paths.get("markdown", ""))
            if md_bytes:
                dl_col4.download_button(
                    "📝 Summary (.md)",
                    data=md_bytes,
                    file_name=Path(paper_export_paths["markdown"]).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                dl_col4.caption("No markdown summary")

            pdf_bytes = _file_bytes(paper_export_paths.get("pdf", ""))
            if pdf_bytes:
                dl_col5.download_button(
                    "📑 PDF brief",
                    data=pdf_bytes,
                    file_name=Path(paper_export_paths["pdf"]).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                dl_col5.caption("PDF not ready")

            if paper_export_paths.get("pdf_error"):
                st.warning(f"PDF export note: {paper_export_paths['pdf_error']}")

            if not export_df.empty:
                st.divider()
                st.markdown("**Preview of the AI-enriched download:**")
                st.dataframe(export_df.head(5), use_container_width=True)
                st.caption(f"Showing first 5 of {len(export_df)} rows.")

    with tab_reject:
        rdf = pd.DataFrame(result.get("rejected_records", []))
        if not rdf.empty:
            if "notes" in rdf.columns:
                rdf["reject_reason"] = (
                    rdf["notes"].astype(str)
                    .str.extract(r"rejected:([^\|]+)", expand=False)
                    .str.strip()
                )
            show_cols = [c for c in [
                "company_name","website","hq_country",
                "confidence_score","reject_reason","notes",
            ] if c in rdf.columns]
            st.dataframe(rdf[show_cols], use_container_width=True, height=400)

            if "reject_reason" in rdf.columns:
                EXPLAIN = {
                    "low_confidence":     "Score below threshold — irrelevant or low-quality page",
                    "directory_or_media": "It's a directory/list/news article, not a real company",
                    "geography_violation":"Company confirmed to be in an excluded country",
                    "duplicate":          "Already found from another source",
                }
                bd = rdf["reject_reason"].value_counts().reset_index()
                bd.columns = ["reason","count"]
                st.subheader("Rejection breakdown")
                for _, row in bd.iterrows():
                    key     = row["reason"].split("(")[0].strip()
                    explain = EXPLAIN.get(key, "")
                    st.write(f"• **{row['reason']}** ({row['count']}) — {explain}")
        else:
            st.info("No rejected records.")

    with tab_task:
        _show_task_banner(result["task_spec"])
        with st.expander("Full task JSON"):    st.json(result["task_spec"])
        with st.expander("Execution plan"):   st.json(result["execution_plan"])
        with st.expander("Generated queries"): st.json(result["queries"])

    with tab_log:
        for log in result["logs"]:
            if any(w in log.lower() for w in ["error","failed","exception"]):
                st.warning(f"⚠️ {log}")
            elif any(w in log for w in ["Stage","LLM","Accepted","Strategy"]):
                st.info(f"ℹ️ {log}")
            else:
                st.write(f"— {log}")

    with tab_budget:
        b = result["budget"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total calls",    f"{b['total_search_calls_used']}/{b['max_total_search_calls']}")
        c2.metric("DDG",            f"{b['ddg_calls_used']}/{b['max_ddg_calls']}")
        c3.metric("Exa",            f"{b['exa_calls_used']}/{b['max_exa_calls']}")
        c4.metric("Tavily",         f"{b['tavily_calls_used']}/{b['max_tavily_calls']}")
        c5.metric("SerpApi",        f"{b['serpapi_calls_used']}/{b['max_serpapi_calls']}")

        c6, c7 = st.columns(2)
        c6.metric("Pages scraped",  f"{b['pages_scraped_used']}/{b['max_pages_to_scrape']}")
        c7.metric("Pages remaining", b["remaining_pages"])

        with st.expander("API keys used (masked)"):
            st.json(result.get("resolved_keys", {}))
            st.caption("Keys are masked — never stored or logged.")
