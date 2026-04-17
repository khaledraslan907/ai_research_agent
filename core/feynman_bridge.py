
"""
feynman_bridge.py
==================
Automatic paper summarization + optional PDF export bridge between
your AI Research Agent and Feynman.

TARGET WORKFLOW
===============
1. Your agent finds papers.
2. This bridge automatically summarizes the top papers one by one.
3. It also produces a combined literature review / deep-research report.
4. It can export the result package to Markdown or PDF.

NOTES
=====
- On Windows this bridge calls the standalone Feynman launcher directly:
    %LOCALAPPDATA%\Programs\feynman\bin\feynman.cmd
  This avoids PATH / local-Node issues.
- For paper-by-paper summaries, the bridge uses `feynman chat "<prompt>"`.
- For topic synthesis, it uses `feynman lit "<topic>"` or
  `feynman deepresearch "<topic>"`.
- For PDF export, it tries reportlab first. If reportlab is unavailable,
  it falls back to writing Markdown and returns an informative error.

RECOMMENDED APP FLOW
====================
After your search agent returns paper records, call:

    enriched_papers, synthesis_report, export_paths = auto_summarize_and_export(
        papers=found_papers,
        topic=user_prompt,
        export_dir="outputs",
        export_pdf=True,
        per_paper_limit=5,
        synthesis_mode="lit",
    )

You can then show:
- each paper.notes  -> short paper summary
- synthesis_report  -> combined topic report
- export_paths      -> files written to disk
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from collections import Counter
from textwrap import wrap
from typing import Any, Dict, List, Optional, Tuple

from core.models import CompanyRecord


# ---------------------------------------------------------------------------
# Small safety helpers
# ---------------------------------------------------------------------------

def _safe_get(obj: Any, attr: str, default: str = "") -> str:
    """Safe getattr that always returns a string-like value or default."""
    try:
        value = getattr(obj, attr, default)
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def _truncate(text: str, n: int = 3000) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def _slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len].strip("-") or "report"


def _extract_doi(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.I)
    return m.group(1).rstrip(").,;") if m else ""


def _extract_arxiv_id(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", text, re.I)
    return m.group(1) if m else ""


def _best_reference(paper: CompanyRecord) -> str:
    """Choose the best available machine-usable reference for a paper."""
    website = _safe_get(paper, "website")
    source_url = _safe_get(paper, "source_url")
    doi = _safe_get(paper, "doi")

    arxiv_id = _extract_arxiv_id(website) or _extract_arxiv_id(source_url) or _extract_arxiv_id(doi)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    doi_value = _extract_doi(doi) or _extract_doi(website) or _extract_doi(source_url)
    if doi_value:
        return f"https://doi.org/{doi_value}"

    return website or source_url or doi


def _paper_title(paper: CompanyRecord) -> str:
    return (
        _safe_get(paper, "company_name")
        or _safe_get(paper, "title")
        or "Untitled Paper"
    )


def _paper_authors(paper: CompanyRecord) -> str:
    return _safe_get(paper, "authors", "Unknown authors")


def _paper_year(paper: CompanyRecord) -> str:
    for attr in ("publication_year", "year"):
        value = _safe_get(paper, attr)
        if value:
            return value
    return ""


def _paper_abstract(paper: CompanyRecord) -> str:
    for attr in ("description", "abstract", "summary"):
        value = _safe_get(paper, attr)
        if value:
            return value
    return ""


# ---------------------------------------------------------------------------
# Built-in fallback summarizer
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _split_sentences(text: str) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _top_keywords(texts: List[str], limit: int = 8) -> List[str]:
    stop = {
        "the","and","for","with","from","that","this","into","their","were","was","are","using",
        "use","used","study","paper","research","results","result","based","analysis","data","method",
        "methods","model","models","system","systems","approach","approaches","application","applications",
        "electrical","submersible","pump","pumps","esp","review","reviews","between","within","under",
        "over","than","into","onto","about","have","has","had","been","being","also","such","can",
        "could","should","would","may","might","our","your","these","those","they","them","its","it's",
        "abstract","context","authors","year"
    }
    tokens = []
    for text in texts:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
            if tok not in stop and not tok.isdigit():
                tokens.append(tok)
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(limit)]



def _builtin_single_paper_summary(topic: str, paper: CompanyRecord) -> str:
    title = _paper_title(paper)
    abstract = _paper_abstract(paper)
    sentences = _split_sentences(abstract)
    first = sentences[0] if sentences else "No abstract was available, so this summary is based on the paper title and metadata."
    second = sentences[1] if len(sentences) > 1 else ""
    third = sentences[2] if len(sentences) > 2 else ""
    year = _paper_year(paper)
    authors = _paper_authors(paper)
    ref = _best_reference(paper)

    lines = [
        f"Plain-English summary: {first}"
    ]
    if second:
        lines.append(f"Method / context: {second}")
    if third:
        lines.append(f"Key takeaway: {third}")

    if not second and abstract:
        kws = _top_keywords([title, abstract], limit=4)
        if kws:
            lines.append(f"Likely focus areas: {', '.join(kws)}.")

    lines.append(f"Why it matters for {topic}: This paper appears relevant to the topic because it addresses {topic} directly or provides supporting technical context.")
    if authors and authors != 'Unknown authors':
        lines.append(f"Authors: {authors}.")
    if year:
        lines.append(f"Year: {year}.")
    if ref:
        lines.append(f"Reference: {ref}")

    return "\n".join(lines).strip()



def _builtin_topic_synthesis(topic: str, papers: List[CompanyRecord]) -> str:
    if not papers:
        return "No papers were available for synthesis."

    texts = []
    titles = []
    with_abstract = 0
    years = []
    for paper in papers:
        title = _paper_title(paper)
        abstract = _paper_abstract(paper)
        titles.append(title)
        texts.append(title)
        if abstract:
            texts.append(abstract)
            with_abstract += 1
        year = _paper_year(paper)
        if year:
            years.append(year)

    keywords = _top_keywords(texts, limit=6)
    sample_titles = "; ".join(titles[:3])
    date_span = ""
    if years:
        cleaned = [y for y in years if re.fullmatch(r"\d{4}", str(y).strip())]
        if cleaned:
            date_span = f"Publication years represented: {min(cleaned)} to {max(cleaned)}."

    lines = [
        f"This automated synthesis covers {len(papers)} papers related to {topic}.",
        f"Most frequent themes in the retrieved set: {', '.join(keywords) if keywords else 'topic-specific technical concepts' }.",
        f"{with_abstract} of the {len(papers)} papers included abstract or context text that could be summarized directly.",
    ]
    if date_span:
        lines.append(date_span)
    lines.append(f"Representative papers include: {sample_titles}.")
    lines.append("Common gaps: some records may be missing full abstracts, benchmark details, or implementation details, so this synthesis should be treated as a rapid briefing rather than a full systematic review.")
    lines.append("Recommended next step: open the top-ranked papers first, especially those with full abstracts, DOI links, or recent publication years.")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Feynman executable helpers
# ---------------------------------------------------------------------------

def _get_feynman_executable() -> str:
    """
    Resolve the standalone Feynman launcher first.

    On Windows we prefer:
      %LOCALAPPDATA%\Programs\feynman\bin\feynman.cmd

    This avoids PATH issues and avoids forcing local Node.
    """
    candidates: List[Optional[str]] = []

    if sys.platform == "win32":
        candidates.extend([
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\feynman\bin\feynman.cmd"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\feynman\bin\feynman.ps1"),
            shutil.which("feynman.cmd"),
            shutil.which("feynman"),
        ])
    else:
        candidates.extend([
            shutil.which("feynman"),
            os.path.expanduser("~/.local/bin/feynman"),
        ])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return ""


def is_feynman_installed() -> bool:
    exe = _get_feynman_executable()
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_feynman_version() -> str:
    exe = _get_feynman_executable()
    if not exe:
        return "not installed"
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        return (result.stdout or result.stderr).strip() or "unknown"
    except Exception:
        return "not installed"


def install_feynman_command() -> str:
    if sys.platform == "win32":
        return "irm https://feynman.is/install.ps1 | iex"
    return "curl -fsSL https://feynman.is/install | bash"


# ---------------------------------------------------------------------------
# Feynman subprocess runner
# ---------------------------------------------------------------------------

def _run_feynman_command(
    args: List[str],
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run Feynman without forcing a local Node runtime.

    Important:
    - We DO NOT set FEYNMAN_NODE.
    - We call the standalone launcher directly.
    """
    exe = _get_feynman_executable()
    if not exe:
        return {
            "success": False,
            "output": "",
            "error": f"Feynman not installed. Run: {install_feynman_command()}",
            "command": "",
            "returncode": -1,
        }

    normalized = args[1:] if args and args[0].lower() == "feynman" else args

    try:
        proc = subprocess.run(
            [exe, *normalized],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            shell=False,
        )
        return {
            "success": proc.returncode == 0,
            "output": (proc.stdout or "").strip(),
            "error": (proc.stderr or "").strip(),
            "command": " ".join([Path(exe).name, *normalized[:3]]) + (" ..." if len(normalized) > 3 else ""),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Feynman timed out after {timeout}s",
            "command": " ".join(normalized[:3]) + (" ..." if len(normalized) > 3 else ""),
            "returncode": -1,
        }
    except Exception as exc:
        return {
            "success": False,
            "output": "",
            "error": str(exc),
            "command": " ".join(normalized[:3]) + (" ..." if len(normalized) > 3 else ""),
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Prompt / context builders
# ---------------------------------------------------------------------------

def papers_to_feynman_context(
    papers: List[CompanyRecord],
    topic: str,
    max_papers: int = 20,
) -> str:
    """
    Convert found papers into a compact context block.

    This is used for topic-level workflows like lit / deepresearch.
    """
    lines = [f"Research topic: {topic}", "", "Key papers found by the search agent:", ""]

    for i, paper in enumerate(papers[:max_papers], 1):
        title = _paper_title(paper)
        authors = _paper_authors(paper)
        year = _paper_year(paper)
        ref = _best_reference(paper)
        abstract = _truncate(_paper_abstract(paper), 700)

        lines.append(f"{i}. {title}")
        if authors:
            lines.append(f"   Authors: {authors}")
        if year:
            lines.append(f"   Year: {year}")
        if ref:
            lines.append(f"   Source: {ref}")
        if abstract:
            lines.append(f"   Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)


def _build_per_paper_summary_prompt(topic: str, paper: CompanyRecord) -> str:
    """
    Build a deterministic structured prompt for one paper summary.
    """
    title = _paper_title(paper)
    authors = _paper_authors(paper)
    year = _paper_year(paper)
    ref = _best_reference(paper)
    abstract = _truncate(_paper_abstract(paper), 2500)

    return f"""
Summarize the following research paper for the topic: {topic}

Return your answer in this exact structure:

Title:
One-line relevance:
Plain-English summary:
Main contribution:
Method / model:
Data / benchmark:
Key findings:
Limitations:
Practical relevance:
Confidence note:

Paper title: {title}
Authors: {authors}
Year: {year}
Reference: {ref}
Abstract / context:
{abstract}
""".strip()


def _build_review_markdown(topic: str, papers: List[CompanyRecord], max_papers: int = 10) -> str:
    """
    Build a markdown artifact in case you want to run `feynman review`
    against a local file.
    """
    lines = [f"# Research packet for: {topic}", ""]
    for i, paper in enumerate(papers[:max_papers], 1):
        lines.append(f"## {i}. {_paper_title(paper)}")
        lines.append(f"- Authors: {_paper_authors(paper)}")
        year = _paper_year(paper)
        if year:
            lines.append(f"- Year: {year}")
        ref = _best_reference(paper)
        if ref:
            lines.append(f"- Reference: {ref}")
        abstract = _paper_abstract(paper)
        if abstract:
            lines.append("")
            lines.append("### Abstract / Context")
            lines.append(abstract)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Feynman workflow wrappers
# ---------------------------------------------------------------------------

def run_feynman_paper_summary(
    topic: str,
    paper: CompanyRecord,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Summarize one paper.

    Uses:
        feynman chat "<structured prompt>"
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": f"Feynman not installed. Run: {install_feynman_command()}",
            "output": "",
            "command": "",
        }

    prompt = _build_per_paper_summary_prompt(topic, paper)
    return _run_feynman_command(["feynman", "chat", prompt], timeout=timeout)


def run_feynman_lit_review(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 420,
) -> Dict[str, Any]:
    """
    Run topic-level literature review.

    Uses:
        feynman lit "<topic + context>"
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": f"Feynman not installed. Run: {install_feynman_command()}",
            "output": "",
            "command": "",
        }

    context = papers_to_feynman_context(papers, topic, max_papers=12)
    prompt = (
        f"Literature review on: {topic}\n\n"
        "Prioritize the papers listed below first. "
        "Write consensus findings, disagreements, gaps, and open questions.\n\n"
        f"{context}"
    )
    return _run_feynman_command(["feynman", "lit", prompt], timeout=timeout)


def run_feynman_deep_research(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 900,
) -> Dict[str, Any]:
    """
    Run topic-level deep research.

    Uses:
        feynman deepresearch "<topic + context>"
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": f"Feynman not installed. Run: {install_feynman_command()}",
            "output": "",
            "command": "",
        }

    context = papers_to_feynman_context(papers, topic, max_papers=10)
    prompt = (
        f"Deep research on: {topic}\n\n"
        "Start from these papers found by my upstream search agent. "
        "Prioritize them in the synthesis before broadening to external evidence.\n\n"
        f"{context}"
    )
    return _run_feynman_command(["feynman", "deepresearch", prompt], timeout=timeout)


def run_feynman_review(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run a peer review on a generated local markdown artifact.

    Uses:
        feynman review <local_markdown_file>
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": f"Feynman not installed. Run: {install_feynman_command()}",
            "output": "",
            "severity_counts": {},
            "command": "",
        }

    import tempfile

    review_text = _build_review_markdown(topic, papers, max_papers=10)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            temp_path = tmp.name
            tmp.write(review_text)

        result = _run_feynman_command(["feynman", "review", temp_path], timeout=timeout)
        output = result.get("output", "")
        result["severity_counts"] = {
            "critical": len(re.findall(r"(?i)\bcritical\b", output)),
            "major": len(re.findall(r"(?i)\bmajor\b", output)),
            "minor": len(re.findall(r"(?i)\bminor\b", output)),
            "nit": len(re.findall(r"(?i)\bnit\b", output)),
        }
        return result
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def run_feynman_audit(
    paper: CompanyRecord,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run code/claim audit where possible.

    Best inputs:
    - arxiv:<id>
    - GitHub repo URL

    If neither exists, skip gracefully.
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": "Feynman not installed.",
            "output": "",
            "is_mismatch": False,
            "command": "",
        }

    website = _safe_get(paper, "website")
    source_url = _safe_get(paper, "source_url")
    doi = _safe_get(paper, "doi")

    ref = ""
    arxiv_id = _extract_arxiv_id(website) or _extract_arxiv_id(source_url) or _extract_arxiv_id(doi)
    if arxiv_id:
        ref = arxiv_id
    elif "github.com/" in website:
        ref = website
    elif "github.com/" in source_url:
        ref = source_url

    if not ref:
        return {
            "success": False,
            "error": f"Audit needs arXiv ID or GitHub URL for: {_paper_title(paper)}",
            "output": "",
            "is_mismatch": False,
            "command": "",
        }

    result = _run_feynman_command(["feynman", "audit", ref], timeout=timeout)
    result["is_mismatch"] = "mismatch" in result.get("output", "").lower()
    return result


# ---------------------------------------------------------------------------
# Main automatic summarization entry point
# ---------------------------------------------------------------------------

def enrich_papers_with_feynman(
    papers: List[CompanyRecord],
    topic: str,
    mode: str = "paper_summaries",
    progress_callback=None,
    per_paper_limit: int = 5,
    include_global_synthesis: bool = True,
    synthesis_mode: str = "lit",
) -> Tuple[List[CompanyRecord], str]:
    """
    Main function to enrich search results automatically.

    Default behavior:
    - summarize top N papers individually into paper.notes
    - optionally generate one combined synthesis report

    Args:
        papers: list of CompanyRecord objects representing papers
        topic: search topic
        mode:
            "paper_summaries" -> summarize papers one by one
            "lit"             -> only literature review
            "deepresearch"    -> only deep research
            "review"          -> review a local packet
            "audit"           -> audit each paper where possible
        progress_callback: optional status callback
        per_paper_limit: number of papers to summarize individually
        include_global_synthesis: whether to also generate one combined report
        synthesis_mode: "lit" or "deepresearch"

    Returns:
        enriched_papers, synthesis_report
    """
    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if not papers:
        _log("⚠️ No papers found to summarize.")
        return papers, ""

    use_feynman = is_feynman_installed()
    if use_feynman:
        _log("✅ Using Feynman for paper summaries.")
    else:
        _log("ℹ️ Feynman not installed — using built-in summarizer fallback.")

    synthesis_report = ""

    if mode == "audit":
        _log(f"🔍 Running Feynman audit on {len(papers)} papers...")
        for paper in papers:
            audit = run_feynman_audit(paper)
            if audit["success"]:
                tag = "⚠️ MISMATCH" if audit.get("is_mismatch") else "✅ VERIFIED"
                existing = _safe_get(paper, "notes")
                paper.notes = f"{tag}\n{existing}".strip()
                _log(f"  {tag}: {_paper_title(paper)[:60]}")
            else:
                paper.notes = f"[Audit skipped] {audit['error']}"
        return papers, "Audit complete."

    if mode in {"paper_summaries", "summaries"}:
        limit = min(len(papers), max(1, per_paper_limit))
        engine_label = "Feynman" if use_feynman else "built-in summarizer"
        _log(f"🔬 Summarizing {limit} papers with {engine_label}...")
        for idx, paper in enumerate(papers[:limit], 1):
            if use_feynman:
                result = run_feynman_paper_summary(topic, paper)
                if result["success"] and result.get("output"):
                    paper.notes = result["output"]
                    _log(f"  ✅ summarized {idx}/{limit}: {_paper_title(paper)[:60]}")
                else:
                    paper.notes = _builtin_single_paper_summary(topic, paper)
                    _log(f"  ⚠️ Feynman failed, used fallback for {idx}/{limit}: {_paper_title(paper)[:60]}")
            else:
                paper.notes = _builtin_single_paper_summary(topic, paper)
                _log(f"  ✅ summarized {idx}/{limit}: {_paper_title(paper)[:60]}")

        for paper in papers[limit:]:
            if not _safe_get(paper, "notes"):
                paper.notes = ""

        if include_global_synthesis:
            _log(f"🧠 Building combined {synthesis_mode} report...")
            if use_feynman:
                if synthesis_mode == "deepresearch":
                    synth = run_feynman_deep_research(topic, papers)
                else:
                    synth = run_feynman_lit_review(topic, papers)

                if synth["success"] and synth.get("output"):
                    synthesis_report = synth.get("output", "")
                    _log("✅ Combined synthesis complete.")
                else:
                    synthesis_report = _builtin_topic_synthesis(topic, papers)
                    _log("⚠️ Combined synthesis fallback used.")
            else:
                synthesis_report = _builtin_topic_synthesis(topic, papers)
                _log("✅ Combined synthesis complete.")

        return papers, synthesis_report

    if mode == "lit":
        if use_feynman:
            result = run_feynman_lit_review(topic, papers)
            return papers, result.get("output", "") if result["success"] else _builtin_topic_synthesis(topic, papers)
        return papers, _builtin_topic_synthesis(topic, papers)

    if mode == "deepresearch":
        if use_feynman:
            result = run_feynman_deep_research(topic, papers)
            return papers, result.get("output", "") if result["success"] else _builtin_topic_synthesis(topic, papers)
        return papers, _builtin_topic_synthesis(topic, papers)

    if mode == "review":
        if use_feynman:
            result = run_feynman_review(topic, papers)
            return papers, result.get("output", "") if result["success"] else _builtin_topic_synthesis(topic, papers)
        return papers, _builtin_topic_synthesis(topic, papers)

    return papers, synthesis_report


# ---------------------------------------------------------------------------
# Report rendering + export
# ---------------------------------------------------------------------------

def _build_markdown_report(
    topic: str,
    papers: List[CompanyRecord],
    synthesis_report: str = "",
) -> str:
    """
    Build a nicely structured markdown report.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = [
        f"# Research Summary Report",
        "",
        f"**Topic:** {topic}",
        f"**Generated:** {now}",
        f"**Papers included:** {len(papers)}",
        "",
    ]

    if synthesis_report:
        lines.extend([
            "## Combined Topic Synthesis",
            "",
            synthesis_report.strip(),
            "",
        ])

    lines.extend([
        "## Paper-by-Paper Summaries",
        "",
    ])

    for i, paper in enumerate(papers, 1):
        title = _paper_title(paper)
        authors = _paper_authors(paper)
        year = _paper_year(paper)
        ref = _best_reference(paper)
        abstract = _truncate(_paper_abstract(paper), 1200)
        notes = _safe_get(paper, "notes")

        lines.append(f"### {i}. {title}")
        lines.append("")
        lines.append(f"- **Authors:** {authors}")
        if year:
            lines.append(f"- **Year:** {year}")
        if ref:
            lines.append(f"- **Reference:** {ref}")
        lines.append("")

        if notes:
            lines.append("**AI Summary**")
            lines.append("")
            lines.append(notes.strip())
            lines.append("")

        if abstract:
            lines.append("**Original Abstract / Context**")
            lines.append("")
            lines.append(abstract)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _markdown_to_simple_html(markdown_text: str, title: str = "Research Summary") -> str:
    """
    Very lightweight markdown-ish to HTML conversion.
    Enough for browser viewing or future HTML export.
    """
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, Helvetica, sans-serif; margin: 40px; line-height: 1.55; color: #222; }",
        "    h1, h2, h3 { color: #0f172a; }",
        "    pre { white-space: pre-wrap; font-family: inherit; }",
        "    .meta { color: #475569; }",
        "    .block { margin-bottom: 18px; }",
        "    hr { border: none; border-top: 1px solid #cbd5e1; margin: 24px 0; }",
        "  </style>",
        "</head>",
        "<body>",
    ]

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            html_lines.append("<div class='block'></div>")
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("- **"):
            html_lines.append(f"<div class='meta'>{escape(line)}</div>")
        else:
            html_lines.append(f"<p>{escape(line)}</p>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def export_research_summary_markdown(
    topic: str,
    papers: List[CompanyRecord],
    synthesis_report: str = "",
    output_dir: str = "outputs",
    base_filename: Optional[str] = None,
) -> str:
    """
    Export combined research summary to Markdown.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = base_filename or f"research_summary_{_slugify(topic)}"
    md_path = out_dir / f"{slug}.md"
    md_path.write_text(
        _build_markdown_report(topic, papers, synthesis_report),
        encoding="utf-8",
    )
    return str(md_path)


def export_research_summary_pdf(
    topic: str,
    papers: List[CompanyRecord],
    synthesis_report: str = "",
    output_dir: str = "outputs",
    base_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export combined research summary to PDF.

    Strategy:
    1. Try reportlab.
    2. If reportlab is missing, also save Markdown and return an informative error.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = base_filename or f"research_summary_{_slugify(topic)}"
    pdf_path = out_dir / f"{slug}.pdf"
    md_path = out_dir / f"{slug}.md"

    markdown_text = _build_markdown_report(topic, papers, synthesis_report)
    md_path.write_text(markdown_text, encoding="utf-8")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        left = 2.0 * cm
        right = 2.0 * cm
        top = height - 2.0 * cm
        bottom = 2.0 * cm
        usable_width = width - left - right

        def draw_wrapped_paragraph(text: str, y: float, font_name: str = "Helvetica", font_size: int = 10,
                                   leading: int = 14, bold: bool = False) -> float:
            nonlocal c
            current_font = "Helvetica-Bold" if bold else font_name
            c.setFont(current_font, font_size)

            if not text:
                return y - leading

            words = text.split()
            if not words:
                return y - leading

            line = ""
            lines = []
            for word in words:
                test_line = f"{line} {word}".strip()
                if stringWidth(test_line, current_font, font_size) <= usable_width:
                    line = test_line
                else:
                    if line:
                        lines.append(line)
                    line = word
            if line:
                lines.append(line)

            for ln in lines:
                if y < bottom + leading:
                    c.showPage()
                    y = top
                    c.setFont(current_font, font_size)
                c.drawString(left, y, ln)
                y -= leading

            return y - 4

        y = top
        c.setTitle(f"Research Summary - {topic}")

        y = draw_wrapped_paragraph("Research Summary Report", y, font_size=16, leading=20, bold=True)
        y = draw_wrapped_paragraph(f"Topic: {topic}", y, font_size=11, leading=15)
        y = draw_wrapped_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", y, font_size=10, leading=14)
        y = draw_wrapped_paragraph(f"Papers included: {len(papers)}", y, font_size=10, leading=14)
        y -= 8

        if synthesis_report:
            y = draw_wrapped_paragraph("Combined Topic Synthesis", y, font_size=13, leading=18, bold=True)
            for para in synthesis_report.splitlines():
                if para.strip():
                    y = draw_wrapped_paragraph(para.strip(), y, font_size=10, leading=14)

        y = draw_wrapped_paragraph("Paper-by-Paper Summaries", y, font_size=13, leading=18, bold=True)

        for i, paper in enumerate(papers, 1):
            y = draw_wrapped_paragraph(f"{i}. {_paper_title(paper)}", y, font_size=12, leading=16, bold=True)
            y = draw_wrapped_paragraph(f"Authors: {_paper_authors(paper)}", y)
            year = _paper_year(paper)
            if year:
                y = draw_wrapped_paragraph(f"Year: {year}", y)
            ref = _best_reference(paper)
            if ref:
                y = draw_wrapped_paragraph(f"Reference: {ref}", y)

            notes = _safe_get(paper, "notes")
            if notes:
                y = draw_wrapped_paragraph("AI Summary:", y, font_size=10, leading=14, bold=True)
                for para in notes.splitlines():
                    if para.strip():
                        y = draw_wrapped_paragraph(para.strip(), y, font_size=10, leading=14)

            abstract = _truncate(_paper_abstract(paper), 1000)
            if abstract:
                y = draw_wrapped_paragraph("Original Abstract / Context:", y, font_size=10, leading=14, bold=True)
                for para in abstract.splitlines():
                    if para.strip():
                        y = draw_wrapped_paragraph(para.strip(), y, font_size=10, leading=14)

            y -= 8

        c.save()
        return {
            "success": True,
            "pdf_path": str(pdf_path),
            "markdown_path": str(md_path),
            "error": "",
        }

    except ImportError:
        return {
            "success": False,
            "pdf_path": "",
            "markdown_path": str(md_path),
            "error": "reportlab is not installed. Run: pip install reportlab",
        }
    except Exception as exc:
        return {
            "success": False,
            "pdf_path": "",
            "markdown_path": str(md_path),
            "error": f"PDF export failed: {exc}",
        }


def auto_summarize_and_export(
    papers: List[CompanyRecord],
    topic: str,
    export_dir: str = "outputs",
    export_pdf: bool = True,
    per_paper_limit: int = 5,
    synthesis_mode: str = "lit",
    progress_callback=None,
) -> Tuple[List[CompanyRecord], str, Dict[str, str]]:
    """
    High-level one-call helper for your app.

    This is the function you likely want to call after the search step.

    Steps:
    1. Summarize papers automatically
    2. Generate one combined synthesis report
    3. Export markdown
    4. Optionally export PDF

    Returns:
        enriched_papers, synthesis_report, export_paths
    """
    papers, synthesis_report = enrich_papers_with_feynman(
        papers=papers,
        topic=topic,
        mode="paper_summaries",
        progress_callback=progress_callback,
        per_paper_limit=per_paper_limit,
        include_global_synthesis=True,
        synthesis_mode=synthesis_mode,
    )

    export_paths: Dict[str, str] = {"summary_engine": "feynman" if is_feynman_installed() else "built-in"}

    md_path = export_research_summary_markdown(
        topic=topic,
        papers=papers,
        synthesis_report=synthesis_report,
        output_dir=export_dir,
    )
    export_paths["markdown"] = md_path

    if export_pdf:
        pdf_result = export_research_summary_pdf(
            topic=topic,
            papers=papers,
            synthesis_report=synthesis_report,
            output_dir=export_dir,
        )
        if pdf_result.get("markdown_path"):
            export_paths["markdown"] = pdf_result["markdown_path"]
        if pdf_result.get("pdf_path"):
            export_paths["pdf"] = pdf_result["pdf_path"]
        if pdf_result.get("error"):
            export_paths["pdf_error"] = pdf_result["error"]

    return papers, synthesis_report, export_paths


# ---------------------------------------------------------------------------
# Optional UI helper
# ---------------------------------------------------------------------------

def build_export_preview_text(
    topic: str,
    papers: List[CompanyRecord],
    synthesis_report: str = "",
) -> str:
    """
    Small plain-text preview for UI tabs or debug logs.
    """
    lines = [
        f"Topic: {topic}",
        f"Papers: {len(papers)}",
        "",
    ]

    if synthesis_report:
        lines.append("Combined synthesis:")
        lines.append(_truncate(synthesis_report, 1200))
        lines.append("")

    lines.append("Paper summaries:")
    for i, paper in enumerate(papers, 1):
        title = _paper_title(paper)
        note = _safe_get(paper, "notes")
        lines.append(f"{i}. {title}")
        lines.append(_truncate(note, 500) if note else "[No summary]")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# CLI info when run directly
# ---------------------------------------------------------------------------

INTEGRATION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║      FEYNMAN + AI RESEARCH AGENT — AUTO SUMMARY + PDF EXPORT            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ONE-CALL FLOW                                                           ║
║    1) agent finds papers                                                 ║
║    2) auto_summarize_and_export(...)                                     ║
║    3) paper.notes gets paper summaries                                   ║
║    4) one combined synthesis report is generated                         ║
║    5) markdown + optional PDF are written to disk                        ║
║                                                                          ║
║  MAIN FUNCTION                                                           ║
║    auto_summarize_and_export(                                            ║
║        papers=found_papers,                                              ║
║        topic=user_prompt,                                                ║
║        export_dir="outputs",                                             ║
║        export_pdf=True,                                                  ║
║        per_paper_limit=5,                                                ║
║        synthesis_mode="lit",                                             ║
║    )                                                                     ║
║                                                                          ║
║  REQUIREMENTS                                                            ║
║    - Feynman installed and set up                                        ║
║    - Optional for PDF: reportlab                                         ║
║      pip install reportlab                                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
    print(f"Feynman installed: {is_feynman_installed()}")
    print(f"Feynman version:   {get_feynman_version()}")
