"""
feynman_bridge.py
==================
Integration between the AI Research Agent and Feynman
(https://feynman.is / https://github.com/getcompanion-ai/feynman).

HOW THE TWO SYSTEMS COMPLEMENT EACH OTHER
==========================================

  Your Agent:                         Feynman:
  ─────────────────────────────────   ───────────────────────────────────
  Broad multi-source discovery        Deep single-topic synthesis
  Geographic filtering                Claim verification against source
  People / company search             Peer review simulation
  Excel / PDF / CSV export            Literature review with consensus map
  Free API chain (Groq/Gemini)        Multi-agent (Researcher/Reviewer/
  25–150 papers found fast            Writer/Verifier)
                                      AlphaXiv + web search

INTEGRATION MODES
=================

Mode 1 — "Discover then Deep-Dive" (recommended)
  Your agent finds 20–100 papers on a topic (fast, broad).
  User selects the most relevant ones.
  Feynman runs /deepresearch or /lit on those titles.
  Your agent exports the combined output to Excel/PDF.

Mode 2 — "Feynman as a Research Provider"
  Feynman's CLI is called as a subprocess for each paper found.
  Its output (cited brief) is stored in the paper's description field.
  Exported to PDF with full synthesis.

Mode 3 — "Claim Verification Pass"
  After your agent finds papers, run Feynman /audit on each DOI.
  Flags papers where code doesn't match claimed results.
  Adds a "verified" / "unverified" / "mismatch" column to export.

INSTALLATION CHECK
==================
Run: feynman --version
If not installed: curl -fsSL https://feynman.is/install | bash

USAGE IN YOUR AGENT
===================
1. Run a document_research search as normal.
2. In the results tab, click "🔬 Deep Research with Feynman".
3. Choose: Literature Review / Deep Research / Claim Audit.
4. Feynman runs in background; results appear in a new tab.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.models import CompanyRecord


# ---------------------------------------------------------------------------
# Feynman installation helpers
# ---------------------------------------------------------------------------

def is_feynman_installed() -> bool:
    """Return True if the `feynman` CLI is reachable on PATH."""
    try:
        result = subprocess.run(
            ["feynman", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_feynman_version() -> str:
    try:
        r = subprocess.run(["feynman", "--version"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or r.stderr.strip() or "unknown"
    except Exception:
        return "not installed"


def install_feynman_command() -> str:
    """Return the shell command to install Feynman for the current platform."""
    if sys.platform == "win32":
        return "irm https://feynman.is/install.ps1 | iex"
    return "curl -fsSL https://feynman.is/install | bash"


# ---------------------------------------------------------------------------
# Build Feynman input from search results
# ---------------------------------------------------------------------------

def papers_to_feynman_context(
    papers: List[CompanyRecord],
    topic: str,
    max_papers: int = 20,
) -> str:
    """
    Convert a list of CompanyRecord paper objects into a Feynman-friendly
    context string. Feynman accepts natural language + DOI/URL references.

    Returns a multi-line string ready to pass to: feynman "<context>"
    """
    lines = [f"Research topic: {topic}", ""]
    lines.append("Key papers found by search agent:")
    lines.append("")

    for i, p in enumerate(papers[:max_papers], 1):
        title   = p.company_name or "Untitled"
        authors = p.authors or "Unknown authors"
        doi     = p.doi or ""
        url     = p.website or p.source_url or ""
        year    = p.publication_year or ""
        abstract_snippet = (p.description or "")[:200].strip()

        lines.append(f"{i}. {title}")
        if authors and authors != "Unknown authors":
            lines.append(f"   Authors: {authors}")
        if year:
            lines.append(f"   Year: {year}")
        ref = doi or url
        if ref:
            lines.append(f"   Source: {ref}")
        if abstract_snippet:
            lines.append(f"   Abstract: {abstract_snippet}...")
        lines.append("")

    return "\n".join(lines)


def papers_to_doi_list(papers: List[CompanyRecord]) -> List[str]:
    """Extract DOIs / URLs from papers for batch Feynman processing."""
    refs = []
    for p in papers:
        if p.doi:
            # Extract bare DOI (e.g. 10.1234/xyz) from URL
            m = re.search(r"(10\.\d{4,}/\S+)", p.doi)
            if m:
                refs.append(m.group(1).rstrip("."))
                continue
        if p.website and ("doi.org" in p.website or "arxiv.org" in p.website):
            refs.append(p.website)
        elif p.website:
            refs.append(p.website)
    return refs


# ---------------------------------------------------------------------------
# Feynman workflow runners
# ---------------------------------------------------------------------------

def run_feynman_lit_review(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run: feynman lit "<topic with paper context>"

    Feynman's /lit workflow produces a literature review with:
    - Consensus findings across papers
    - Open questions and gaps
    - Inline citations to source papers

    Returns dict with keys: success, output, error, command
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": "Feynman not installed. Run: curl -fsSL https://feynman.is/install | bash",
            "output": "",
            "command": "",
        }

    context = papers_to_feynman_context(papers, topic, max_papers=15)
    prompt  = f"Literature review on: {topic}\n\n{context}"

    return _run_feynman_command(
        ["feynman", "lit", prompt],
        timeout=timeout,
    )


def run_feynman_deep_research(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Run: feynman deepresearch "<topic>"

    Feynman's multi-agent /deepresearch workflow:
    1. Researcher agent hunts evidence
    2. Reviewer runs simulated peer review
    3. Writer drafts a cited research brief
    4. Verifier checks all citations

    Returns dict with: success, output, error, command
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": "Feynman not installed. Run: curl -fsSL https://feynman.is/install | bash",
            "output": "",
            "command": "",
        }

    context = papers_to_feynman_context(papers, topic, max_papers=10)
    prompt  = f"{topic}\n\nContext from preliminary search:\n{context}"

    return _run_feynman_command(
        ["feynman", "deepresearch", prompt],
        timeout=timeout,
    )


def run_feynman_audit(
    paper: CompanyRecord,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Run: feynman audit <doi_or_arxiv_id>

    Feynman's /audit workflow checks whether a paper's claims
    match its actual codebase. Useful for reproducibility checks.

    Returns dict with: success, output, error, is_mismatch, command
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": "Feynman not installed.",
            "output": "",
            "is_mismatch": False,
            "command": "",
        }

    # Extract arxiv ID or DOI
    ref = ""
    if paper.doi:
        m = re.search(r"(10\.\d{4,}/\S+)", paper.doi)
        if m:
            ref = m.group(1).rstrip(".")
    if not ref and paper.website:
        arxiv_m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)", paper.website)
        if arxiv_m:
            ref = arxiv_m.group(1)
    if not ref:
        ref = paper.website or paper.source_url or ""

    if not ref:
        return {
            "success": False,
            "error": f"No DOI or arXiv ID found for: {paper.company_name}",
            "output": "",
            "is_mismatch": False,
            "command": "",
        }

    result = _run_feynman_command(["feynman", "audit", ref], timeout=timeout)
    result["is_mismatch"] = "mismatch" in result.get("output", "").lower()
    return result


def run_feynman_review(
    topic: str,
    papers: List[CompanyRecord],
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Run: feynman review "<draft content>"

    Feynman's /review workflow performs simulated peer review with:
    - Severity-graded feedback (critical / major / minor)
    - A revision plan

    Returns dict with: success, output, error, severity_counts, command
    """
    if not is_feynman_installed():
        return {
            "success": False,
            "error": "Feynman not installed.",
            "output": "",
            "severity_counts": {},
            "command": "",
        }

    context = papers_to_feynman_context(papers, topic, max_papers=10)
    result  = _run_feynman_command(
        ["feynman", "review", f"Peer review this research on {topic}:\n\n{context}"],
        timeout=timeout,
    )

    # Parse severity counts from output
    output = result.get("output", "")
    result["severity_counts"] = {
        "critical": len(re.findall(r"(?i)critical", output)),
        "major":    len(re.findall(r"(?i)major",    output)),
        "minor":    len(re.findall(r"(?i)minor",    output)),
    }
    return result


def run_feynman_watch(
    topic: str,
    interval_hours: int = 24,
) -> Dict[str, Any]:
    """
    Run: feynman watch "<topic>"

    Sets up a recurring monitor that alerts when new papers appear.
    Useful for ongoing research tracking.
    """
    if not is_feynman_installed():
        return {"success": False, "error": "Feynman not installed.", "output": "", "command": ""}

    return _run_feynman_command(
        ["feynman", "watch", topic, "--interval", str(interval_hours)],
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Enrich paper records with Feynman synthesis
# ---------------------------------------------------------------------------

def enrich_papers_with_feynman(
    papers: List[CompanyRecord],
    topic: str,
    mode: str = "lit",
    progress_callback=None,
) -> Tuple[List[CompanyRecord], str]:
    """
    Run Feynman on the found papers and enrich each record's description
    with Feynman's synthesis. Also returns the full Feynman report.

    Args:
        papers:            List of paper CompanyRecord objects
        topic:             Research topic string
        mode:              "lit" | "deepresearch" | "audit" | "review"
        progress_callback: Optional callable(str) for progress updates

    Returns:
        (enriched_papers, full_feynman_report_text)
    """
    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    if not is_feynman_installed():
        _log("⚠️ Feynman not installed — skipping deep synthesis")
        return papers, ""

    _log(f"🔬 Running Feynman {mode} on {len(papers)} papers...")

    if mode == "lit":
        result = run_feynman_lit_review(topic, papers)
    elif mode == "deepresearch":
        result = run_feynman_deep_research(topic, papers)
    elif mode == "review":
        result = run_feynman_review(topic, papers)
    elif mode == "audit":
        # Audit each paper individually for code/claim mismatches
        for paper in papers:
            audit = run_feynman_audit(paper)
            if audit["success"]:
                tag = "⚠️ MISMATCH" if audit.get("is_mismatch") else "✅ VERIFIED"
                paper.notes = f"{tag} | {paper.notes or ''}"
                _log(f"  {tag}: {paper.company_name[:50]}")
        return papers, "Audit complete — see notes column"
    else:
        return papers, ""

    if not result["success"]:
        _log(f"⚠️ Feynman error: {result.get('error', 'unknown')}")
        return papers, result.get("error", "")

    report = result.get("output", "")
    _log(f"✅ Feynman synthesis complete ({len(report)} chars)")

    # Attach a short excerpt of the synthesis to the first paper's description
    # as a "synthesis note" — the full report is returned separately
    if papers and report:
        excerpt = report[:500].strip()
        papers[0].notes = f"[Feynman synthesis] {excerpt}..."

    return papers, report


# ---------------------------------------------------------------------------
# Internal subprocess runner
# ---------------------------------------------------------------------------

def _run_feynman_command(
    cmd: List[str],
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run a Feynman CLI command and capture output."""
    env_full = {**os.environ, **(env or {})}

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env_full,
        )
        success = proc.returncode == 0
        output  = (proc.stdout or "").strip()
        error   = (proc.stderr or "").strip() if not success else ""

        return {
            "success":  success,
            "output":   output,
            "error":    error,
            "command":  " ".join(cmd[:3]) + " ...",
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success":  False,
            "output":   "",
            "error":    f"Feynman timed out after {timeout}s",
            "command":  " ".join(cmd[:3]) + " ...",
            "returncode": -1,
        }
    except FileNotFoundError:
        return {
            "success":  False,
            "output":   "",
            "error":    "feynman command not found — install with: curl -fsSL https://feynman.is/install | bash",
            "command":  " ".join(cmd[:3]) + " ...",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success":  False,
            "output":   "",
            "error":    str(e),
            "command":  " ".join(cmd[:3]) + " ...",
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Integration guide (printed when module is run directly)
# ---------------------------------------------------------------------------

INTEGRATION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║           FEYNMAN + AI RESEARCH AGENT — INTEGRATION GUIDE               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  STEP 1 — Install Feynman (one-time)                                     ║
║    curl -fsSL https://feynman.is/install | bash                          ║
║                                                                          ║
║  STEP 2 — Run your agent to find papers (as normal)                      ║
║    Prompt: "Find papers about ESP electrical submersible pump"           ║
║    → Agent finds 20–80 papers with titles, authors, DOIs                 ║
║                                                                          ║
║  STEP 3 — In Results tab, click "🔬 Deep Research with Feynman"          ║
║    Choose a workflow:                                                    ║
║    • Literature Review  — consensus + open questions (feynman lit)       ║
║    • Deep Research      — full multi-agent synthesis (feynman deepres.)  ║
║    • Peer Review        — severity-graded critique (feynman review)      ║
║    • Claim Audit        — code vs claim check (feynman audit)            ║
║                                                                          ║
║  STEP 4 — Export                                                         ║
║    Combined output: papers list + Feynman report → PDF/Excel             ║
║                                                                          ║
║  WHAT EACH AGENT DOES:                                                   ║
║    Researcher  → Searches papers, web, repos, docs for evidence          ║
║    Reviewer    → Simulated peer review with severity scores              ║
║    Writer      → Structures notes into cited research brief              ║
║    Verifier    → Checks every citation URL, removes dead links           ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
    print(f"Feynman installed: {is_feynman_installed()}")
    print(f"Feynman version:   {get_feynman_version()}")
