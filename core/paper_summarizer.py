"""
paper_summarizer.py
====================
Enhanced research-paper summarizer with PDF export.
Safe for Streamlit Cloud import.
"""

from __future__ import annotations

import io
import re
from typing import Dict, List

from core.models import CompanyRecord


SUMMARY_PROMPT = """You are a research assistant helping a petroleum engineer review papers quickly.

Paper title: {title}
Authors: {authors}
Abstract / description: {abstract}
Search topic: {topic}

Write a concise but information-dense summary using exactly this structure:

**Problem studied:** (1 sentence — what technical problem the paper addresses)
**What they did:** (1-2 sentences — method, experiment, model, field trial, or review approach)
**Main findings:**
- (bullet 1: key result or conclusion)
- (bullet 2: key result or conclusion)
- (bullet 3: practical implication, limitation, or comparison)

**Methods / data:** (1 sentence — data type, simulation, field data, experiment, review, or model)
**Why it matters:** (1 sentence — why a petroleum engineer should care)

Rules:
- Be concrete, not generic
- Prefer technical meaning over marketing language
- If the abstract is vague, say what is clear and do not invent details
- No preamble
"""


_NON_PAPER_TITLE_HINTS = {
    "journal", "for authors", "articles in press", "volume", "issue",
    "homepage", "guide", "database", "submission", "editorial board",
    "table of contents",
}

_NON_PAPER_BODY_HINTS = {
    "submit your manuscript",
    "for authors",
    "aim and scope",
    "editorial board",
    "instructions for authors",
    "quarterly journal",
    "articles in press",
    "browse volumes",
    "journal homepage",
    "subscription database",
}


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _looks_like_non_paper(title: str, abstract: str, doi_or_url: str) -> bool:
    t = _norm_spaces((title or "").lower())
    a = _norm_spaces((abstract or "").lower())
    d = _norm_spaces((doi_or_url or "").lower())

    if any(h in t for h in _NON_PAPER_TITLE_HINTS):
        return True
    if any(h in a for h in _NON_PAPER_BODY_HINTS):
        return True
    if "for-authors" in d or "articles-in-press" in d:
        return True

    if len(a) < 80 and any(x in t for x in {"journal", "onepetro", "sciengine"}):
        return True

    return False


def summarize_papers(
    papers: List[CompanyRecord],
    topic: str,
    llm,
    max_papers: int = 20,
    progress_callback=None,
) -> List[Dict]:
    """
    Generate a detailed plain-English summary for each paper using the provided LLM client.
    Returns list of dicts with:
      title, authors, doi, summary, error, skipped
    """
    results: List[Dict] = []

    filtered_papers = [
        p for p in papers
        if getattr(p, "description", None) and len((p.description or "").strip()) > 50
    ][:max_papers]

    if not filtered_papers:
        return []

    for i, paper in enumerate(filtered_papers):
        title = (paper.company_name or "Untitled").strip()
        authors = (paper.authors or "Unknown").strip()
        abstract = (paper.description or "").strip()[:2500]
        doi = (paper.doi or paper.website or "").strip()

        if progress_callback:
            progress_callback(f"Summarizing {i + 1}/{len(filtered_papers)}: {title[:60]}...")

        if not abstract:
            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "summary": "No abstract or descriptive text was available for this result.",
                    "error": False,
                    "skipped": True,
                }
            )
            continue

        if _looks_like_non_paper(title, abstract, doi):
            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "summary": (
                        "Skipped: this result appears to be a journal page, portal page, "
                        "author-guide page, or other non-paper source rather than an individual paper."
                    ),
                    "error": False,
                    "skipped": True,
                }
            )
            continue

        prompt = SUMMARY_PROMPT.format(
            title=title,
            authors=authors,
            abstract=abstract,
            topic=topic,
        )

        try:
            summary = llm.generate(prompt, timeout=40) or ""
            summary = summary.strip()
            if not summary:
                summary = "Summary could not be generated (LLM returned empty response)."
            error = False
        except Exception as e:
            summary = f"Summary error: {str(e)[:200]}"
            error = True

        results.append(
            {
                "title": title,
                "authors": authors,
                "doi": doi,
                "summary": summary,
                "error": error,
                "skipped": False,
            }
        )

    return results


def summaries_to_text(summaries: List[Dict], topic: str) -> str:
    lines = [f"RESEARCH SUMMARIES: {topic.upper()}", "=" * 60, ""]

    for i, s in enumerate(summaries, 1):
        lines += [
            f"{i}. {s.get('title', 'Untitled')}",
            f"   Authors: {s.get('authors', '')}",
            f"   Source:  {s.get('doi', '')}",
            "",
            f"   {s.get('summary', '').replace(chr(10), chr(10) + '   ')}",
            "",
        ]

    return "\n".join(lines)


def summaries_to_pdf_bytes(summaries: List[Dict], topic: str) -> bytes | None:
    """
    Returns PDF bytes using reportlab.
    Safe fallback: returns None if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    except Exception:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    head_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    story = [
        Paragraph(f"Research Summaries: {topic}", title_style),
        Spacer(1, 10),
        Paragraph(f"{len(summaries)} results summarized", small_style),
        Spacer(1, 14),
    ]

    for i, s in enumerate(summaries, 1):
        story.append(Paragraph(f"{i}. {s.get('title', 'Untitled')}", head_style))
        if s.get("authors") and s["authors"] != "Unknown":
            story.append(Paragraph(f"<b>Authors:</b> {s['authors']}", small_style))
        if s.get("doi"):
            story.append(Paragraph(f"<b>Source:</b> {s['doi']}", small_style))

        summary_html = (
            s.get("summary", "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(Paragraph(summary_html, body_style))
        story.append(Spacer(1, 12))

        if i < len(summaries):
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()
