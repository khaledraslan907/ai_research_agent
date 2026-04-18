"""
paper_summarizer.py
====================
Summarizes research papers using the free LLM already in the pipeline.
No extra API keys needed — uses Groq/Gemini/Ollama chain.

Each summary contains:
  - Plain-English 3-sentence summary
  - Key findings (bullet points)
  - Relevance to the search topic
"""
from __future__ import annotations
from typing import List, Dict, Optional
from core.models import CompanyRecord


SUMMARY_PROMPT = """You are a research assistant. Summarize this paper for a petroleum engineer.

Paper title: {title}
Authors: {authors}
Abstract: {abstract}
Search topic: {topic}

Write a summary with exactly this structure (keep it short):

**What it's about:** (1 sentence, plain English, no jargon)
**Key findings:** (2-3 bullet points, each under 15 words)
**Relevant because:** (1 sentence linking to "{topic}")

Be concise. No preamble."""


def summarize_papers(
    papers: List[CompanyRecord],
    topic: str,
    llm,
    max_papers: int = 20,
    progress_callback=None,
) -> List[Dict]:
    """
    Generate a plain-English summary for each paper using the free LLM.
    Returns list of dicts with: title, authors, doi, url, summary, error
    """
    results = []
    papers  = [p for p in papers if p.description and len(p.description) > 50][:max_papers]

    if not papers:
        return []

    for i, paper in enumerate(papers):
        if progress_callback:
            progress_callback(f"Summarizing {i+1}/{len(papers)}: {paper.company_name[:50]}...")

        title    = paper.company_name or "Untitled"
        authors  = paper.authors or "Unknown"
        abstract = (paper.description or "")[:1500]   # cap to save tokens
        doi      = paper.doi or paper.website or ""

        if not abstract.strip():
            results.append({
                "title":   title,
                "authors": authors,
                "doi":     doi,
                "summary": "No abstract available for this paper.",
                "error":   False,
            })
            continue

        prompt = SUMMARY_PROMPT.format(
            title=title, authors=authors,
            abstract=abstract, topic=topic,
        )

        try:
            summary = llm.generate(prompt, timeout=30) or ""
            summary = summary.strip()
            if not summary:
                summary = "Summary could not be generated (LLM returned empty response)."
            error = False
        except Exception as e:
            summary = f"Summary error: {str(e)[:100]}"
            error   = True

        results.append({
            "title":   title,
            "authors": authors,
            "doi":     doi,
            "summary": summary,
            "error":   error,
        })

    return results


def summaries_to_markdown(summaries: List[Dict], topic: str) -> str:
    """Convert summaries list to a downloadable Markdown document."""
    lines = [
        f"# Research Summaries: {topic}",
        f"*{len(summaries)} papers summarized*",
        "",
    ]
    for i, s in enumerate(summaries, 1):
        lines += [
            f"---",
            f"## {i}. {s['title']}",
        ]
        if s["authors"] and s["authors"] != "Unknown":
            lines.append(f"**Authors:** {s['authors']}")
        if s["doi"]:
            lines.append(f"**Source:** {s['doi']}")
        lines += ["", s["summary"], ""]
    return "\n".join(lines)


def summaries_to_text(summaries: List[Dict], topic: str) -> str:
    """Plain text version for simple download."""
    lines = [f"RESEARCH SUMMARIES: {topic.upper()}", "=" * 60, ""]
    for i, s in enumerate(summaries, 1):
        lines += [
            f"{i}. {s['title']}",
            f"   Authors: {s['authors']}",
            f"   Source:  {s['doi']}",
            "",
            f"   {s['summary'].replace(chr(10), chr(10)+'   ')}",
            "",
        ]
    return "\n".join(lines)
