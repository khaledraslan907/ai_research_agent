from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from core.config import OUTPUT_DIR
from core.models import CompanyRecord


# Columns shown in export — order matters for readability
_COMPANY_COLUMNS = [
    "company_name", "website", "email", "phone", "linkedin_url",
    "hq_country", "country", "presence_countries",
    "description", "contact_page", "source_provider",
    "confidence_score", "page_type", "notes",
]

_PAPER_COLUMNS = [
    "company_name",   # used as paper title
    "website",        # paper URL / DOI link
    "authors",        # extracted authors
    "doi",            # DOI link
    "description",    # abstract / summary
    "source_provider",
    "confidence_score",
    "notes",
]


def _records_to_df(records: List[CompanyRecord], task_type: str = "") -> pd.DataFrame:
    rows = []
    for r in records:
        d = r.to_dict()
        d["presence_countries"] = ", ".join(d.get("presence_countries") or [])
        rows.append(d)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Choose column set based on task type
    if task_type == "document_research":
        cols = [c for c in _PAPER_COLUMNS if c in df.columns]
        # Rename company_name → title for papers
        df = df.rename(columns={"company_name": "title"})
        cols = ["title" if c == "company_name" else c for c in cols]
    else:
        cols = [c for c in _COMPANY_COLUMNS if c in df.columns]

    return df[[c for c in cols if c in df.columns]]


def export_records(
    records: List[CompanyRecord],
    output_format: str,
    filename: str,
    task_type: str = "",
) -> Optional[Path]:
    if not records:
        return None

    output_format = (output_format or "xlsx").lower()
    filename      = filename or f"results.{output_format}"

    # Ensure correct extension
    ext_map = {"xlsx": ".xlsx", "csv": ".csv", "json": ".json", "pdf": ".pdf"}
    ext = ext_map.get(output_format, ".xlsx")
    if not filename.lower().endswith(ext):
        filename = Path(filename).stem + ext

    path = OUTPUT_DIR / filename
    df   = _records_to_df(records, task_type=task_type)

    try:
        if output_format == "csv":
            df.to_csv(path, index=False, encoding="utf-8-sig")

        elif output_format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in records], f, indent=2, default=str)

        elif output_format == "pdf":
            _export_pdf(df, path, task_type=task_type)

        else:  # xlsx (default)
            _export_xlsx(df, path)

        return path

    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _export_xlsx(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        ws = writer.sheets["Results"]
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)


def _export_pdf(df: pd.DataFrame, path: Path, task_type: str = "") -> None:
    """Export results to PDF using ReportLab. Falls back to CSV if not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer, HRFlowable,
        )

        doc = SimpleDocTemplate(
            str(path),
            pagesize=landscape(A4),
            rightMargin=1*cm, leftMargin=1*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "AgentTitle",
            parent=styles["Title"],
            fontSize=16,
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "AgentSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#666666"),
            spaceAfter=12,
        )
        cell_style = ParagraphStyle(
            "Cell",
            parent=styles["Normal"],
            fontSize=7,
            leading=9,
        )

        elements: list = []

        # Title
        title = "Research Paper Results" if task_type == "document_research" else "Research Results"
        elements.append(Paragraph(title, title_style))
        elements.append(Paragraph(f"{len(df)} results exported", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        elements.append(Spacer(1, 0.3*cm))

        if task_type == "document_research":
            # For papers: one row per paper with full details
            _build_paper_pdf(elements, df, styles, cell_style)
        else:
            # For companies: table format
            _build_company_pdf(elements, df, cell_style)

        doc.build(elements)

    except ImportError:
        # ReportLab not installed → save as CSV instead
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        # Rename to .pdf so download works
        csv_path.rename(path)


def _build_paper_pdf(elements, df, styles, cell_style):
    """Build PDF content for paper/document results — one block per paper."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    paper_title_style = ParagraphStyle(
        "PaperTitle", parent=styles["Heading3"], fontSize=9, spaceAfter=2
    )
    meta_style = ParagraphStyle(
        "PaperMeta", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#444444"), spaceAfter=2
    )
    abstract_style = ParagraphStyle(
        "Abstract", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#222222"), leading=10, spaceAfter=8
    )

    title_col = "title" if "title" in df.columns else "company_name"

    for _, row in df.iterrows():
        title   = str(row.get(title_col, "") or "Untitled Paper")
        url     = str(row.get("website", "") or "")
        authors = str(row.get("authors", "") or "Authors not extracted")
        doi     = str(row.get("doi", "") or "")
        desc    = str(row.get("description", "") or "")[:600]
        score   = row.get("confidence_score", "")

        elements.append(Paragraph(title[:180], paper_title_style))
        meta_text = f"<b>Authors:</b> {authors}"
        if doi:
            meta_text += f"  |  <b>DOI:</b> {doi}"
        if url:
            meta_text += f"  |  <b>URL:</b> {url[:80]}"
        if score:
            meta_text += f"  |  Score: {score:.0f}/100"
        elements.append(Paragraph(meta_text, meta_style))
        if desc:
            elements.append(Paragraph(desc + "...", abstract_style))
        elements.append(Spacer(1, 0.1))


def _build_company_pdf(elements, df, cell_style):
    """Build PDF table for company results."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    # Pick readable columns
    show_cols = ["company_name", "website", "email", "phone", "hq_country",
                 "description", "confidence_score"]
    show_cols = [c for c in show_cols if c in df.columns]

    col_labels = {
        "company_name": "Company", "website": "Website", "email": "Email",
        "phone": "Phone", "hq_country": "HQ Country",
        "description": "Description", "confidence_score": "Score",
    }
    header = [col_labels.get(c, c) for c in show_cols]

    col_widths = {
        "company_name": 4.5, "website": 4.5, "email": 3.5, "phone": 2.5,
        "hq_country": 2.2, "description": 8, "confidence_score": 1.2,
    }
    from reportlab.lib.units import cm
    widths = [col_widths.get(c, 3) * cm for c in show_cols]

    # Build rows
    rows = [header]
    for _, row in df.iterrows():
        data_row = []
        for c in show_cols:
            val = str(row.get(c, "") or "")
            if c == "description":
                val = val[:120] + "..." if len(val) > 120 else val
            if c == "confidence_score":
                try:
                    val = f"{float(val):.0f}"
                except Exception:
                    pass
            data_row.append(Paragraph(val, cell_style))
        rows.append(data_row)

    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    elements.append(table)
