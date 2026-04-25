from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import io
import re

from core.utils import extract_emails, extract_phones, unique_list, clean_text
from core.geography import find_countries_in_text, find_first_country_in_text

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None


@dataclass
class PDFExtractionResult:
    source: str = ""
    title: str = ""
    text: str = ""
    page_count: int = 0
    emails: List[str] = None
    phones: List[str] = None
    countries: List[str] = None
    first_country: str = ""
    urls: List[str] = None
    success: bool = False
    error: str = ""

    def __post_init__(self):
        self.emails = self.emails or []
        self.phones = self.phones or []
        self.countries = self.countries or []
        self.urls = self.urls or []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_URL_RE = re.compile(r"https?://[^\s<>()\]\[]+")


def _read_pdf_bytes(data: bytes):
    if PdfReader is not None:
        return PdfReader(io.BytesIO(data))
    if PyPDF2 is not None:
        return PyPDF2.PdfReader(io.BytesIO(data))
    raise RuntimeError("No PDF reader available. Install pypdf or PyPDF2.")


def extract_pdf_text_from_bytes(data: bytes, source: str = "") -> PDFExtractionResult:
    result = PDFExtractionResult(source=source)
    if not data:
        result.error = "Empty PDF bytes"
        return result
    try:
        reader = _read_pdf_bytes(data)
        pages = getattr(reader, "pages", []) or []
        result.page_count = len(pages)
        texts: List[str] = []
        for page in pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
        full_text = clean_text("\n".join(texts))
        result.text = full_text
        meta = getattr(reader, "metadata", None)
        if meta:
            try:
                result.title = clean_text(str(getattr(meta, "title", "") or meta.get("/Title", "")))
            except Exception:
                pass
        result.emails = unique_list(extract_emails(full_text))
        result.phones = unique_list(extract_phones(full_text))
        result.countries = unique_list(find_countries_in_text(full_text))
        result.first_country = find_first_country_in_text(full_text)
        result.urls = unique_list(_URL_RE.findall(full_text))
        result.success = True
        return result
    except Exception as exc:
        result.error = str(exc)
        return result


def extract_pdf_text(path: str | Path) -> PDFExtractionResult:
    path = Path(path)
    result = PDFExtractionResult(source=str(path))
    if not path.exists() or not path.is_file():
        result.error = "PDF path not found"
        return result
    try:
        return extract_pdf_text_from_bytes(path.read_bytes(), source=str(path))
    except Exception as exc:
        result.error = str(exc)
        return result
