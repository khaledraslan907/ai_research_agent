from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from actions.extract_contacts import extract_contacts_from_records
from actions.extract_locations import extract_locations_from_records
from core.evidence import attach_evidence_to_record, text_evidence
from core.models import CrawlResult


@dataclass
class EnrichmentOutput:
    records: List[Any] = field(default_factory=list)
    enriched_count: int = 0
    skipped_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enriched_count': self.enriched_count,
            'skipped_count': self.skipped_count,
            'errors': list(self.errors),
        }


class EnrichmentPipeline:
    """Lightweight enrichment using website scraping and structured extraction."""

    def __init__(self, scraper=None, structured_extractor=None, max_records: int = 25):
        self.scraper = scraper
        self.structured_extractor = structured_extractor
        self.max_records = max_records

    def run(self, records: Iterable[Any], follow_internal_links: bool = True) -> EnrichmentOutput:
        out_records: List[Any] = []
        errors: List[str] = []
        enriched_count = 0
        skipped_count = 0

        for idx, record in enumerate(records or []):
            if idx >= self.max_records:
                out_records.append(record)
                skipped_count += 1
                continue

            website = getattr(record, 'website', '') or getattr(record, 'source_url', '')
            if not website or not self.scraper:
                out_records.append(record)
                skipped_count += 1
                continue

            try:
                crawl = self.scraper.scrape(website, follow_internal_links=follow_internal_links)
                if getattr(crawl, 'success', False):
                    self._apply_crawl(record, crawl)
                    enriched_count += 1
                else:
                    errors.append(f"{website}: {getattr(crawl, 'error', 'crawl_failed')}")
            except Exception as exc:
                errors.append(f"{website}: {exc}")

            out_records.append(record)

        return EnrichmentOutput(records=out_records, enriched_count=enriched_count, skipped_count=skipped_count, errors=errors)

    def _apply_crawl(self, record: Any, crawl: CrawlResult) -> None:
        if crawl.emails and not getattr(record, 'email', ''):
            try:
                record.email = crawl.emails[0]
            except Exception:
                pass
        if crawl.phones and not getattr(record, 'phone', ''):
            try:
                record.phone = crawl.phones[0]
            except Exception:
                pass
        if crawl.contact_links and not getattr(record, 'contact_page', ''):
            try:
                record.contact_page = crawl.contact_links[0]
            except Exception:
                pass
        if crawl.detected_company_name and (not getattr(record, 'company_name', '')):
            try:
                record.company_name = crawl.detected_company_name
            except Exception:
                pass
        if crawl.detected_hq_country and not getattr(record, 'hq_country', ''):
            try:
                record.hq_country = crawl.detected_hq_country
            except Exception:
                pass
        if crawl.detected_country and not getattr(record, 'country', ''):
            try:
                record.country = crawl.detected_country
            except Exception:
                pass
        if crawl.detected_presence_countries:
            existing = list(getattr(record, 'presence_countries', []) or [])
            merged = list(dict.fromkeys(existing + list(crawl.detected_presence_countries or [])))
            try:
                record.presence_countries = merged
            except Exception:
                pass

        snippets: List[str] = []
        if crawl.meta_description:
            snippets.append(crawl.meta_description)
        if crawl.text:
            snippets.append(crawl.text[:500])
        if snippets:
            attach_evidence_to_record(record, [text_evidence(' '.join(snippets), getattr(record, 'website', ''), 'crawl', 'crawl')])

        if self.structured_extractor:
            html = getattr(crawl, '_raw_html', '') or ''
            if html:
                try:
                    structured = self.structured_extractor.extract_from_html(html, getattr(record, 'website', ''))
                    emails = list(structured.get('emails', []) or [])
                    phones = list(structured.get('phones', []) or [])
                    if emails and not getattr(record, 'email', ''):
                        record.email = emails[0]
                    if phones and not getattr(record, 'phone', ''):
                        record.phone = phones[0]
                    if structured.get('linkedin_url') and not getattr(record, 'linkedin_url', ''):
                        record.linkedin_url = structured.get('linkedin_url', '')
                    if structured.get('hq_country') and not getattr(record, 'hq_country', ''):
                        record.hq_country = structured.get('hq_country', '')
                    if structured.get('company_name') and not getattr(record, 'company_name', ''):
                        record.company_name = structured.get('company_name', '')
                except Exception:
                    pass
