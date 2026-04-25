from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from actions.extract_authors import extract_authors_from_records
from actions.summarize import summarize_records


@dataclass
class SummarizationOutput:
    summaries: List[str] = field(default_factory=list)
    records: List[Any] = field(default_factory=list)
    entity_type: str = 'company'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity_type,
            'summary_count': len(self.summaries),
            'metadata': dict(self.metadata),
        }


class SummarizationPipeline:
    """Applies summarization and light post-processing by entity type."""

    def __init__(self, llm=None, max_sentences: int = 2):
        self.llm = llm
        self.max_sentences = max_sentences

    def run(self, records: Iterable[Any], entity_type: str = 'company') -> SummarizationOutput:
        rows = list(records or [])
        et = (entity_type or 'company').lower().strip()

        if et == 'paper':
            authors_map = extract_authors_from_records(rows)
            for idx, record in enumerate(rows):
                if idx < len(authors_map):
                    authors = authors_map[idx]
                    if authors and not getattr(record, 'authors', ''):
                        try:
                            record.authors = ', '.join(authors)
                        except Exception:
                            pass

        summaries = summarize_records(rows, entity_type=et, llm=self.llm, max_sentences=self.max_sentences)
        for record, summary in zip(rows, summaries):
            if summary:
                try:
                    if not getattr(record, 'notes', ''):
                        record.notes = summary
                except Exception:
                    pass

        metadata = {
            'entity_type': et,
            'used_llm': bool(self.llm and getattr(self.llm, 'is_available', lambda: False)()),
            'max_sentences': self.max_sentences,
        }
        return SummarizationOutput(summaries=summaries, records=rows, entity_type=et, metadata=metadata)
