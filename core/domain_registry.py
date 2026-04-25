from __future__ import annotations

"""
domain_registry.py
==================
Selects domain packs and provides domain-aware hints for query planning,
validation, and ranking.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from core.ontology import (
    INDUSTRY_ALIASES,
    OIL_GAS_DOMAIN_TERMS,
    GENERIC_SOLUTION_TERMS,
    infer_industries,
    normalize_label,
)


@dataclass
class DomainPack:
    name: str
    match_terms: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    entity_bias: List[str] = field(default_factory=list)
    target_category_bias: List[str] = field(default_factory=list)
    solution_keywords: List[str] = field(default_factory=list)
    domain_keywords: List[str] = field(default_factory=list)
    trusted_domains: List[str] = field(default_factory=list)
    query_hints: List[str] = field(default_factory=list)
    source_hints: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])

    def matches(self, text: str) -> int:
        low = normalize_label(text)
        score = 0
        for term in self.match_terms:
            t = normalize_label(term)
            if t and t in low:
                score += 1
        for ind in self.industries:
            i = normalize_label(ind)
            if i and i in low:
                score += 2
        return score


BUILTIN_PACKS: Dict[str, DomainPack] = {
    "generic": DomainPack(
        name="generic",
        match_terms=["company", "paper", "people", "profile", "vendor", "supplier", "research"],
        industries=list(INDUSTRY_ALIASES.keys()),
        source_hints=["official_site", "directory", "linkedin", "pdf"],
        languages=["en", "ar"],
    ),
    "oil_gas": DomainPack(
        name="oil_gas",
        match_terms=INDUSTRY_ALIASES["oil and gas"] + [term for vals in OIL_GAS_DOMAIN_TERMS.values() for term in vals[:2]],
        industries=["oil and gas", "energy"],
        entity_bias=["company", "paper", "person", "event", "tender"],
        target_category_bias=["service_company", "software_company", "operator"],
        solution_keywords=list(GENERIC_SOLUTION_TERMS.keys()),
        domain_keywords=list(OIL_GAS_DOMAIN_TERMS.keys()),
        trusted_domains=["onepetro.org", "slb.com", "halliburton.com", "bakerhughes.com"],
        query_hints=["operator vendor list", "exhibitor", "approved vendor", "oilfield service", "wireline", "well logging"],
        source_hints=["official_site", "conference", "vendor_list", "tender", "pdf"],
        languages=["en", "ar"],
    ),
    "academia": DomainPack(
        name="academia",
        match_terms=INDUSTRY_ALIASES["academia"] + ["doi", "abstract", "author", "citation", "journal"],
        industries=["academia"],
        entity_bias=["paper", "person", "organization", "dataset"],
        target_category_bias=["general"],
        trusted_domains=["sciencedirect.com", "springer.com", "wiley.com", "arxiv.org", "onepetro.org"],
        query_hints=["pdf", "doi", "site:scholar.google.com", "conference proceedings", "journal"],
        source_hints=["publisher", "repository", "scholar", "pdf"],
        languages=["en"],
    ),
    "software": DomainPack(
        name="software",
        match_terms=INDUSTRY_ALIASES["software"] + ["saas", "platform", "api", "cloud", "developer"],
        industries=["software", "technology"],
        entity_bias=["company", "product", "person"],
        target_category_bias=["software_company"],
        solution_keywords=list(GENERIC_SOLUTION_TERMS.keys()),
        query_hints=["b2b software", "platform", "product", "pricing", "case study"],
        source_hints=["official_site", "product_page", "documentation", "linkedin"],
        languages=["en"],
    ),
    "manufacturing": DomainPack(
        name="manufacturing",
        match_terms=INDUSTRY_ALIASES["manufacturing"] + ["factory", "plant", "mes", "automation"],
        industries=["manufacturing"],
        entity_bias=["company", "product", "event"],
        target_category_bias=["manufacturer", "software_company", "service_company"],
        query_hints=["factory automation", "industrial software", "mes", "quality control"],
        source_hints=["official_site", "directory", "trade_show"],
        languages=["en"],
    ),
    "healthcare": DomainPack(
        name="healthcare",
        match_terms=INDUSTRY_ALIASES["healthcare"] + ["hospital", "medical device", "clinical", "pharma"],
        industries=["healthcare"],
        entity_bias=["company", "paper", "person", "organization"],
        target_category_bias=["general", "software_company", "manufacturer"],
        query_hints=["clinical study", "medical device", "hospital software", "pharma company"],
        source_hints=["official_site", "publisher", "registry"],
        languages=["en", "ar"],
    ),
    "energy": DomainPack(
        name="energy",
        match_terms=INDUSTRY_ALIASES["energy"] + ["renewable", "solar", "wind", "power grid"],
        industries=["energy"],
        entity_bias=["company", "paper", "event", "tender"],
        target_category_bias=["service_company", "software_company", "manufacturer"],
        query_hints=["renewable energy", "power systems", "grid analytics", "energy storage"],
        source_hints=["official_site", "conference", "tender", "pdf"],
        languages=["en", "ar"],
    ),
}


def list_domain_packs() -> List[str]:
    return list(BUILTIN_PACKS.keys())


def get_domain_pack(name: str) -> DomainPack:
    return BUILTIN_PACKS.get(name, BUILTIN_PACKS["generic"])


def detect_domain_packs(text: str, explicit_industry: str = "") -> List[DomainPack]:
    scores: List[tuple[int, DomainPack]] = []
    combined = f"{text or ''} {explicit_industry or ''}"
    industries = set(infer_industries(combined))
    for pack in BUILTIN_PACKS.values():
        score = pack.matches(combined)
        if any(i in pack.industries for i in industries):
            score += 2
        if score > 0:
            scores.append((score, pack))
    scores.sort(key=lambda x: (-x[0], x[1].name))
    if not scores:
        return [BUILTIN_PACKS["generic"]]
    ordered = [p for _, p in scores]
    if ordered[0].name != "generic":
        ordered.append(BUILTIN_PACKS["generic"])
    # dedupe by name
    out: List[DomainPack] = []
    seen = set()
    for pack in ordered:
        if pack.name in seen:
            continue
        seen.add(pack.name)
        out.append(pack)
    return out


def primary_domain_pack(text: str = "", explicit_industry: str = "") -> DomainPack:
    return detect_domain_packs(text, explicit_industry)[0]


def merge_domain_hints(packs: Iterable[DomainPack]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "industries": [],
        "solution_keywords": [],
        "domain_keywords": [],
        "trusted_domains": [],
        "query_hints": [],
        "source_hints": [],
        "languages": [],
    }
    seen = {k: set() for k in out}
    for pack in packs:
        for key in out:
            values = getattr(pack, key, []) if hasattr(pack, key) else []
            for value in values or []:
                s = str(value).strip()
                if not s or s in seen[key]:
                    continue
                seen[key].add(s)
                out[key].append(s)
    return out
