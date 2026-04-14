"""
task_parser.py
==============
Universal intent parser — handles any sector, any entity type, any geography.

Covers:
- Companies in any industry (not just oil & gas)
- US states as geography signals ("in Texas" → include USA)
- Research papers, people, events, products
- Any output format, any field collection
- All geographic aliases (states, provinces, cities, regions)
"""
from __future__ import annotations

import re
from typing import List, Optional

from core.task_models import TaskSpec, GeographyRules, OutputSpec, CredentialMode
from core.geography import (
    all_country_names, expand_region_name, normalize_country_name,
    find_countries_in_text, US_STATES, CANADIAN_PROVINCES, AUSTRALIAN_STATES,
    CITY_TO_COUNTRY, REGION_ALIASES,
)


# ── Entity type hints ──────────────────────────────────────────────────────
ENTITY_HINTS = {
    "company": [
        "company", "companies", "vendor", "vendors", "firm", "firms",
        "provider", "providers", "contractor", "contractors", "operator", "operators",
        "supplier", "suppliers", "startup", "startups", "corporation", "corporations",
        "enterprise", "enterprises", "agency", "agencies",
    ],
    "person": [
        "person", "people", "founder", "ceo", "manager", "contact person",
        "researcher", "expert", "consultant", "director", "executive",
        "scientist", "engineer", "professor",
    ],
    "paper": [
        "paper", "papers", "journal", "study", "studies", "publication",
        "publications", "article", "articles", "report", "reports",
        "thesis", "theses", "preprint", "research",
    ],
    "organization": [
        "organization", "organisation", "association", "society",
        "institute", "foundation", "ngo", "consortium", "alliance", "university",
    ],
    "event": [
        "event", "conference", "expo", "summit", "workshop", "forum",
        "symposium", "webinar", "congress", "trade show",
    ],
    "product": [
        "product", "products", "tool", "tools", "platform", "solution",
        "solutions", "app", "application", "software", "system",
    ],
}

ATTRIBUTE_HINTS = {
    "website":            ["website", "site", "url", "homepage", "link", "web"],
    "email":              ["email", "emails", "mail", "contact email", "e-mail"],
    "phone":              ["phone", "telephone", "tel", "mobile", "contact number",
                           "phone number", "contact details", "contact info",
                           "contact information", "contact"],
    "linkedin":           ["linkedin"],
    "hq_country":         ["hq", "headquarters", "head office", "based in", "headquartered"],
    "presence_countries": ["branches", "offices", "presence", "locations", "regional"],
    "summary":            ["summary", "overview", "description", "abstract", "bio"],
    "author":             ["author", "authors", "written by", "authored by",
                           "who wrote", "researcher", "researchers"],
    "pdf":                ["pdf", "full text", "download"],
}

FORMAT_HINTS = {
    "xlsx": ["excel", "xlsx", "spreadsheet", "xls"],
    "csv":  ["csv"],
    "pdf":  [
        "pdf report", "as pdf", "in pdf", "pdf file", "export pdf",
        "as a pdf", "pdf format", "pdf output", "pdf document", "export as pdf",
        "save as pdf", "output pdf", "export to pdf", "to pdf",
        "in pdf file", "as pdf file",
    ],
    "json": ["json"],
}

GENERIC_STOPWORDS = {
    "find", "show", "get", "search", "look", "for", "me", "please", "about",
    "inside", "outside", "working", "operates", "operating", "with", "without",
    "and", "or", "in", "on", "to", "export", "into", "as", "from", "of", "the",
    "a", "an", "related", "companies", "company", "papers", "paper",
    "organizations", "organization", "events", "event", "products", "product",
    "people", "person", "list", "all", "some", "any", "give", "that", "are",
    "i", "want", "need", "can", "you", "help", "also", "then", "provide",
    "their", "its", "which", "where", "when", "how", "what", "who",
    "each", "every", "per", "this", "these", "those", "have", "has",
    "do", "does", "did", "is", "was", "been", "being", "be",
    # People-search noise — never part of the industry topic
    "linkedin", "profiles", "profile", "account", "accounts",
    "engineers", "managers", "manager", "engineer", "specialist",
    "specialists", "director", "directors", "hr", "executives", "executive",
    "professionals", "professional", "employees", "staff",
    "personnel", "team", "teams",
}

_REQUEST_NOISE = {
    "link", "links", "title", "titles", "author", "authors",
    "doi", "journal", "year", "date", "abstract", "volume",
    "issue", "page", "pages", "citation", "reference", "references",
    "number", "numbers", "contact", "details", "info", "information",
    # Output-format words — never part of topic
    "pdf", "csv", "excel", "xlsx", "xls", "json",
    "file", "files", "export", "download", "output", "format",
    "pdf file", "pdf format", "pdf report", "as pdf", "in pdf",
    "report", "document", "summary",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_entity_types(prompt_lower: str) -> List[str]:
    found = []
    for entity_type, hints in ENTITY_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            found.append(entity_type)

    # Don't add "person" just because user mentioned "author" when searching for papers
    if "paper" in found and "person" in found:
        explicit_person = any(w in prompt_lower for w in [
            "people", "researcher", "expert", "consultant", "ceo", "founder", "director",
        ])
        if not explicit_person:
            found.remove("person")

    return found or ["company"]


def _extract_output_format(prompt_lower: str) -> str:
    for fmt, hints in FORMAT_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            return fmt
    # Catch bare standalone "pdf" anywhere in prompt
    if re.search(r"\bpdf\b", prompt_lower):
        return "pdf"
    return "xlsx"


def _extract_task_type(prompt_lower: str) -> str:
    if any(x in prompt_lower for x in [
        "enrich", "append", "fill missing", "complete list",
        "add email", "add phone", "enrich list", "update list",
    ]):
        return "entity_enrichment"
    if any(x in prompt_lower for x in [
        "similar companies", "similar vendors", "like these",
        "similar to", "alternatives to", "companies like",
    ]):
        return "similar_entity_expansion"
    if any(x in prompt_lower for x in [
        "market map", "landscape", "market research", "industry map",
        "competitive landscape", "market overview",
    ]):
        return "market_research"
    if any(x in prompt_lower for x in [
        "paper", "papers", "study", "studies", "publication", "publications",
        "article", "articles", "report", "reports", "journal", "thesis",
        "research on", "literature",
    ]):
        return "document_research"
    # People / LinkedIn search detection
    if any(x in prompt_lower for x in [
        "linkedin", "linkedin profile", "linkedin account",
        "people who work", "people working", "professionals in",
        "find engineers", "find managers", "find hr", "find people",
        "employees in", "staff in", "workers in",
    ]):
        return "people_search"
    return "entity_discovery"


def _extract_target_category(prompt_lower: str) -> str:
    if any(x in prompt_lower for x in [
        "service company", "service companies", "contractor", "contractors",
        "field service", "services provider", "engineering services",
        "oilfield services", "drilling services", "field services",
        "maintenance", "inspection", "testing", "consulting services",
    ]):
        return "service_company"
    if any(x in prompt_lower for x in [
        "software", "digital", "analytics", "automation",
        "technology provider", "technology vendors", "platform",
        "saas", "cloud", "ai company", "data company", "tech company",
        "app", "application", "mobile app",
    ]):
        return "software_company"
    return "general"


# ── Geography extraction ───────────────────────────────────────────────────

# Build a comprehensive set of all subnational names for matching
_ALL_SUBNATIONAL = {}
_ALL_SUBNATIONAL.update(US_STATES)
_ALL_SUBNATIONAL.update(CANADIAN_PROVINCES)
_ALL_SUBNATIONAL.update(AUSTRALIAN_STATES)

# All region names
_ALL_REGIONS = set(REGION_ALIASES.keys())


def _extract_grouped_geo(prompt_lower: str, marker_patterns: List[str]) -> List[str]:
    """
    Find countries after a geo marker phrase like 'in', 'outside', etc.
    Handles: countries, regions, US states, cities.
    """
    found: List[str] = []

    # Build sorted list of all geo tokens (longest first for greedy matching)
    all_geo_tokens = (
        sorted(all_country_names() + list(_ALL_REGIONS), key=len, reverse=True)
    )

    for marker in marker_patterns:
        for item in all_geo_tokens:
            m = re.search(
                rf"{marker}\s+{re.escape(item)}"
                rf"(?:\s*(?:,|and|or)\s*([a-zA-Z\s,\-]+))?",
                prompt_lower,
            )
            if not m:
                continue

            names = [item]
            tail = m.group(1) or ""
            if tail:
                # find_countries_in_text handles states, cities, aliases
                tail_countries = find_countries_in_text(tail)
                names.extend(tail_countries)

            for n in names:
                expanded = expand_region_name(n)
                if expanded:
                    found.extend(expanded)
                else:
                    canonical = normalize_country_name(n)
                    if canonical:
                        found.append(canonical)

    return sorted(set(found))


def _extract_geography(prompt_lower: str) -> GeographyRules:
    """
    Full geographic extraction — handles any prompt mentioning:
    - Country names and aliases
    - Regions (Europe, MENA, GCC, Scandinavia, etc.)
    - US states → resolves to "usa"
    - Canadian provinces → resolves to "canada"
    - Australian states → resolves to "australia"
    - Major world cities → resolves to country
    - Inside/outside/in/not in/excluding patterns
    """
    include: List[str] = []
    exclude: List[str] = []
    exclude_presence: List[str] = []

    # Detect all countries mentioned in the text (handles states, cities, regions)
    all_mentioned = find_countries_in_text(prompt_lower)

    # For each detected country, determine include vs exclude via context window
    # Strategy: look for exclusion markers BEFORE the country reference
    EXC_MARKERS = [
        r"outside", r"excluding", r"not in", r"not from", r"except",
        r"other than", r"except for", r"beyond", r"avoid",
    ]
    INC_MARKERS = [
        r"inside", r"in", r"within", r"from", r"based in", r"located in",
        r"operating in", r"headquartered in", r"companies in",
        r"vendors in", r"firms in",
    ]

    for country in all_mentioned:
        # We need to find which geo token(s) point to this country in the text
        # and check the marker before them
        # Use find_countries_in_text on sub-phrases around each exclusion marker
        exc_hit = False
        inc_hit = False

        # Check exclusion markers
        for marker in EXC_MARKERS:
            # "outside X" or "not in X" patterns
            if re.search(
                r"\b" + marker + r"\b[^.]{0,80}\b" + re.escape(country) + r"\b",
                prompt_lower
            ):
                exc_hit = True
                break
            # Also check if a sub-phrase after the marker contains this country
            for m in re.finditer(r"\b" + marker + r"\b([^.]{0,80})", prompt_lower):
                ctx = m.group(1)
                if country in find_countries_in_text(ctx):
                    exc_hit = True
                    break
            if exc_hit:
                break

        if not exc_hit:
            for marker in INC_MARKERS:
                if re.search(
                    r"\b" + marker + r"\b[^.]{0,80}\b" + re.escape(country) + r"\b",
                    prompt_lower
                ):
                    inc_hit = True
                    break
                for m in re.finditer(r"\b" + marker + r"\b([^.]{0,80})", prompt_lower):
                    ctx = m.group(1)
                    if country in find_countries_in_text(ctx):
                        inc_hit = True
                        break
                if inc_hit:
                    break

        if exc_hit:
            if country not in exclude:
                exclude.append(country)
        elif inc_hit:
            if country not in include:
                include.append(country)
        # No clear marker → don't assume (avoids false positives from "serving USA clients")

    # Also run pattern-based extraction for explicit phrases
    include.extend(_extract_grouped_geo(prompt_lower, [
        r"inside", r"in", r"within", r"from", r"based in", r"located in",
    ]))
    exclude.extend(_extract_grouped_geo(prompt_lower, [
        r"outside", r"excluding", r"except", r"not in", r"not from", r"beyond",
    ]))
    exclude_presence.extend(_extract_grouped_geo(prompt_lower, [
        r"no branches in", r"no offices in",
        r"without branches in", r"without offices in", r"no presence in",
    ]))

    include          = sorted(set(include))
    exclude          = sorted(set(exclude))
    exclude_presence = sorted(set(exclude_presence))
    include          = [c for c in include if c not in exclude]

    strict_mode = bool(include or exclude or exclude_presence)

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=strict_mode,
    )


def _clean_topic_text(text: str) -> str:
    """Strip geo, attribute, and format noise from a topic string."""
    text = _normalize(text)

    # Strip trailing phrases first (order matters)
    split_patterns = [
        r"\s+and\s+export\b.*$",
        r"\s+export\s+to\b.*$",
        r"\s+export\s+as\b.*$",
        r"\s+export\b.*$",              # "export pdf", "export to excel"
        r"\s+as\s+(?:a\s+)?pdf\b.*$",  # "as pdf", "as a pdf"
        r"\s+in\s+pdf\b.*$",           # "in pdf format"
        r"\s+pdf\s+file\b.*$",         # "pdf file"
        r"\s+pdf\s+format\b.*$",       # "pdf format"
        r"\s+pdf\b$",                  # trailing "pdf"
        r"\s+to\s+pdf\b.*$",           # "to pdf"
        r"\s+i\s+need\b.*$",
        r"\s+(?:give|provide|show|list)\s+(?:me\s+)?(?:the\s+)?(?:link|title|author|doi|abstract)\b.*$",
        r"\s+with\s+(?:email|phone|contact|number|linkedin|website|url|tel|mobile|link|title|author)\b.*$",
        r"\s+without\s+(?:email|phone|contact|number|linkedin)\b.*$",
        r"\s+(?:outside|excluding|except|not\s+in|not\s+from)\s+\b.*$",
        r"\s+(?:inside|within|from)\s+\b.*$",
        r"\s+in\s+(?:europe|asia|africa|north america|south america|middle east|north africa|mena|cis|gcc|nordics|apac)\b.*$",
    ]
    for pat in split_patterns:
        text = re.split(pat, text)[0].strip()

    # Strip noise words
    geo_words = set(all_country_names())
    # Add US states to geo words to strip from topic
    state_words = set(_ALL_SUBNATIONAL.keys())
    attr_noise = {
        "email", "emails", "phone", "phones", "contact", "number", "numbers",
        "linkedin", "website", "websites", "url", "tel", "mobile", "address",
        "operating", "operates", "headquartered",
        "link", "links", "title", "titles", "author", "authors",
        "doi", "journal", "year", "date", "abstract", "citation",
        "paper", "papers", "research", "study", "studies",
        # Output-format noise words
        "pdf", "csv", "excel", "xlsx", "xls", "json", "file", "files",
        "export", "download", "output", "format", "report", "document",
    }

    words = [w for w in re.split(r"[\s,/]+", text) if w]
    words = [
        w for w in words
        if w not in GENERIC_STOPWORDS
        and w not in attr_noise
        and w not in geo_words
        and w not in state_words
    ]

    return " ".join(words).strip()


def _extract_focus_term(prompt: str, prompt_lower: str, task_type: str) -> str:
    """
    Extract core topic from prompt. Universal — any sector.

    Strategy priority:
    1. Explicit "X industry/sector" phrase
    2. "papers/articles about X"
    3. "find [topic] companies" — only if multi-word or non-generic
    4. Post-entity phrase "companies working in X"
    5. Token fallback
    """
    geo_words = set(all_country_names())
    state_words = set(_ALL_SUBNATIONAL.keys())

    ENTITY_WORDS = {
        "companies", "company", "vendors", "vendor", "providers", "provider",
        "firms", "firm", "contractors", "contractor", "operators", "operator",
        "suppliers", "supplier", "businesses", "business",
        "organizations", "organization", "associations", "association",
        "people", "person", "papers", "paper", "reports", "report",
        "articles", "article", "studies", "study", "publications", "publication",
    }

    GENERIC_QUALIFIERS = {
        "service", "services", "digital", "software", "technology",
        "tech", "engineering", "technical", "global", "international",
        "local", "national", "leading", "top", "best", "new", "good",
    }

    def _clean(raw: str) -> str:
        return _clean_topic_text(raw)

    def _valid(s: str) -> bool:
        return bool(s) and len(s) > 1 and s not in ENTITY_WORDS

    # Strategy 1: "X industry/sector/space/field"
    for pat in [
        r"working\s+in\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"(?:in|within)\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"operating\s+in\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"serving\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"for\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    # Strategy 2: "papers/articles about X"
    for pat in [
        r"(?:papers?|studies|articles?|reports?|publications?|theses?|preprints?|research)\s+(?:about|on|related to|concerning|regarding)\s+(.+)",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    # Strategy 3: "find [qualifier] companies/vendors"
    entity_pattern = r"(?:companies|vendors|providers|firms|contractors?|operators?|suppliers?|startups?)"
    m = re.search(
        r"(?:find|search for|get|show|list|discover|look for)\s+"
        r"(?:all\s+|some\s+|top\s+|leading\s+|\d+\s+)?"
        r"(.+?)\s+" + entity_pattern + r"\b",
        prompt_lower,
    )
    if m:
        c = _clean(m.group(1))
        is_meaningful = (
            _valid(c)
            and c not in ENTITY_WORDS
            and (len(c.split()) >= 2 or c not in GENERIC_QUALIFIERS)
        )
        if is_meaningful:
            return c

    # Strategy 4: "companies working in / for X"
    for pat in [
        r"(?:companies|vendors|firms|providers)\s+(?:working|operating|active|specializing)\s+in\s+(.+)",
        r"(?:companies|vendors|firms|providers)\s+(?:in|for|serving)\s+(?:the\s+)?(.+?)\s+(?:sector|industry|space|field|market)\b",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    # Strategy 5: "about X" / "related to X"
    for pat in [r"about\s+(.+)", r"related\s+to\s+(.+)", r"on\s+(?:the\s+)?(.+)"]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c) and len(c.split()) >= 2:
                return c

    # Strategy 6: token fallback — strip all noise
    tokens = re.split(r"[\s,/]+", prompt_lower)
    filtered = [
        t for t in tokens
        if t
        and t not in GENERIC_STOPWORDS
        and t not in _REQUEST_NOISE
        and t not in geo_words
        and t not in state_words
        and t not in ENTITY_WORDS
        and len(t) > 2
    ]
    candidate = " ".join(filtered[:6]).strip()

    if task_type == "document_research" and not candidate:
        return "research"

    return candidate or ""


def _extract_target_attributes(prompt_lower: str, task_type: str) -> List[str]:
    found = []
    for attr, hints in ATTRIBUTE_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            found.append(attr)

    if task_type == "document_research":
        defaults = ["website", "summary", "author"]
        return sorted(set(found or defaults) | {"author"})

    return sorted(set(found or ["website"]))


def _extract_max_results(prompt: str) -> int:
    """Find explicit result count. Skips years. Default 25."""
    # Explicit count patterns first
    explicit = re.findall(
        r"\b(?:top|first|at least|around|about|up to|maximum|max)?\s*(\d{1,4})\s*"
        r"(?:results?|companies|vendors|firms|records?|entries|items?|papers?)\b",
        prompt, re.I,
    )
    for m in explicit:
        v = int(m)
        if 1 <= v <= 5000:
            return v

    # Any standalone number that isn't a year
    for m in re.findall(r"\b(\d{1,5})\b", prompt):
        v = int(m)
        if 1800 <= v <= 2100:
            continue
        if 1 <= v <= 5000:
            return v
    return 25


def parse_task_prompt(prompt: str) -> TaskSpec:
    """Full regex-based parser. Used as fallback when LLM is unavailable."""
    prompt       = (prompt or "").strip()
    prompt_lower = _normalize(prompt)

    task_type         = _extract_task_type(prompt_lower)
    entity_types      = _extract_entity_types(prompt_lower)
    target_category   = _extract_target_category(prompt_lower)
    geography         = _extract_geography(prompt_lower)
    output_format     = _extract_output_format(prompt_lower)
    focus_term        = _extract_focus_term(prompt, prompt_lower, task_type)
    target_attributes = _extract_target_attributes(prompt_lower, task_type)
    max_results       = _extract_max_results(prompt)

    ext      = output_format if output_format != "ui_table" else "xlsx"
    filename = f"results.{ext}"

    return TaskSpec(
        raw_prompt=prompt,
        task_type=task_type,
        target_entity_types=entity_types,
        target_category=target_category,
        industry=focus_term,
        target_attributes=target_attributes,
        geography=geography,
        output=OutputSpec(format=output_format, filename=filename),
        credential_mode=CredentialMode(mode="free"),
        use_local_llm=False,
        use_cloud_llm=False,
        max_results=max_results,
        mode="Balanced",
    )
