"""
task_parser.py
==============
Universal regex-first task parser for the research agent.

Main improvements in this version:
- Preserves the industry/topic separately from company category
- Correctly classifies prompts like "digital companies in oil and gas"
  as target_category="software_company" and industry="oil and gas"
- Treats phrases like "operate outside Egypt and USA" as presence exclusion
  when the language is about operational footprint rather than headquarters
- Improves geography extraction for include / exclude / exclude_presence
- Keeps document-research and people-search behavior compatible with the rest
  of the agent
"""

from __future__ import annotations

import re
from typing import List

from core.task_models import TaskSpec, GeographyRules, OutputSpec, CredentialMode
from core.geography import (
    all_country_names,
    expand_region_name,
    normalize_country_name,
    find_countries_in_text,
    US_STATES,
    CANADIAN_PROVINCES,
    AUSTRALIAN_STATES,
    REGION_ALIASES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Entity / attribute / format hints
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_HINTS = {
    "company": [
        "company", "companies", "vendor", "vendors", "firm", "firms",
        "provider", "providers", "contractor", "contractors", "operator", "operators",
        "supplier", "suppliers", "startup", "startups", "corporation", "corporations",
        "enterprise", "enterprises", "agency", "agencies", "business", "businesses",
    ],
    "person": [
        "person", "people", "founder", "ceo", "manager", "contact person",
        "researcher", "expert", "consultant", "director", "executive",
        "scientist", "engineer", "professor",
    ],
    "paper": [
        "paper", "papers", "journal", "study", "studies", "publication",
        "publications", "article", "articles", "report", "reports",
        "thesis", "theses", "preprint", "research", "literature",
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
    "inside", "outside", "working", "works", "worked", "operate", "operates",
    "operated", "operating", "with", "without", "and", "or", "in", "on", "to",
    "export", "into", "as", "from", "of", "the", "a", "an", "related",
    "companies", "company", "papers", "paper", "organizations", "organization",
    "events", "event", "products", "product", "people", "person", "list", "all",
    "some", "any", "give", "that", "are", "i", "want", "need", "can", "you",
    "help", "also", "then", "provide", "their", "its", "which", "where",
    "when", "how", "what", "who", "each", "every", "per", "this", "these",
    "those", "have", "has", "do", "does", "did", "is", "was", "been", "being",
    "be", "but", "across", "serving", "serve", "serves", "serviced",
    # people / linkedin noise
    "linkedin", "profiles", "profile", "account", "accounts",
    "engineers", "managers", "manager", "engineer", "specialist",
    "specialists", "director", "directors", "hr", "executives", "executive",
    "professionals", "professional", "employees", "staff", "personnel",
    "team", "teams",
}

_REQUEST_NOISE = {
    "link", "links", "title", "titles", "author", "authors",
    "doi", "journal", "year", "date", "abstract", "volume",
    "issue", "page", "pages", "citation", "reference", "references",
    "number", "numbers", "contact", "details", "info", "information",
    "pdf", "csv", "excel", "xlsx", "xls", "json",
    "file", "files", "export", "download", "output", "format",
    "report", "document", "summary",
}

DIGITAL_CATEGORY_HINTS = {
    "digital", "software", "technology", "tech", "analytics", "automation",
    "platform", "platforms", "saas", "cloud", "ai", "artificial intelligence",
    "machine learning", "data", "iot", "scada", "digitalization", "digitization",
    "app", "application",
}

SERVICE_CATEGORY_HINTS = {
    "service company", "service companies", "contractor", "contractors",
    "field service", "services provider", "engineering services",
    "oilfield services", "drilling services", "field services",
    "maintenance", "inspection", "testing", "consulting services",
}

# ─────────────────────────────────────────────────────────────────────────────
# Geo helpers
# ─────────────────────────────────────────────────────────────────────────────

_ALL_SUBNATIONAL = {}
_ALL_SUBNATIONAL.update(US_STATES)
_ALL_SUBNATIONAL.update(CANADIAN_PROVINCES)
_ALL_SUBNATIONAL.update(AUSTRALIAN_STATES)
_ALL_REGIONS = set(REGION_ALIASES.keys())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _expand_geo_name(name: str) -> List[str]:
    name = (name or "").strip().lower()
    if not name:
        return []
    expanded = expand_region_name(name)
    if expanded:
        return list(expanded)
    norm = normalize_country_name(name)
    if norm:
        return [norm]
    return []


def _find_geo_after_marker(prompt_lower: str, marker_patterns: List[str]) -> List[str]:
    """
    Extract countries/regions mentioned after marker phrases.
    """
    found: List[str] = []
    all_geo_tokens = sorted(all_country_names() + list(_ALL_REGIONS), key=len, reverse=True)

    for marker in marker_patterns:
        for item in all_geo_tokens:
            m = re.search(
                rf"{marker}\s+{re.escape(item)}(?:\s*(?:,|and|or)\s*([a-zA-Z\s,\-]+))?",
                prompt_lower,
            )
            if not m:
                continue

            names = [item]
            tail = m.group(1) or ""
            if tail:
                names.extend(find_countries_in_text(tail))

            for n in names:
                found.extend(_expand_geo_name(n))

    return _dedupe_keep_order(found)


def _looks_like_presence_exclusion(prompt_lower: str) -> bool:
    presence_patterns = [
        r"\boperate(?:s|d|ing)?\s+outside\b",
        r"\bwork(?:s|ed|ing)?\s+outside\b",
        r"\bserv(?:e|es|ed|ing)\s+outside\b",
        r"\bactive\s+outside\b",
        r"\bpresent\s+outside\b",
        r"\boutside\b.*\boperate(?:s|d|ing)?\b",
        r"\boutside\b.*\bworking\b",
        r"\boutside\b.*\bserving\b",
        r"\bno\s+presence\s+in\b",
        r"\bwithout\s+presence\s+in\b",
        r"\bno\s+offices\s+in\b",
        r"\bwithout\s+offices\s+in\b",
        r"\bno\s+branches\s+in\b",
        r"\bwithout\s+branches\s+in\b",
    ]
    return any(re.search(pat, prompt_lower) for pat in presence_patterns)


def _extract_geography(prompt_lower: str) -> GeographyRules:
    include: List[str] = []
    exclude: List[str] = []
    exclude_presence: List[str] = []

    all_mentioned = find_countries_in_text(prompt_lower)

    # Context windows around mentioned countries
    for country in all_mentioned:
        exc_hit = False
        inc_hit = False
        exc_presence_hit = False

        # Presence-exclusion phrasing
        presence_markers = [
            r"operate(?:s|d|ing)?\s+outside",
            r"work(?:s|ed|ing)?\s+outside",
            r"serv(?:e|es|ed|ing)\s+outside",
            r"active\s+outside",
            r"present\s+outside",
            r"no\s+presence\s+in",
            r"without\s+presence\s+in",
            r"no\s+offices\s+in",
            r"without\s+offices\s+in",
            r"no\s+branches\s+in",
            r"without\s+branches\s+in",
        ]
        for marker in presence_markers:
            if re.search(r"\b" + marker + r"\b[^.]{0,100}\b" + re.escape(country) + r"\b", prompt_lower):
                exc_presence_hit = True
                break
            for m in re.finditer(r"\b" + marker + r"\b([^.]{0,100})", prompt_lower):
                ctx = m.group(1)
                if country in find_countries_in_text(ctx):
                    exc_presence_hit = True
                    break
            if exc_presence_hit:
                break

        if not exc_presence_hit:
            exc_markers = [
                r"outside", r"excluding", r"not in", r"not from", r"except",
                r"other than", r"except for", r"beyond", r"avoid",
            ]
            for marker in exc_markers:
                if re.search(r"\b" + marker + r"\b[^.]{0,80}\b" + re.escape(country) + r"\b", prompt_lower):
                    exc_hit = True
                    break
                for m in re.finditer(r"\b" + marker + r"\b([^.]{0,80})", prompt_lower):
                    ctx = m.group(1)
                    if country in find_countries_in_text(ctx):
                        exc_hit = True
                        break
                if exc_hit:
                    break

        if not exc_hit and not exc_presence_hit:
            inc_markers = [
                r"inside", r"in", r"within", r"from", r"based in", r"located in",
                r"operating in", r"headquartered in", r"companies in", r"vendors in",
                r"firms in",
            ]
            for marker in inc_markers:
                if re.search(r"\b" + marker + r"\b[^.]{0,80}\b" + re.escape(country) + r"\b", prompt_lower):
                    inc_hit = True
                    break
                for m in re.finditer(r"\b" + marker + r"\b([^.]{0,80})", prompt_lower):
                    ctx = m.group(1)
                    if country in find_countries_in_text(ctx):
                        inc_hit = True
                        break
                if inc_hit:
                    break

        if exc_presence_hit:
            if country not in exclude_presence:
                exclude_presence.append(country)
        elif exc_hit:
            if country not in exclude:
                exclude.append(country)
        elif inc_hit:
            if country not in include:
                include.append(country)

    include.extend(_find_geo_after_marker(prompt_lower, [
        r"inside", r"in", r"within", r"from", r"based in", r"located in",
    ]))
    exclude.extend(_find_geo_after_marker(prompt_lower, [
        r"outside", r"excluding", r"except", r"not in", r"not from", r"beyond",
    ]))
    exclude_presence.extend(_find_geo_after_marker(prompt_lower, [
        r"operate(?:s|d|ing)? outside",
        r"operating outside",
        r"work(?:s|ed|ing)? outside",
        r"working outside",
        r"serv(?:e|es|ed|ing)? outside",
        r"serving outside",
        r"active outside",
        r"present outside",
        r"no branches in",
        r"without branches in",
        r"no offices in",
        r"without offices in",
        r"no presence in",
        r"without presence in",
    ]))

    include = _dedupe_keep_order(include)
    exclude = _dedupe_keep_order(exclude)
    exclude_presence = _dedupe_keep_order(exclude_presence)

    # If the prompt explicitly uses "operate/work/serve outside", prefer presence exclusion.
    if _looks_like_presence_exclusion(prompt_lower) and exclude and not exclude_presence:
        exclude_presence = list(exclude)

    # Keep include clean
    include = [c for c in include if c not in exclude and c not in exclude_presence]

    strict_mode = bool(include or exclude or exclude_presence)

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=strict_mode,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Task / entity / output extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entity_types(prompt_lower: str) -> List[str]:
    found = []
    for entity_type, hints in ENTITY_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            found.append(entity_type)

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
    if any(x in prompt_lower for x in [
        "linkedin", "linkedin profile", "linkedin account",
        "people who work", "people working", "professionals in",
        "find engineers", "find managers", "find hr", "find people",
        "employees in", "staff in", "workers in",
    ]):
        return "people_search"
    return "entity_discovery"


def _extract_target_category(prompt_lower: str) -> str:
    if any(x in prompt_lower for x in SERVICE_CATEGORY_HINTS):
        return "service_company"

    # Strong digital/software detection
    digital_patterns = [
        r"\bdigital\b",
        r"\bsoftware\b",
        r"\bsaas\b",
        r"\bplatform(?:s)?\b",
        r"\banalytics\b",
        r"\bautomation\b",
        r"\bcloud\b",
        r"\biot\b",
        r"\bscada\b",
        r"\bapp(?:lication)?\b",
        r"\bdata company\b",
        r"\btech company\b",
        r"\btechnology (?:provider|vendor|company|companies)\b",
        r"\bai company\b",
        r"\bai companies\b",
        r"\bmachine learning\b",
        r"\bartificial intelligence\b",
    ]
    if any(re.search(pat, prompt_lower) for pat in digital_patterns):
        return "software_company"

    return "general"

# ─────────────────────────────────────────────────────────────────────────────
# Topic extraction
# ─────────────────────────────────────────────────────────────────────────────

def _clean_topic_text(text: str) -> str:
    text = _normalize(text)

    split_patterns = [
        r"\s+and\s+export\b.*$",
        r"\s+export\s+to\b.*$",
        r"\s+export\s+as\b.*$",
        r"\s+export\b.*$",
        r"\s+as\s+(?:a\s+)?pdf\b.*$",
        r"\s+in\s+pdf\b.*$",
        r"\s+pdf\s+file\b.*$",
        r"\s+pdf\s+format\b.*$",
        r"\s+pdf\b$",
        r"\s+to\s+pdf\b.*$",
        r"\s+i\s+need\b.*$",
        r"\s+(?:give|provide|show|list)\s+(?:me\s+)?(?:the\s+)?(?:link|title|author|doi|abstract)\b.*$",
        r"\s+with\s+(?:email|phone|contact|number|linkedin|website|url|tel|mobile|link|title|author)\b.*$",
        r"\s+without\s+(?:email|phone|contact|number|linkedin)\b.*$",
        r"\s+(?:outside|excluding|except|not\s+in|not\s+from)\s+\b.*$",
        r"\s+(?:operate|operates|operating|work|works|working|serve|serves|serving)\s+outside\s+\b.*$",
        r"\s+(?:inside|within|from)\s+\b.*$",
        r"\s+in\s+(?:europe|asia|africa|north america|south america|middle east|north africa|mena|cis|gcc|nordics|apac)\b.*$",
    ]
    for pat in split_patterns:
        text = re.split(pat, text)[0].strip()

    geo_words = set(all_country_names())
    state_words = set(_ALL_SUBNATIONAL.keys())
    attr_noise = {
        "email", "emails", "phone", "phones", "contact", "number", "numbers",
        "linkedin", "website", "websites", "url", "tel", "mobile", "address",
        "operating", "operates", "headquartered", "link", "links", "title",
        "titles", "author", "authors", "doi", "journal", "year", "date",
        "abstract", "citation", "paper", "papers", "research", "study",
        "studies", "pdf", "csv", "excel", "xlsx", "xls", "json", "file",
        "files", "export", "download", "output", "format", "report", "document",
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
    geo_words = set(all_country_names())
    state_words = set(_ALL_SUBNATIONAL.keys())

    entity_words = {
        "companies", "company", "vendors", "vendor", "providers", "provider",
        "firms", "firm", "contractors", "contractor", "operators", "operator",
        "suppliers", "supplier", "businesses", "business", "organizations",
        "organization", "associations", "association", "people", "person",
        "papers", "paper", "reports", "report", "articles", "article",
        "studies", "study", "publications", "publication",
    }

    generic_qualifiers = {
        "service", "services", "digital", "software", "technology", "tech",
        "engineering", "technical", "global", "international", "local",
        "national", "leading", "top", "best", "new", "good", "ai", "data",
        "analytics", "automation", "platform", "platforms", "cloud",
    }

    def _clean(raw: str) -> str:
        return _clean_topic_text(raw)

    def _valid(s: str) -> bool:
        return bool(s) and len(s) > 1 and s not in entity_words

    # Strategy 1: explicit "X industry/sector"
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

    # Strategy 2: papers/articles about X
    for pat in [
        r"(?:papers?|studies|articles?|reports?|publications?|theses?|preprints?|research|literature)\s+(?:about|on|related to|concerning|regarding)\s+(.+)",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    # Strategy 3: "companies working/operating in X"
    for pat in [
        r"(?:companies|vendors|firms|providers)\s+(?:working|operating|active|specializing)\s+in\s+(.+)",
        r"(?:companies|vendors|firms|providers)\s+(?:in|for|serving)\s+(?:the\s+)?(.+?)\s+(?:sector|industry|space|field|market)\b",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    # Strategy 4: "find [topic] companies" but avoid category-only qualifiers
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
            and c not in entity_words
            and (len(c.split()) >= 2 or c not in generic_qualifiers)
        )
        if is_meaningful:
            return c

    # Strategy 5: about X / related to X
    for pat in [r"about\s+(.+)", r"related\s+to\s+(.+)", r"on\s+(?:the\s+)?(.+)"]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c) and len(c.split()) >= 2:
                return c

    # Fallback: strip noise
    tokens = re.split(r"[\s,/]+", prompt_lower)
    filtered = [
        t for t in tokens
        if t
        and t not in GENERIC_STOPWORDS
        and t not in _REQUEST_NOISE
        and t not in geo_words
        and t not in state_words
        and t not in entity_words
        and t not in generic_qualifiers
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
    explicit = re.findall(
        r"\b(?:top|first|at least|around|about|up to|maximum|max)?\s*(\d{1,4})\s*"
        r"(?:results?|companies|vendors|firms|records?|entries|items?|papers?)\b",
        prompt, re.I,
    )
    for m in explicit:
        v = int(m)
        if 1 <= v <= 5000:
            return v

    for m in re.findall(r"\b(\d{1,5})\b", prompt):
        v = int(m)
        if 1800 <= v <= 2100:
            continue
        if 1 <= v <= 5000:
            return v
    return 25

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_task_prompt(prompt: str) -> TaskSpec:
    prompt = (prompt or "").strip()
    prompt_lower = _normalize(prompt)

    task_type = _extract_task_type(prompt_lower)
    entity_types = _extract_entity_types(prompt_lower)
    target_category = _extract_target_category(prompt_lower)
    geography = _extract_geography(prompt_lower)
    output_format = _extract_output_format(prompt_lower)
    focus_term = _extract_focus_term(prompt, prompt_lower, task_type)
    target_attributes = _extract_target_attributes(prompt_lower, task_type)
    max_results = _extract_max_results(prompt)

    ext = output_format if output_format != "ui_table" else "xlsx"
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
