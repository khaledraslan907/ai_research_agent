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
    "website": ["website", "site", "url", "homepage", "link", "web"],
    "email": ["email", "emails", "mail", "contact email", "e-mail"],
    "phone": [
        "phone", "telephone", "tel", "mobile", "contact number",
        "phone number", "contact details", "contact info",
        "contact information", "contact",
    ],
    "linkedin": ["linkedin"],
    "hq_country": ["hq", "headquarters", "head office", "based in", "headquartered"],
    "presence_countries": ["branches", "offices", "presence", "locations", "regional"],
    "summary": ["summary", "overview", "description", "abstract", "bio"],
    "author": ["author", "authors", "written by", "authored by", "who wrote", "researcher", "researchers"],
    "pdf": ["pdf", "full text", "download"],
}

FORMAT_HINTS = {
    "xlsx": ["excel", "xlsx", "spreadsheet", "xls"],
    "csv": ["csv"],
    "pdf": [
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
    "linkedin", "profiles", "profile", "account", "accounts",
    "engineers", "managers", "manager", "engineer", "specialist",
    "specialists", "director", "directors", "hr", "executives", "executive",
    "professionals", "professional", "employees", "staff", "personnel",
    "team", "teams",
    "agent", "agency", "agencies", "distributor", "distributors",
    "reseller", "resellers", "representative", "representation",
    "partner", "partners", "channel", "local", "suitable", "prioritize",
    "prioritise",
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
    "app", "application", "monitoring", "optimization", "optimisation",
}

SERVICE_CATEGORY_HINTS = {
    "service company", "service companies", "contractor", "contractors",
    "field service", "services provider", "engineering services",
    "oilfield services", "drilling services", "field services",
    "maintenance", "inspection", "testing", "consulting services",
}

# exact broad technical families only
SOLUTION_KEYWORD_PATTERNS = {
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "artificial intelligence": [r"\bartificial intelligence\b"],
    "ai": [r"\bai\b"],
    "analytics": [r"\banalytics\b", r"\banalytic\b", r"\binsights\b"],
    "monitoring": [r"\bmonitoring\b", r"\bremote monitoring\b", r"\bsurveillance\b"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b", r"\boptimizer\b", r"\boptimiser\b"],
    "automation": [r"\bautomation\b", r"\bautomated\b", r"\bautonomous\b"],
    "iot": [r"\biot\b", r"\binternet of things\b"],
    "scada": [r"\bscada\b"],
    "digital twin": [r"\bdigital twin\b", r"\bdigital twins\b"],
    "predictive maintenance": [r"\bpredictive maintenance\b"],
}

# exact domain/use-case phrases
DOMAIN_KEYWORD_PATTERNS = {
    "esp": [r"\besp\b", r"\belectrical submersible pump\b", r"\belectric submersible pump\b"],
    "virtual flow metering": [r"\bvirtual flow metering\b", r"\bvirtual flow meter\b", r"\bvirtual meter\b"],
    "well performance": [r"\bwell performance\b"],
    "artificial lift": [r"\bartificial lift\b"],
    "production optimization": [r"\bproduction optimization\b", r"\bproduction optimisation\b"],
    "well surveillance": [r"\bwell surveillance\b"],
    "multiphase metering": [r"\bmultiphase metering\b", r"\bmultiphase meter\b"],
    "flow assurance": [r"\bflow assurance\b"],
    "production monitoring": [r"\bproduction monitoring\b"],
    "reservoir simulation": [r"\breservoir simulation\b"],
    "reservoir modeling": [r"\breservoir modeling\b", r"\breservoir modelling\b"],
    "drilling optimization": [r"\bdrilling optimization\b", r"\bdrilling optimisation\b"],
    "production engineering": [r"\bproduction engineering\b"],
}

# critical short aliases
GEO_ALIAS_MAP = {
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "us": "united states",
}

NEGATIVE_GEO_CUES = [
    "no ", "not ", "without ", "exclude", "excluding", "except",
    "reject", "remove", "outside", "avoid", "other than",
    "do not have", "does not have", "has no", "with no",
]

PRESENCE_ENTITY_WORDS = [
    "office", "offices",
    "branch", "branches",
    "subsidiary", "subsidiaries",
    "local entity", "local entities",
    "entity", "entities",
    "presence",
    "legal entity", "legal entities",
    "registered entity", "registered entities",
    "operations", "operation",
    "distributor", "distributors",
    "agent", "agents",
]

NEGATIVE_CUE_RE = re.compile(
    r"\b(?:exclude|excluding|reject|remove|avoid|except|other than|not in|not from|outside|without|no|has no|with no|do(?:es)? not have)\b",
    re.IGNORECASE,
)

PRESENCE_NOUN_RE = re.compile(
    r"\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|operations?|legal entity|legal entities|registered entity|registered entities|distributor|distributors|agent|agents)\b",
    re.IGNORECASE,
)

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


def _dedupe_repeated_prompt(prompt: str) -> str:
    """
    If the exact same prompt is accidentally pasted twice back-to-back,
    keep only one copy.
    """
    raw = (prompt or "").strip()
    if not raw:
        return raw

    half = len(raw) // 2
    left = raw[:half].strip()
    right = raw[half:].strip()

    if left and right and left == right:
        return left

    if len(raw) > 40:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        deduped = []
        prev = None
        for s in sentences:
            if prev is not None and s == prev:
                continue
            deduped.append(s)
            prev = s
        return " ".join(deduped)

    return raw


def _expand_geo_name(name: str) -> List[str]:
    name = (name or "").strip().lower()
    if not name:
        return []

    if name in GEO_ALIAS_MAP:
        return [GEO_ALIAS_MAP[name]]

    expanded = expand_region_name(name)
    if expanded:
        return list(expanded)

    norm = normalize_country_name(name)
    if norm:
        return [norm]

    return []


def _find_geo_tokens_in_text(text: str) -> List[str]:
    text = _normalize(text)
    if not text:
        return []

    found: List[str] = []
    all_geo_tokens = sorted(
        all_country_names() + list(_ALL_REGIONS) + list(GEO_ALIAS_MAP.keys()),
        key=len,
        reverse=True,
    )

    for item in all_geo_tokens:
        if re.search(r"\b" + re.escape(item) + r"\b", text):
            found.append(item)

    return _dedupe_keep_order(found)


def _find_geo_after_marker(prompt_lower: str, marker_patterns: List[str], mode: str = "generic") -> List[str]:
    found: List[str] = []
    all_geo_tokens = sorted(
        all_country_names() + list(_ALL_REGIONS) + list(GEO_ALIAS_MAP.keys()),
        key=len,
        reverse=True,
    )

    for marker in marker_patterns:
        for item in all_geo_tokens:
            for m in re.finditer(
                rf"({marker})\s+{re.escape(item)}(?:\s*(?:,|and|or)\s*([a-zA-Z\s,\-]+))?",
                prompt_lower,
            ):
                start = m.start()
                left_ctx = prompt_lower[max(0, start - 50):start]
                whole_ctx = prompt_lower[max(0, start - 80):min(len(prompt_lower), m.end() + 40)]

                if mode == "include":
                    if any(cue in left_ctx for cue in NEGATIVE_GEO_CUES):
                        continue
                    if re.search(
                        r"\b(?:no|without|not|exclude|excluding|reject|remove|avoid|outside|except|do not have|does not have|has no|with no)\b",
                        whole_ctx,
                    ):
                        continue

                names = [item]
                tail = m.group(2) or ""
                if tail:
                    names.extend(_find_geo_tokens_in_text(tail))

                for n in names:
                    found.extend(_expand_geo_name(n))

    return _dedupe_keep_order(found)


def _looks_like_presence_exclusion(prompt_lower: str) -> bool:
    presence_patterns = [
        r"\boperate(?:s|d|ing)?\s+outside\b",
        r"\bwork(?:s|ed|ing)?\s+outside\b",
        r"\bserv(?:e|es|ed|ing)?\s+outside\b",
        r"\bactive\s+outside\b",
        r"\bpresent\s+outside\b",
        r"\bno\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|presence|operations?|legal entities?)\s+in\b",
        r"\bwithout\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|presence|operations?|legal entities?)\s+in\b",
        r"\bhas\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|presence|operations?|legal entities?)\s+in\b",
        r"\bwith\s+no\s+(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|presence|operations?|legal entities?)\s+in\b",
        r"\bdo(?:es)?\s+not\s+have\s+(?:an?\s+)?(?:office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|presence|operations?|legal entities?)\s+in\b",
        r"\b(?:exclude|excluding|reject|remove|avoid)\b[^.\n;]{0,80}\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|operations?|legal entities?)\b",
        r"\b(?:exclude|excluding|reject|remove|avoid)\b[^.\n;]{0,80}\b(?:egypt|usa|united states)\b[^.\n;]{0,20}\bpresence\b",
    ]
    return any(re.search(pat, prompt_lower) for pat in presence_patterns)


def _extract_geo_from_negative_presence_phrases(prompt_lower: str) -> List[str]:
    found: List[str] = []
    span_patterns = [
        r"\b(?:no|without|has no|with no|do(?:es)? not have)\b([^.:\n;]{0,180})",
        r"\b(?:exclude|excluding|reject|remove|avoid)\b([^.:\n;]{0,180})",
        r"\b(?:operate(?:s|d|ing)?|work(?:s|ed|ing)?|serv(?:e|es|ed|ing)?|active|present)\s+outside\b([^.:\n;]{0,180})",
    ]

    for pat in span_patterns:
        for m in re.finditer(pat, prompt_lower):
            span = m.group(1)
            if "outside" in pat:
                for token in _find_geo_tokens_in_text(span):
                    found.extend(_expand_geo_name(token))
                continue
            if PRESENCE_NOUN_RE.search(span):
                for token in _find_geo_tokens_in_text(span):
                    found.extend(_expand_geo_name(token))

    return _dedupe_keep_order(found)


def _extract_geo_from_negative_phrases(prompt_lower: str) -> List[str]:
    found: List[str] = []
    span_patterns = [
        r"\b(?:exclude|excluding|reject|remove|avoid|except|other than|not in|not from|outside)\b([^.:\n;]{0,180})",
    ]
    for pat in span_patterns:
        for m in re.finditer(pat, prompt_lower):
            span = m.group(1)
            if PRESENCE_NOUN_RE.search(span):
                continue
            for token in _find_geo_tokens_in_text(span):
                found.extend(_expand_geo_name(token))
    return _dedupe_keep_order(found)


def _extract_geo_from_positive_in_phrases(prompt_lower: str) -> List[str]:
    """
    Catch positive geography lists like:
    - software in Norway, UK, UAE, and Saudi Arabia
    - operations in Australia, Canada, and the UK

    Only scans text before exclusion language begins, and only within
    short sentence-bounded spans to avoid negative-cue bleed.
    """
    found: List[str] = []

    positive_part = re.split(
        r"\b(?:exclude|excluding|reject|remove|avoid|except|other than|not in|not from|outside|without|no|has no|with no|do(?:es)? not have)\b",
        prompt_lower,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    candidate_spans = re.split(r"[.;:!?]", positive_part)

    for sentence in candidate_spans:
        sentence = sentence.strip()
        if not sentence:
            continue

        for m in re.finditer(r"\bin\s+([a-zA-Z\s,\-]+)", sentence, flags=re.IGNORECASE):
            span = m.group(1)

            span = re.split(
                r"\b(?:with|that|which|who|providing|offering|suitable|return|keep|only|where)\b",
                span,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            geo_tokens = _find_geo_tokens_in_text(span)
            if not geo_tokens:
                continue

            for token in geo_tokens:
                found.extend(_expand_geo_name(token))

    return _dedupe_keep_order(found)


def _extract_geography(prompt_lower: str) -> GeographyRules:
    include: List[str] = []
    exclude: List[str] = []
    exclude_presence: List[str] = []

    include_markers = [
        r"search in",
        r"search only in",
        r"search within",
        r"look in",
        r"companies in",
        r"vendors in",
        r"firms in",
        r"providers in",
        r"software companies in",
        r"technology companies in",
        r"headquartered in",
        r"based in",
        r"located in",
        r"from",
        r"within",
        r"inside",
        r"in the following regions",
        r"in the following countries",
        r"in the following regions only",
        r"in the following countries only",
    ]

    exclude_markers = [
        r"outside", r"excluding", r"exclude", r"except", r"not in",
        r"not from", r"other than", r"beyond", r"avoid", r"reject", r"remove",
    ]

    presence_markers = [
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
        r"do not have offices in",
        r"does not have offices in",
        r"do not have branches in",
        r"does not have branches in",
        r"do not have subsidiaries in",
        r"does not have subsidiaries in",
        r"do not have local entities in",
        r"does not have local entities in",
    ]

    exclude_presence.extend(_extract_geo_from_negative_presence_phrases(prompt_lower))
    exclude.extend(_extract_geo_from_negative_phrases(prompt_lower))
    include.extend(_extract_geo_from_positive_in_phrases(prompt_lower))

    include.extend(_find_geo_after_marker(prompt_lower, include_markers, mode="include"))
    exclude.extend(_find_geo_after_marker(prompt_lower, exclude_markers, mode="exclude"))
    exclude_presence.extend(_find_geo_after_marker(prompt_lower, presence_markers, mode="exclude_presence"))

    all_mentioned = find_countries_in_text(prompt_lower)
    for country in all_mentioned:
        for m in re.finditer(r"\b" + re.escape(country) + r"\b", prompt_lower):
            left = prompt_lower[max(0, m.start() - 90):m.start()]
            right = prompt_lower[m.end():min(len(prompt_lower), m.end() + 90)]
            window = left + country + right

            if (
                (NEGATIVE_CUE_RE.search(left) and PRESENCE_NOUN_RE.search(window))
                or re.search(
                    r"\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|operations?|legal entity|legal entities)\b\s+in\s+"
                    + re.escape(country) + r"\b",
                    window,
                )
                or re.search(
                    r"\b" + re.escape(country) + r"\b\s+(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|local entity|local entities|entity|entities|operations?|legal entity|legal entities)\b",
                    window,
                )
            ):
                exclude_presence.append(country)
                continue

            if NEGATIVE_CUE_RE.search(left):
                exclude.append(country)
                continue

    include = _dedupe_keep_order(include)
    exclude = _dedupe_keep_order(exclude)
    exclude_presence = _dedupe_keep_order(exclude_presence)

    if _looks_like_presence_exclusion(prompt_lower) and exclude and not exclude_presence:
        exclude_presence = list(exclude)
        exclude = []

    include = [c for c in include if c not in exclude and c not in exclude_presence]

    # hard protection: never let an explicitly included country remain in excludes
    exclude_presence = [c for c in exclude_presence if c not in include]
    exclude = [c for c in exclude if c not in include]
    exclude = [c for c in exclude if c not in exclude_presence]

    strict_mode = bool(include or exclude or exclude_presence)

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=strict_mode,
    )


def _extract_entity_types(prompt_lower: str) -> List[str]:
    found = []
    for entity_type, hints in ENTITY_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            found.append(entity_type)

    if "paper" in found and "person" in found:
        explicit_person = any(w in prompt_lower for w in ["people", "researcher", "expert", "consultant", "ceo", "founder", "director"])
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
        r"\bapp(?:application)?\b",
        r"\bdata company\b",
        r"\btech company\b",
        r"\btechnology (?:provider|vendor|company|companies)\b",
        r"\bai company\b",
        r"\bai companies\b",
        r"\bmachine learning\b",
        r"\bartificial intelligence\b",
        r"\bmonitoring\b",
        r"\boptimization\b",
        r"\boptimisation\b",
    ]
    if any(re.search(pat, prompt_lower) for pat in digital_patterns):
        return "software_company"

    return "general"


def _normalize_industry_candidate(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"\boil\s*(?:&|and)?\s*gas\b", "oil and gas", text)
    text = re.sub(r"\boil\s+gas\b", "oil and gas", text)

    removable_phrases = sorted(
        set(DIGITAL_CATEGORY_HINTS) | set(SERVICE_CATEGORY_HINTS) | {
            "company", "companies", "vendor", "vendors", "provider", "providers",
            "firm", "firms", "startup", "startups", "supplier", "suppliers",
            "technology company", "technology companies",
            "software company", "software companies",
            "digital company", "digital companies",
        },
        key=len,
        reverse=True,
    )

    for phrase in removable_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip(" ,-")

    if re.search(r"\boil\b", text) and re.search(r"\bgas\b", text):
        return "oil and gas"

    return text


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
        r"\s+prioriti[sz]e\b.*$",
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

    cleaned = " ".join(words).strip()
    cleaned = _normalize_industry_candidate(cleaned)
    return cleaned


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
        "monitoring", "optimization", "optimisation",
        "machine", "learning", "artificial", "intelligence", "iot", "scada",
        "esp", "virtual", "flow", "metering", "well", "performance",
        "lift", "production", "surveillance", "multiphase",
        "assurance", "simulation", "modeling", "modelling", "drilling",
    }

    def _clean(raw: str) -> str:
        return _clean_topic_text(raw)

    def _valid(s: str) -> bool:
        return bool(s) and len(s) > 1 and s not in entity_words

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

    for pat in [
        r"(?:papers?|studies|articles?|reports?|publications?|theses?|preprints?|research|literature)\s+(?:about|on|related to|concerning|regarding)\s+(.+)",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

    for pat in [
        r"(?:companies|vendors|firms|providers)\s+(?:working|operating|active|specializing)\s+in\s+(.+)",
        r"(?:companies|vendors|firms|providers)\s+(?:in|for|serving)\s+(?:the\s+)?(.+?)\s+(?:sector|industry|space|field|market)\b",
    ]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c):
                return c

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

    for pat in [r"about\s+(.+)", r"related\s+to\s+(.+)", r"on\s+(?:the\s+)?(.+)"]:
        m = re.search(pat, prompt_lower)
        if m:
            c = _clean(m.group(1))
            if _valid(c) and len(c.split()) >= 2:
                return c

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
    candidate = _normalize_industry_candidate(" ".join(filtered[:6]).strip())

    if task_type == "document_research" and not candidate:
        return "research"

    return candidate or ""


def _extract_solution_keywords(prompt_lower: str) -> List[str]:
    found: List[str] = []
    for label, patterns in SOLUTION_KEYWORD_PATTERNS.items():
        if any(re.search(pat, prompt_lower) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)


def _extract_domain_keywords(prompt_lower: str) -> List[str]:
    found: List[str] = []
    for label, patterns in DOMAIN_KEYWORD_PATTERNS.items():
        if any(re.search(pat, prompt_lower) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)


def _extract_commercial_intent(prompt_lower: str) -> str:
    if re.search(r"\b(agent|agency|distributor|distribution|local representation|representative|representation)\b", prompt_lower):
        return "agent_or_distributor"
    if re.search(r"\b(reseller|resellers)\b", prompt_lower):
        return "reseller"
    if re.search(r"\b(partner|partners|channel partner|channel partners)\b", prompt_lower):
        return "partner"
    return "general"


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


def parse_task_prompt(prompt: str) -> TaskSpec:
    prompt = _dedupe_repeated_prompt((prompt or "").strip())
    prompt_lower = _normalize(prompt)

    task_type = _extract_task_type(prompt_lower)
    entity_types = _extract_entity_types(prompt_lower)
    target_category = _extract_target_category(prompt_lower)
    geography = _extract_geography(prompt_lower)
    output_format = _extract_output_format(prompt_lower)
    focus_term = _extract_focus_term(prompt, prompt_lower, task_type)
    solution_keywords = _extract_solution_keywords(prompt_lower)
    domain_keywords = _extract_domain_keywords(prompt_lower)
    commercial_intent = _extract_commercial_intent(prompt_lower)
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
        solution_keywords=solution_keywords,
        domain_keywords=domain_keywords,
        commercial_intent=commercial_intent,
        target_attributes=target_attributes,
        geography=geography,
        output=OutputSpec(format=output_format, filename=filename),
        credential_mode=CredentialMode(mode="free"),
        use_local_llm=False,
        use_cloud_llm=False,
        max_results=max_results,
        mode="Balanced",
    )
