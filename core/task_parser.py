from __future__ import annotations

import re
from typing import List

from core.task_models import CredentialMode, GeographyRules, OutputSpec, TaskSpec
from core.geography import (
    all_country_names,
    expand_region_name,
    normalize_country_name,
    US_STATES,
    CANADIAN_PROVINCES,
    AUSTRALIAN_STATES,
    REGION_ALIASES,
    find_countries_in_text,
)

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
    "presence_countries": ["branches", "offices", "presence", "locations", "regional", "operating in", "active in"],
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
    "be", "but", "across", "serving", "serve", "serves", "serviced", "suitable",
    "prioritize", "prioritise", "especially", "include", "including",
    "foreign", "local", "egyptian", "active", "operating", "less-known", "unknown",
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
    "wireline services", "slickline services", "well logging services",
    "well intervention", "completion services", "stimulation services",
}

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

DOMAIN_KEYWORD_PATTERNS = {
    "wireline": [r"\bwireline\b", r"\bwire line\b", r"وايرلاين", r"ويرلاين"],
    "slickline": [r"\bslickline\b", r"\bslick line\b", r"سليكلاين", r"سلكلاين"],
    "e-line": [r"\be-line\b", r"\beline\b", r"\belectric line\b"],
    "well logging": [r"\bwell logging\b", r"\bwell log(?:ging)?\b", r"well\s*logging", r"تسجيل الابار", r"تسجيل الآبار"],
    "open hole logging": [r"\bopen[- ]hole logging\b"],
    "cased hole logging": [r"\bcased[- ]hole logging\b"],
    "perforation": [r"\bperforation\b", r"\bperforating\b"],
    "memory gauge": [r"\bmemory gauge(?:s)?\b"],
    "well intervention": [r"\bwell intervention\b"],
    "downhole tools": [r"\bdownhole tools?\b"],
    "completion": [r"\bcompletion services?\b", r"\bwell completion\b"],
    "coiled tubing": [r"\bcoiled tubing\b"],
    "well testing": [r"\bwell testing\b", r"\bwell test\b"],
    "stimulation": [r"\bstimulation\b", r"\bwell stimulation\b"],
    "acidizing": [r"\bacidizing\b", r"\bacidising\b"],
    "cementing": [r"\bcementing\b"],
    "mud logging": [r"\bmud logging\b"],
    "drilling fluids": [r"\bdrilling fluids\b", r"\bmud chemicals\b"],
    "directional drilling": [r"\bdirectional drilling\b"],
    "geosteering": [r"\bgeosteering\b"],
    "managed pressure drilling": [r"\bmanaged pressure drilling\b", r"\bmpd\b"],
    "drilling automation": [r"\bdrilling automation\b"],
    "drilling monitoring": [r"\bdrilling monitoring\b", r"\breal[- ]time drilling\b"],
    "drilling optimization": [r"\bdrilling optimization\b", r"\bdrilling optimisation\b"],
    "esp": [r"\besp\b", r"\belectrical submersible pump\b", r"\belectric submersible pump\b"],
    "artificial lift": [r"\bartificial lift\b"],
    "gas lift": [r"\bgas lift\b"],
    "rod pump": [r"\brod pump\b"],
    "virtual flow metering": [r"\bvirtual flow metering\b", r"\bvirtual flow meter\b", r"\bvirtual meter\b"],
    "multiphase metering": [r"\bmultiphase metering\b", r"\bmultiphase meter\b"],
    "well surveillance": [r"\bwell surveillance\b"],
    "production monitoring": [r"\bproduction monitoring\b"],
    "production optimization": [r"\bproduction optimization\b", r"\bproduction optimisation\b"],
    "well performance": [r"\bwell performance\b"],
    "flow assurance": [r"\bflow assurance\b"],
    "well integrity": [r"\bwell integrity\b"],
    "pipeline monitoring": [r"\bpipeline monitoring\b"],
    "leak detection": [r"\bleak detection\b"],
    "asset integrity": [r"\basset integrity\b"],
    "inspection": [r"\binspection\b", r"\bndt\b"],
    "corrosion monitoring": [r"\bcorrosion monitoring\b"],
    "reservoir simulation": [r"\breservoir simulation\b"],
    "reservoir modeling": [r"\breservoir modeling\b", r"\breservoir modelling\b"],
    "production engineering": [r"\bproduction engineering\b"],
}

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

NEGATIVE_CUE_RE = re.compile(
    r"\b(?:exclude|excluding|reject|remove|avoid|except|other than|"
    r"not\s+(?:in|from|inside|within)|outside|without|"
    r"no|has\s+no|with\s+no|do(?:es)?\s+not|don't|doesn't|cannot|can't)\b",
    re.IGNORECASE,
)

PRESENCE_NOUN_RE = re.compile(
    r"\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|"
    r"local entity|local entities|entity|entities|operations?|legal entity|legal entities|"
    r"registered entity|registered entities|distributor|distributors|agent|agents|"
    r"representative|representation|partner|partners)\b",
    re.IGNORECASE,
)

OUTSIDE_ACTIVITY_RE = re.compile(
    r"\b(?:operate(?:s|d|ing)?|work(?:s|ed|ing)?|serv(?:e|es|ed|ing)?|active|present)\s+outside\b",
    re.IGNORECASE,
)

NEGATED_ACTIVITY_IN_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|don't|doesn't|cannot|can't|not)\s+"
    r"(?:operate|work|serve|have|be\s+active|be\s+present)\b.*?\b(?:in|inside|within)\b",
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
        if not item:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _dedupe_repeated_prompt(prompt: str) -> str:
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


def _normalize_geo_aliases_in_text(text: str) -> str:
    out = text
    for alias, canonical in sorted(GEO_ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        out = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, out, flags=re.IGNORECASE)
    return out


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
    text = _normalize_geo_aliases_in_text(_normalize(text))
    if not text:
        return []

    found: List[str] = []
    all_geo_tokens = sorted(
        all_country_names() + list(_ALL_REGIONS) + list(GEO_ALIAS_MAP.values()),
        key=len,
        reverse=True,
    )

    for item in all_geo_tokens:
        if re.search(r"\b" + re.escape(item) + r"\b", text):
            found.append(item)

    return _dedupe_keep_order(found)


def _extract_country_list_from_text(text: str) -> List[str]:
    countries: List[str] = []
    try:
        countries.extend(find_countries_in_text(text))
    except Exception:
        pass
    for token in _find_geo_tokens_in_text(text):
        countries.extend(_expand_geo_name(token))
    return _dedupe_keep_order(countries)


def _split_geo_clauses(text: str) -> List[str]:
    text = _normalize_geo_aliases_in_text(_normalize(text))
    text = re.sub(
        r"\b(?:and|but)\s+(?=(?:do(?:es)?\s+not|don't|doesn't|excluding|exclude|except|without|avoid|remove|reject|other than|not\s+in|not\s+inside|not\s+within)\b)",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",\s*(?=(?:do(?:es)?\s+not|don't|doesn't|excluding|exclude|except|without|avoid|remove|reject|other than|not\s+in|not\s+inside|not\s+within)\b)",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"[.;:!?]\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_negative_sentence(sentence: str) -> bool:
    return bool(NEGATIVE_CUE_RE.search(sentence))


def _is_presence_exclusion_sentence(sentence: str) -> bool:
    return bool(
        PRESENCE_NOUN_RE.search(sentence)
        or OUTSIDE_ACTIVITY_RE.search(sentence)
        or NEGATED_ACTIVITY_IN_RE.search(sentence)
        or re.search(r"\b(?:outside|without presence|without offices?|without branches?)\b", sentence)
    )


def _extract_geography(prompt_lower: str) -> GeographyRules:
    geo_text = _normalize_geo_aliases_in_text(prompt_lower)

    include: List[str] = []
    exclude: List[str] = []
    exclude_presence: List[str] = []

    for clause in _split_geo_clauses(geo_text):
        countries = _extract_country_list_from_text(clause)
        if not countries:
            continue

        if _is_negative_sentence(clause):
            if _is_presence_exclusion_sentence(clause):
                exclude_presence.extend(countries)
            else:
                exclude.extend(countries)
        else:
            include.extend(countries)

    include = _dedupe_keep_order(include)
    exclude = _dedupe_keep_order(exclude)
    exclude_presence = _dedupe_keep_order(exclude_presence)

    exclude_presence = [c for c in exclude_presence if c not in include]
    exclude = [c for c in exclude if c not in include and c not in exclude_presence]

    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=bool(include or exclude or exclude_presence),
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
        r"\bdigital\b", r"\bsoftware\b", r"\bsaas\b", r"\bplatform(?:s)?\b",
        r"\banalytics\b", r"\bautomation\b", r"\bcloud\b", r"\biot\b",
        r"\bscada\b", r"\bapp(?:lication)?\b", r"\bdata company\b",
        r"\btech company\b", r"\btechnology (?:provider|vendor|company|companies)\b",
        r"\bai company\b", r"\bai companies\b", r"\bmachine learning\b",
        r"\bartificial intelligence\b", r"\bmonitoring\b", r"\boptimization\b", r"\boptimisation\b",
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
            "service", "services",
        },
        key=len,
        reverse=True,
    )

    for phrase in removable_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip(" ,-")
    if not text:
        return ""

    if re.search(r"\boil and gas\b", text):
        lead = re.sub(r"\boil and gas\b", " ", text).strip(" ,-")
        lead = re.sub(r"\s+", " ", lead).strip(" ,-")
        return f"{lead} oil and gas".strip() if lead else "oil and gas"

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
        r"\s+(?:outside|excluding|except|not\s+in|not\s+from|not\s+inside|not\s+within)\b.*$",
        r"\s+(?:operate|operates|operating|work|works|working|serve|serves|serving)\s+outside\b.*$",
        r"\s+(?:and\s+)?(?:do(?:es)?\s+not|don't|doesn't|cannot|can't)\s+(?:operate|work|serve|have|be)\b.*$",
        r"\s+(?:and\s+)?(?:excluding|exclude|except|avoid|without)\b.*$",
        r"\s+in\s+(?:europe|asia|africa|north america|south america|middle east|north africa|mena|cis|gcc|nordics|apac)\b.*$",
        r"\s+prioriti[sz]e\b.*$",
    ]
    for pat in split_patterns:
        text = re.split(pat, text)[0].strip()

    geo_words = set(all_country_names())
    geo_words.update(GEO_ALIAS_MAP.keys())
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

    return _normalize_industry_candidate(" ".join(words).strip())


def _extract_focus_term(prompt: str, prompt_lower: str, task_type: str) -> str:
    geo_words = set(all_country_names())
    geo_words.update(GEO_ALIAS_MAP.keys())
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
        "monitoring", "optimization", "optimisation", "machine", "learning",
        "artificial", "intelligence", "iot", "scada",
    }

    def _clean(raw: str) -> str:
        return _clean_topic_text(raw)

    def _valid(s: str) -> bool:
        return bool(s) and len(s) > 1 and s not in entity_words

    patterns = [
        r"(?:speciali[sz]ed in|focused on|focus on)\s+(.+?)\s+(?:for|in)\s+(?:the\s+)?oil(?:\s+and\s+gas|\s*&\s*gas)\b",
        r"(?:find|search for|get|show|list)\s+(.+?)\s+(?:service companies|service providers|contractors?|companies|vendors|firms)\b",
        r"(?:companies|vendors|firms|providers|contractors?)\s+(?:speciali[sz]ing|focused|working|operating|active)\s+in\s+(.+)",
        r"(?:companies|vendors|firms|providers|contractors?)\s+(?:in|for|serving)\s+(?:the\s+)?(.+?)\s+(?:sector|industry|space|field|market)\b",
        r"(?:working|operating|active|serving)\s+in\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"(?:papers?|studies|articles?|reports?|publications?|theses?|preprints?|research|literature)\s+(?:about|on|related to|concerning|regarding)\s+(.+)",
    ]
    for pat in patterns:
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
        is_meaningful = _valid(c) and c not in entity_words and (len(c.split()) >= 2 or c not in generic_qualifiers)
        if is_meaningful:
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
    candidate = _normalize_industry_candidate(" ".join(filtered[:8]).strip())

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


def _extract_target_attributes(prompt_lower: str, task_type: str, geography: GeographyRules) -> List[str]:
    found = []
    for attr, hints in ATTRIBUTE_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints):
            found.append(attr)

    if task_type == "document_research":
        attrs = sorted(set(found or ["website", "summary", "author"]) | {"author"})
    else:
        attrs = sorted(set(found or ["website"]))

    if geography.strict_mode:
        attrs = sorted(set(attrs) | {"presence_countries", "hq_country"})

    return attrs


def _extract_max_results(prompt: str) -> int:
    explicit = re.findall(
        r"\b(?:top|first|at least|around|about|up to|maximum|max|target)?\s*(\d{1,4})\s*"
        r"(?:results?|companies|vendors|firms|records?|entries|items?|papers?)\b",
        prompt,
        re.I,
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


def _finalize_industry(
    focus_term: str,
    prompt_lower: str,
    domain_keywords: List[str],
    solution_keywords: List[str],
) -> str:
    focus = _normalize_industry_candidate(focus_term)
    has_oil_gas = bool(re.search(r"\boil(?:\s+and\s+gas|\s*&\s*gas)\b", prompt_lower))

    if domain_keywords:
        domain_phrase = " ".join(domain_keywords[:4]).strip()
        if has_oil_gas and (not focus or focus == "oil and gas"):
            return _normalize_industry_candidate(f"{domain_phrase} oil and gas")
        if focus and focus != "oil and gas" and not any(k in focus for k in domain_keywords[:3]):
            return _normalize_industry_candidate(f"{domain_phrase} {focus}")

    if solution_keywords and has_oil_gas and (not focus or focus == "oil and gas"):
        solution_phrase = " ".join(solution_keywords[:3]).strip()
        return _normalize_industry_candidate(f"{solution_phrase} oil and gas")

    if has_oil_gas and not focus:
        return "oil and gas"

    return focus


def parse_task_prompt(prompt: str) -> TaskSpec:
    prompt = _dedupe_repeated_prompt((prompt or "").strip())
    prompt_lower = _normalize(prompt)

    task_type = _extract_task_type(prompt_lower)
    entity_types = _extract_entity_types(prompt_lower)
    target_category = _extract_target_category(prompt_lower)
    geography = _extract_geography(prompt_lower)
    output_format = _extract_output_format(prompt_lower)
    domain_keywords = _extract_domain_keywords(prompt_lower)
    solution_keywords = _extract_solution_keywords(prompt_lower)
    focus_term = _extract_focus_term(prompt, prompt_lower, task_type)
    industry = _finalize_industry(focus_term, prompt_lower, domain_keywords, solution_keywords)
    commercial_intent = _extract_commercial_intent(prompt_lower)
    target_attributes = _extract_target_attributes(prompt_lower, task_type, geography)
    max_results = _extract_max_results(prompt)

    ext = output_format if output_format != "ui_table" else "xlsx"
    filename = f"results.{ext}"

    return TaskSpec(
        raw_prompt=prompt,
        task_type=task_type,
        target_entity_types=entity_types,
        target_category=target_category,
        industry=industry,
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
