from __future__ import annotations

import re
from typing import List

from core.task_models import CredentialMode, GeographyRules, OutputSpec, TaskSpec
from core.geography import (
    all_country_names,
    expand_region_name,
    normalize_country_name,
    normalize_geo_text,
    US_STATES,
    CANADIAN_PROVINCES,
    AUSTRALIAN_STATES,
    REGION_ALIASES,
)

# ---------------------------------------------------------------------------
# Hints and vocab
# ---------------------------------------------------------------------------
ENTITY_HINTS = {
    "company": [
        "company", "companies", "vendor", "vendors", "firm", "firms",
        "provider", "providers", "contractor", "contractors", "operator", "operators",
        "supplier", "suppliers", "startup", "startups", "corporation", "corporations",
        "enterprise", "enterprises", "agency", "agencies", "business", "businesses",
        "شركة", "شركات", "مقاول", "مقاولين", "مزود", "مزودين", "مورد", "موردين",
    ],
    "person": [
        "person", "people", "founder", "ceo", "manager", "contact person",
        "researcher", "expert", "consultant", "director", "executive",
        "scientist", "engineer", "professor", "linkedin", "profile", "profiles",
        "شخص", "أشخاص", "موظف", "موظفين", "مهندس", "مدير", "باحث", "خبير",
    ],
    "paper": [
        "paper", "papers", "journal", "study", "studies", "publication",
        "publications", "article", "articles", "report", "reports",
        "thesis", "theses", "preprint", "research", "literature",
        "ورقة", "أوراق", "بحث", "أبحاث", "دراسة", "دراسات", "مقال", "مقالات",
    ],
    "organization": [
        "organization", "organisation", "association", "society",
        "institute", "foundation", "ngo", "consortium", "alliance", "university",
        "منظمة", "جمعية", "معهد", "جامعة",
    ],
    "event": [
        "event", "conference", "expo", "summit", "workshop", "forum",
        "symposium", "webinar", "congress", "trade show",
        "مؤتمر", "معرض", "قمة", "فعالية", "حدث",
    ],
    "product": [
        "product", "products", "tool", "tools", "platform", "solution",
        "solutions", "app", "application", "software", "system",
        "منتج", "منتجات", "أداة", "أدوات", "منصة", "حل", "حلول", "برنامج", "نظام",
    ],
}

ATTRIBUTE_HINTS = {
    "website": ["website", "site", "url", "homepage", "link", "web", "موقع", "رابط"],
    "email": ["email", "emails", "mail", "contact email", "e-mail", "ايميل", "بريد"],
    "phone": [
        "phone", "telephone", "tel", "mobile", "contact number", "phone number",
        "contact details", "contact info", "contact information", "contact", "هاتف", "موبايل", "رقم",
    ],
    "linkedin": ["linkedin", "لينكدان"],
    "hq_country": ["hq", "headquarters", "head office", "based in", "headquartered", "المقر", "المركز الرئيسي"],
    "presence_countries": [
        "branches", "offices", "presence", "locations", "regional", "operating in", "active in",
        "فروع", "مكاتب", "تواجد", "تعمل في", "نشطة في",
    ],
    "summary": ["summary", "overview", "description", "abstract", "bio", "ملخص", "وصف"],
    "author": ["author", "authors", "written by", "authored by", "who wrote", "researcher", "researchers", "مؤلف", "مؤلفين"],
    "pdf": ["pdf", "full text", "download", "ملف pdf", "تحميل"],
}

FORMAT_HINTS = {
    "xlsx": ["excel", "xlsx", "spreadsheet", "xls"],
    "csv": ["csv"],
    "pdf": [
        "pdf report", "as pdf", "in pdf", "pdf file", "export pdf", "as a pdf",
        "pdf format", "pdf output", "pdf document", "export as pdf", "save as pdf",
        "output pdf", "export to pdf", "to pdf", "in pdf file", "as pdf file",
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
    "prioritize", "prioritise", "especially", "include", "including", "foreign", "local",
    "active", "less-known", "unknown", "finde", "ابحث", "اعطني", "أعطني", "اريد", "أريد",
    "في", "من", "الى", "إلى", "عن", "مع", "بدون", "خارج", "داخل", "شركة", "شركات",
    "خدمات", "بحث", "أبحاث", "ورقة", "أوراق", "اشخاص", "أشخاص", "منتجات", "معارض",
}

_REQUEST_NOISE = {
    "link", "links", "title", "titles", "author", "authors", "doi", "journal", "year", "date",
    "abstract", "volume", "issue", "page", "pages", "citation", "reference", "references",
    "number", "numbers", "contact", "details", "info", "information", "pdf", "csv", "excel",
    "xlsx", "xls", "json", "file", "files", "export", "download", "output", "format",
    "report", "document", "summary", "ملف", "ملفات", "صيغة", "تنزيل", "تحميل",
}

DIGITAL_CATEGORY_HINTS = {
    "digital", "software", "technology", "tech", "analytics", "automation", "platform", "platforms",
    "saas", "cloud", "ai", "artificial intelligence", "machine learning", "data", "iot", "scada",
    "digitalization", "digitization", "app", "application", "monitoring", "optimization", "optimisation",
    "رقمي", "برمجيات", "برنامج", "برامج", "منصة", "منصات", "تحليلات", "أتمتة", "ذكاء اصطناعي",
}

SERVICE_CATEGORY_HINTS = {
    "service company", "service companies", "contractor", "contractors", "field service", "services provider",
    "engineering services", "oilfield services", "drilling services", "field services", "maintenance",
    "inspection", "testing", "consulting services", "wireline services", "slickline services",
    "well logging services", "well intervention", "completion services", "stimulation services",
    "service", "services", "oilfield", "شركة خدمات", "شركات خدمات", "خدمات هندسية", "خدمات بترول",
    "خدمات النفط", "خدمات الغاز", "خدمات حقول النفط", "صيانة", "تفتيش", "اختبار", "مقاول",
}

# English + Arabic keyword capture for solution families
SOLUTION_KEYWORD_PATTERNS = {
    "machine learning": [r"\bmachine learning\b", r"\bml\b", r"تعلم الآلة"],
    "artificial intelligence": [r"\bartificial intelligence\b", r"ذكاء اصطناعي"],
    "ai": [r"\bai\b", r"\bA\.I\.?\b"],
    "analytics": [r"\banalytics\b", r"\banalytic\b", r"\binsights\b", r"تحليلات"],
    "monitoring": [r"\bmonitoring\b", r"\bremote monitoring\b", r"\bsurveillance\b", r"مراقبة"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b", r"\boptimizer\b", r"تحسين"],
    "automation": [r"\bautomation\b", r"\bautomated\b", r"\bautonomous\b", r"أتمتة"],
    "iot": [r"\biot\b", r"\binternet of things\b", r"انترنت الاشياء", r"إنترنت الأشياء"],
    "scada": [r"\bscada\b"],
    "digital twin": [r"\bdigital twin\b", r"\bdigital twins\b", r"توأم رقمي"],
    "predictive maintenance": [r"\bpredictive maintenance\b", r"صيانة تنبؤية"],
}

DOMAIN_KEYWORD_PATTERNS = {
    "wireline": [r"\bwireline\b", r"\bwire line\b", r"وايرلاين", r"wireline services"],
    "slickline": [r"\bslickline\b", r"\bslick line\b", r"سليك لاين", r"slickline services"],
    "e-line": [r"\be-line\b", r"\beline\b", r"\belectric line\b", r"electric line", r"إي لاين"],
    "well logging": [r"\bwell logging\b", r"\bwell log(?:ging)?\b", r"تسجيل الآبار", r"well logging services"],
    "open hole logging": [r"\bopen[- ]hole logging\b", r"open hole"],
    "cased hole logging": [r"\bcased[- ]hole logging\b", r"cased hole"],
    "perforation": [r"\bperforation\b", r"\bperforating\b", r"تثقيب"],
    "memory gauge": [r"\bmemory gauge(?:s)?\b"],
    "well intervention": [r"\bwell intervention\b", r"تدخل آبار"],
    "downhole tools": [r"\bdownhole tools?\b", r"معدات قاع البئر"],
    "completion": [r"\bcompletion services?\b", r"\bwell completion\b", r"إكمال الآبار"],
    "coiled tubing": [r"\bcoiled tubing\b", r"كويلد تيوبنج", r"coiled-tubing"],
    "well testing": [r"\bwell testing\b", r"\bwell test\b", r"اختبار الآبار"],
    "stimulation": [r"\bstimulation\b", r"\bwell stimulation\b", r"تحفيز الآبار"],
    "acidizing": [r"\bacidizing\b", r"\bacidising\b", r"تحميض"],
    "cementing": [r"\bcementing\b", r"إسمنت", r"cementing services"],
    "mud logging": [r"\bmud logging\b", r"mud-logging", r"mud logger", r"طين الحفر"],
    "drilling fluids": [r"\bdrilling fluids\b", r"\bmud chemicals\b", r"سوائل الحفر"],
    "directional drilling": [r"\bdirectional drilling\b", r"الحفر الاتجاهي"],
    "geosteering": [r"\bgeosteering\b"],
    "managed pressure drilling": [r"\bmanaged pressure drilling\b", r"\bmpd\b"],
    "drilling automation": [r"\bdrilling automation\b", r"أتمتة الحفر"],
    "drilling monitoring": [r"\bdrilling monitoring\b", r"\breal[- ]time drilling\b", r"مراقبة الحفر"],
    "drilling optimization": [r"\bdrilling optimization\b", r"\bdrilling optimisation\b", r"تحسين الحفر"],
    "esp": [r"\besp\b", r"\belectrical submersible pump\b", r"\belectric submersible pump\b"],
    "artificial lift": [r"\bartificial lift\b", r"الرفع الصناعي"],
    "gas lift": [r"\bgas lift\b"],
    "rod pump": [r"\brod pump\b", r"sucker rod pump", r"beam pump"],
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
    "inspection": [r"\binspection\b", r"\bndt\b", r"تفتيش"],
    "corrosion monitoring": [r"\bcorrosion monitoring\b"],
    "reservoir simulation": [r"\breservoir simulation\b"],
    "reservoir modeling": [r"\breservoir modeling\b", r"\breservoir modelling\b"],
    "production engineering": [r"\bproduction engineering\b"],
}

# ---------------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------------
GEO_ALIAS_MAP = {
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "usa": "usa",
    "u.s.a.": "usa",
    "u.s.": "usa",
    "us": "usa",
}

NEGATIVE_CUE_RE = re.compile(
    r"\b(?:exclude|excluding|reject|remove|avoid|except|other than|"
    r"not\s+(?:in|from|inside|within)|outside|without|"
    r"no|has\s+no|with\s+no|do(?:es)?\s+not|don't|doesn't|cannot|can't|"
    r"باستثناء|بدون|خارج|ليس في|لا يعمل|لا تعمل|لا توجد)\b",
    re.IGNORECASE,
)

PRESENCE_NOUN_RE = re.compile(
    r"\b(?:presence|office|offices|branch|branches|subsidiar(?:y|ies)|"
    r"local entity|local entities|entity|entities|operations?|legal entity|legal entities|"
    r"registered entity|registered entities|distributor|distributors|agent|agents|"
    r"representative|representation|partner|partners|office in|presence in|"
    r"مكتب|مكاتب|فرع|فروع|تواجد|وكيل|موزع|شريك)\b",
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


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return normalize_geo_text(text or "")


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
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
    return raw


def _normalize_geo_aliases_in_text(text: str) -> str:
    out = _normalize(text)
    for alias, canonical in sorted(GEO_ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        out = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, out, flags=re.IGNORECASE)
    return out


def _expand_geo_name(name: str) -> List[str]:
    name = _normalize(name)
    if not name:
        return []
    if name in GEO_ALIAS_MAP:
        return [GEO_ALIAS_MAP[name]]
    expanded = expand_region_name(name)
    if expanded:
        return list(expanded)
    norm = normalize_country_name(name)
    return [norm] if norm else []


def _find_geo_tokens_in_text(text: str) -> List[str]:
    text = _normalize_geo_aliases_in_text(text)
    if not text:
        return []
    found: List[str] = []
    all_geo_tokens = sorted(
        list(all_country_names()) + list(_ALL_REGIONS) + list(GEO_ALIAS_MAP.values()),
        key=len,
        reverse=True,
    )
    for item in all_geo_tokens:
        if re.search(r"\b" + re.escape(item) + r"\b", text):
            found.append(item)
    return _dedupe_keep_order(found)


def _extract_country_list_from_text(text: str) -> List[str]:
    countries: List[str] = []
    for token in _find_geo_tokens_in_text(text):
        countries.extend(_expand_geo_name(token))
    return _dedupe_keep_order(countries)


def _split_geo_clauses(text: str) -> List[str]:
    text = _normalize_geo_aliases_in_text(text)
    text = re.sub(
        r"\b(?:and|but|و|لكن)\s+(?=(?:do(?:es)?\s+not|don't|doesn't|excluding|exclude|except|without|avoid|remove|reject|other than|not\s+in|not\s+inside|not\s+within|باستثناء|بدون|خارج|ليس في|لا يعمل|لا تعمل)\b)",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"[.;:!?،]\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_negative_sentence(sentence: str) -> bool:
    return bool(NEGATIVE_CUE_RE.search(sentence))


def _is_presence_exclusion_sentence(sentence: str) -> bool:
    return bool(
        PRESENCE_NOUN_RE.search(sentence)
        or OUTSIDE_ACTIVITY_RE.search(sentence)
        or NEGATED_ACTIVITY_IN_RE.search(sentence)
        or re.search(r"\b(?:outside|without presence|without offices?|without branches?|خارج|بدون تواجد|بدون مكاتب|بدون فروع)\b", sentence)
    )


def _extract_geography(prompt_lower: str) -> GeographyRules:
    include: List[str] = []
    exclude: List[str] = []
    exclude_presence: List[str] = []

    for clause in _split_geo_clauses(prompt_lower):
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
    exclude_presence = [c for c in _dedupe_keep_order(exclude_presence) if c not in include]
    exclude = [c for c in _dedupe_keep_order(exclude) if c not in include and c not in exclude_presence]
    return GeographyRules(
        include_countries=include,
        exclude_countries=exclude,
        exclude_presence_countries=exclude_presence,
        strict_mode=bool(include or exclude or exclude_presence),
    )


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------
def _extract_entity_types(prompt_lower: str) -> List[str]:
    found = []
    for entity_type, hints in ENTITY_HINTS.items():
        if any(re.search(r"\b" + re.escape(h) + r"\b", prompt_lower) for h in hints if re.fullmatch(r"[A-Za-z0-9 .&/-]+", h)):
            found.append(entity_type)
        elif any(h in prompt_lower for h in hints if not re.fullmatch(r"[A-Za-z0-9 .&/-]+", h)):
            found.append(entity_type)
    if "paper" in found and "person" in found:
        explicit_person = any(w in prompt_lower for w in ["people", "researcher", "expert", "consultant", "ceo", "founder", "director", "موظف", "باحث"])
        if not explicit_person:
            found.remove("person")
    return found or ["company"]


def _extract_output_format(prompt_lower: str) -> str:
    for fmt, hints in FORMAT_HINTS.items():
        if any(h in prompt_lower for h in hints):
            return fmt
    return "xlsx"


def _extract_task_type(prompt_lower: str) -> str:
    if any(x in prompt_lower for x in ["enrich", "append", "fill missing", "complete list", "add email", "add phone", "update list", "استكمال", "إثراء"]):
        return "entity_enrichment"
    if any(x in prompt_lower for x in ["similar companies", "similar vendors", "similar to", "alternatives to", "companies like", "مشابه", "بدائل"]):
        return "similar_entity_expansion"
    if any(x in prompt_lower for x in ["market map", "landscape", "market research", "industry map", "competitive landscape", "market overview", "خريطة سوق", "دراسة سوق"]):
        return "market_research"
    if any(x in prompt_lower for x in [
        "paper", "papers", "study", "studies", "publication", "publications", "article", "articles",
        "report", "reports", "journal", "thesis", "research on", "literature", "بحث", "أبحاث", "ورقة", "أوراق", "دراسة", "مقال",
    ]):
        return "document_research"
    if any(x in prompt_lower for x in [
        "linkedin", "linkedin profile", "linkedin account", "people who work", "people working", "professionals in",
        "find engineers", "find managers", "find hr", "find people", "employees in", "staff in", "workers in",
        "ملفات linkedin", "لينكدان", "موظفين", "مهندسين", "مديرين",
    ]):
        return "people_search"
    return "entity_discovery"


def _extract_target_category(prompt_lower: str) -> str:
    if any(x in prompt_lower for x in SERVICE_CATEGORY_HINTS):
        return "service_company"
    if any(x in prompt_lower for x in DIGITAL_CATEGORY_HINTS):
        return "software_company"
    return "general"


def _normalize_industry_candidate(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"\boil\s*(?:&|and)?\s*gas\b", "oil and gas", text)
    text = re.sub(r"\bpetroleum\b", "oil and gas", text)
    text = re.sub(r"\bنفط وغاز\b", "oil and gas", text)
    text = re.sub(r"\bقطاع البترول\b", "oil and gas", text)

    removable_phrases = sorted(
        set(DIGITAL_CATEGORY_HINTS) | set(SERVICE_CATEGORY_HINTS) | {
            "company", "companies", "vendor", "vendors", "provider", "providers", "firm", "firms", "service", "services",
            "technology company", "technology companies", "software company", "software companies", "digital company", "digital companies",
            "شركة", "شركات", "خدمات",
        },
        key=len,
        reverse=True,
    )
    for phrase in removable_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip(" ,-./")
    return text


def _clean_topic_text(text: str) -> str:
    text = _normalize(text)
    split_patterns = [
        r"\s+and\s+export\b.*$",
        r"\s+export\b.*$",
        r"\s+as\s+(?:a\s+)?pdf\b.*$",
        r"\s+with\s+(?:email|phone|contact|number|linkedin|website|url|tel|mobile|link|title|author)\b.*$",
        r"\s+without\s+(?:email|phone|contact|number|linkedin)\b.*$",
        r"\s+(?:outside|excluding|except|not\s+in|not\s+from|not\s+inside|not\s+within)\b.*$",
        r"\s+(?:and\s+)?(?:do(?:es)?\s+not|don't|doesn't|cannot|can't)\s+(?:operate|work|serve|have|be)\b.*$",
        r"\s+(?:and\s+)?(?:excluding|exclude|except|avoid|without)\b.*$",
        r"\s+in\s+(?:europe|asia|africa|north america|south america|middle east|north africa|mena|cis|gcc|nordics|apac)\b.*$",
    ]
    for pat in split_patterns:
        text = re.split(pat, text)[0].strip()
    return _normalize_industry_candidate(text)


def _extract_focus_term(prompt: str, prompt_lower: str, task_type: str, domain_keywords: List[str], solution_keywords: List[str]) -> str:
    # 1) document topic specific
    if task_type == "document_research":
        m = re.search(r"(?:papers?|studies|articles?|reports?|publications?|research|literature)\s+(?:about|on|related to|concerning|regarding)\s+(.+)", prompt_lower)
        if m:
            candidate = _clean_topic_text(m.group(1))
            if candidate:
                return candidate

    # 2) sector / industry patterns
    patterns = [
        r"working\s+in\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"(?:in|within|for)\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"serving\s+(?:the\s+)?(.+?)\s+(?:industry|sector|space|field)\b",
        r"(?:قطاع|مجال|صناعة)\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, prompt_lower)
        if m:
            candidate = _clean_topic_text(m.group(1))
            if candidate:
                return candidate

    sector_terms = []
    if re.search(r"\boil and gas\b|\bpetroleum\b|نفط|غاز|بترول", prompt_lower):
        sector_terms.append("oil and gas")
    if re.search(r"\benergy\b|طاقة", prompt_lower):
        sector_terms.append("energy")
    if re.search(r"\bccs\b|carbon capture|احتجاز الكربون", prompt_lower):
        sector_terms.append("ccs")

    pieces = []
    if domain_keywords:
        pieces.append(", ".join(domain_keywords[:4]))
    elif solution_keywords:
        pieces.append(", ".join(solution_keywords[:4]))
    if sector_terms:
        pieces.append(sector_terms[0])

    topic = " in ".join([pieces[0], pieces[1]]) if len(pieces) >= 2 else (pieces[0] if pieces else "")
    topic = topic.replace(", ", " / ")
    return topic or (sector_terms[0] if sector_terms else "")


def _extract_solution_keywords(prompt_lower: str) -> List[str]:
    found = []
    for label, patterns in SOLUTION_KEYWORD_PATTERNS.items():
        if any(re.search(pat, prompt_lower, flags=re.IGNORECASE) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)


def _extract_domain_keywords(prompt_lower: str) -> List[str]:
    found = []
    for label, patterns in DOMAIN_KEYWORD_PATTERNS.items():
        if any(re.search(pat, prompt_lower, flags=re.IGNORECASE) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)


def _extract_commercial_intent(prompt_lower: str) -> str:
    if re.search(r"\b(agent|agency|distributor|distribution|local representation|representative|representation|وكيل|موزع)\b", prompt_lower):
        return "agent_or_distributor"
    if re.search(r"\b(reseller|resellers|موزع معتمد)\b", prompt_lower):
        return "reseller"
    if re.search(r"\b(partner|partners|channel partner|alliance|partner program|شريك|شراكة)\b", prompt_lower):
        return "partner"
    return "general"


def _extract_target_attributes(prompt_lower: str, task_type: str, geography: GeographyRules) -> List[str]:
    found = []
    for attr, hints in ATTRIBUTE_HINTS.items():
        if any(h in prompt_lower for h in hints):
            found.append(attr)

    if task_type == "document_research":
        attrs = sorted(set(found or ["website", "summary", "author"]) | {"author"})
    elif task_type == "people_search":
        attrs = sorted(set(found or ["linkedin", "website"]))
    else:
        attrs = sorted(set(found or ["website"]))

    if geography.strict_mode:
        attrs = sorted(set(attrs) | {"presence_countries", "hq_country"})
    return attrs


def _extract_max_results(prompt: str) -> int:
    explicit = re.findall(
        r"\b(?:top|first|at least|around|about|up to|maximum|max)?\s*(\d{1,4})\s*(?:results?|companies|vendors|firms|records?|entries|items?|papers?|profiles?)\b",
        prompt,
        re.I,
    )
    for m in explicit:
        v = int(m)
        if 1 <= v <= 5000:
            return v
    return 25


def parse_task_prompt(prompt: str) -> TaskSpec:
    prompt = _dedupe_repeated_prompt((prompt or "").strip())
    prompt_lower = _normalize_geo_aliases_in_text(prompt)

    task_type = _extract_task_type(prompt_lower)
    entity_types = _extract_entity_types(prompt_lower)
    target_category = _extract_target_category(prompt_lower)
    geography = _extract_geography(prompt_lower)
    output_format = _extract_output_format(prompt_lower)
    solution_keywords = _extract_solution_keywords(prompt_lower)
    domain_keywords = _extract_domain_keywords(prompt_lower)
    focus_term = _extract_focus_term(prompt, prompt_lower, task_type, domain_keywords, solution_keywords)
    commercial_intent = _extract_commercial_intent(prompt_lower)
    target_attributes = _extract_target_attributes(prompt_lower, task_type, geography)
    max_results = _extract_max_results(prompt)

    if not focus_term:
        # sensible generic fallbacks
        if task_type == "document_research":
            focus_term = "research"
        elif target_category == "service_company" and domain_keywords:
            focus_term = " / ".join(domain_keywords[:4])
        elif target_category == "software_company" and solution_keywords:
            focus_term = " / ".join(solution_keywords[:4])
        else:
            focus_term = "general"

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
