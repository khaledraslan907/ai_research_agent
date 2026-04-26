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
)

ENTITY_HINTS = {
    "company": [
        "company", "companies", "vendor", "vendors", "firm", "firms",
        "provider", "providers", "contractor", "contractors", "operator", "operators",
        "supplier", "suppliers", "startup", "startups", "corporation", "corporations",
        "enterprise", "enterprises", "agency", "agencies", "business", "businesses",
        "exhibitor", "exhibitors",
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
        "symposium", "webinar", "congress", "trade show", "exhibitor", "exhibitors",
        "egyps",
    ],
    "product": [
        "product", "products", "tool", "tools", "platform", "solution",
        "solutions", "app", "application", "software", "system",
    ],
}

ATTRIBUTE_HINTS = {
    "website": ["website", "site", "url", "homepage", "link", "web", "company names and websites"],
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
    "exhibitors": ["exhibitor", "exhibitors"],
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
    "extract", "names", "name", "websites", "website", "email", "emails", "phone", "phones",
}

_REQUEST_NOISE = {
    "link", "links", "title", "titles", "author", "authors",
    "doi", "journal", "year", "date", "abstract", "volume",
    "issue", "page", "pages", "citation", "reference", "references",
    "number", "numbers", "contact", "details", "info", "information",
    "pdf", "csv", "excel", "xlsx", "xls", "json",
    "file", "files", "export", "download", "output", "format",
    "report", "document", "summary", "deadline", "buyer",
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
    "wireline": [r"\bwireline\b", r"\bwire line\b"],
    "well logging": [r"\bwell logging\b", r"\bwell log(?:ging)?\b"],
    "pipeline inspection": [r"\bpipeline inspection\b"],
    "food manufacturing": [r"\bfood manufacturing\b", r"\bfood processing\b"],
}

ARABIC_GEO_ALIAS_MAP = {
    "مصر": "egypt",
    "جمهورية مصر العربية": "egypt",
    "الخليج": "gcc",
    "دول الخليج": "gcc",
    "السعودية": "saudi arabia",
    "المملكة العربية السعودية": "saudi arabia",
    "الإمارات": "united arab emirates",
    "الامارات": "united arab emirates",
    "الإمارات العربية المتحدة": "united arab emirates",
    "بريطانيا": "united kingdom",
    "المملكة المتحدة": "united kingdom",
    "أمريكا": "united states",
    "الولايات المتحدة": "united states",
}

GEO_ALIAS_MAP = {
    "uk": "united kingdom", "u.k.": "united kingdom",
    "uae": "united arab emirates", "u.a.e.": "united arab emirates",
    "usa": "united states", "u.s.a.": "united states", "u.s.": "united states",
    "us": "united states", "gcc": "gcc", "خليج": "gcc", "الخليج": "gcc", "دول الخليج": "gcc",
    "egyps": "egypt",
}
_ALL_SUBNATIONAL = {}
_ALL_SUBNATIONAL.update(US_STATES)
_ALL_SUBNATIONAL.update(CANADIAN_PROVINCES)
_ALL_SUBNATIONAL.update(AUSTRALIAN_STATES)
_ALL_REGIONS = set(REGION_ALIASES.keys()) | {"gcc"}

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
def _dedupe_keep_order(items: List[str]) -> List[str]:
    out=[]; seen=set()
    for item in items:
        s=str(item).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _normalize_geo_aliases_in_text(text: str) -> str:
    out = text
    for alias, canonical in sorted(GEO_ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        out = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, out, flags=re.IGNORECASE)
    return out

def _expand_geo_name(name: str) -> List[str]:
    name=(name or "").strip().lower()
    if name=="gcc":
        return ["saudi arabia","united arab emirates","qatar","oman","kuwait","bahrain"]
    expanded=expand_region_name(name)
    if expanded: return list(expanded)
    norm=normalize_country_name(name)
    return [norm] if norm else []

def _find_geo_tokens_in_text(text: str) -> List[str]:
    text = _normalize_geo_aliases_in_text(_normalize(text))
    found=[]
    all_geo_tokens = sorted(all_country_names()+list(_ALL_REGIONS)+list(GEO_ALIAS_MAP.values()), key=len, reverse=True)
    for item in all_geo_tokens:
        if re.search(r"\b" + re.escape(item) + r"\b", text):
            found.append(item)
    return _dedupe_keep_order(found)

def _extract_country_list_from_text(text:str)->List[str]:
    countries=[]
    for token in _find_geo_tokens_in_text(text):
        countries.extend(_expand_geo_name(token))
    return _dedupe_keep_order(countries)

def _split_geo_clauses(text:str)->List[str]:
    return [p.strip() for p in re.split(r"[.;:!?]\s*|\n+", _normalize_geo_aliases_in_text(_normalize(text))) if p.strip()]

def _extract_geography(prompt_lower:str)->GeographyRules:
    include=[]
    for clause in _split_geo_clauses(prompt_lower):
        include.extend(_extract_country_list_from_text(clause))
    return GeographyRules(include_countries=_dedupe_keep_order(include), exclude_countries=[], exclude_presence_countries=[], strict_mode=bool(include))

def _extract_entity_types(prompt_lower:str)->List[str]:
    if re.search(r"\b(tender|tenders|rfq|rfp|itt|itb|procurement|bid|bids)\b", prompt_lower):
        return ["tender"]
    if re.search(r"\b(product|products|platform|platforms|tool|tools)\b", prompt_lower):
        return ["product"]
    if re.search(r"\b(exhibitor|exhibitors|egyps|conference|expo|trade show)\b", prompt_lower):
        return ["company"]
    if re.search(r"\b(paper|papers|study|studies|publication|publications|article|articles|research|literature)\b", prompt_lower):
        return ["paper"]
    if re.search(r"\b(linkedin|profiles?|people|engineers?|managers?|recruiters?|hr)\b", prompt_lower):
        return ["person"]
    found=[]
    for entity_type, hints in ENTITY_HINTS.items():
        if any(re.search(r"\b"+re.escape(h)+r"\b", prompt_lower) for h in hints):
            found.append(entity_type)
    return ["company"] if not found else [("company" if "company" in found else found[0])]

def _extract_task_type(prompt_lower:str)->str:
    if re.search(r"\b(paper|papers|study|studies|publication|publications|article|articles|research|literature)\b", prompt_lower):
        return "document_research"
    if re.search(r"\b(linkedin|profiles?|people|engineers?|managers?|recruiters?|hr)\b", prompt_lower):
        return "people_search"
    if re.search(r"\b(tender|tenders|rfq|rfp|itt|itb|procurement|bid|bids)\b", prompt_lower):
        return "market_research"
    if re.search(r"\b(exhibitor|exhibitors|egyps|conference|expo|trade show)\b", prompt_lower):
        return "market_research"
    return "entity_discovery"

def _extract_target_category(prompt_lower:str)->str:
    # service/company-event first, including Arabic petroleum-service wording
    if re.search(r"(شركات خدمات البترول|خدمات البترول|خدمات النفط|oilfield service|petroleum service|service companies?|contractors?|wireline|slickline|well logging|exhibitor|exhibitors)", prompt_lower):
        return "service_company"
    if re.search(r"\b(product|products|platform|platforms|software companies?|software vendors?|digital|saas|automation|mes|factory automation)\b", prompt_lower) or any(h in prompt_lower for h in DIGITAL_CATEGORY_HINTS):
        return "software_company"
    return "general"

def _clean_topic_text(text:str)->str:
    text = _normalize(text)
    for pat in [
        r"\bwith\s+(website|websites|email|emails|phone|phones|linkedin|summary|hq|headquarters)\b.*$",
        r"\bextract\s+(company names?|names?|websites?|deadline|buyer)(?:\s+and\s+(company names?|names?|websites?|deadline|buyer))*\b.*$",
        r"\bcompany names?\b", r"\bwebsites?\b", r"\bdeadline\b", r"\bbuyer\b",
        r"\begyps\b", r"\bexhibitors?\b",
    ]:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip(" ,.-")
    return _normalize(text)

def _extract_focus_term(prompt:str, prompt_lower:str, task_type:str)->str:
    if re.search(r"(food manufacturing|food processing)", prompt_lower):
        return "food manufacturing"
    if re.search(r"(pipeline inspection)", prompt_lower):
        return "pipeline inspection"
    if re.search(r"(multiphase metering)", prompt_lower):
        return "multiphase metering in oil and gas" if re.search(r"oil and gas|petroleum", prompt_lower) else "multiphase metering"
    if re.search(r"(wireline|well logging|slickline)", prompt_lower):
        return "wireline well logging"
    if re.search(r"(nanobubbles).*(enhanced oil recovery|eor)|((enhanced oil recovery|eor).*(nanobubbles))", prompt_lower):
        return "nanobubbles in enhanced oil recovery"
    if re.search(r"(electrical submersible pump|\besp\b)", prompt_lower):
        return "electrical submersible pump"
    if re.search(r"(شركات خدمات البترول|خدمات البترول|خدمات النفط|oilfield service|petroleum service|oil and gas service)", prompt_lower):
        return "oil and gas"
    words = [w for w in re.split(r"[\s,/]+", _clean_topic_text(prompt_lower)) if w and w not in GENERIC_STOPWORDS and w not in _REQUEST_NOISE]
    return " ".join(words[:4]).strip()

def _extract_solution_keywords(prompt_lower:str)->List[str]:
    # do not treat plain "software" as technical solution keyword in generic B2B software searches
    found=[]
    for label, patterns in SOLUTION_KEYWORD_PATTERNS.items():
        if label == "software":
            continue
        if any(re.search(pat, prompt_lower) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)

def _extract_domain_keywords(prompt_lower:str)->List[str]:
    found=[]
    for label, patterns in DOMAIN_KEYWORD_PATTERNS.items():
        if label in {"food manufacturing"}:
            continue
        if any(re.search(pat, prompt_lower) for pat in patterns):
            found.append(label)
    return _dedupe_keep_order(found)

def _extract_commercial_intent(prompt_lower:str)->str:
    if re.search(r"\b(agent|agency|distributor|distribution|local representative|representative|representation)\b", prompt_lower):
        return "agent_or_distributor"
    if re.search(r"\b(reseller|resellers)\b", prompt_lower):
        return "reseller"
    if re.search(r"\b(partner|partners|channel partner|channel partners)\b", prompt_lower):
        return "partner"
    return "general"

def _extract_target_attributes(prompt_lower:str)->List[str]:
    found=["website"]
    if "email" in prompt_lower or "الإيميل" in prompt_lower or "البريد" in prompt_lower:
        found.append("email")
    if re.search(r"\b(phone|phones|telephone|contact number|رقم الهاتف|هاتف)\b", prompt_lower):
        found.append("phone")
    if re.search(r"\b(linkedin|profile url|profile link)\b", prompt_lower):
        found.append("linkedin")
    if re.search(r"\b(doi|authors?|abstract|title)\b", prompt_lower):
        if "title" in prompt_lower: found.append("title")
        if re.search(r"\bauthors?\b", prompt_lower): found.append("author")
        if "doi" in prompt_lower: found.append("doi")
        if "abstract" in prompt_lower: found.append("summary")
    if re.search(r"\b(deadline|due date|closing date)\b", prompt_lower):
        found.append("deadline")
    if re.search(r"\b(buyer|issuer|procuring entity|tendering authority)\b", prompt_lower):
        found.append("buyer")
    return _dedupe_keep_order(found)

def _extract_output_format(prompt_lower:str)->str:
    for fmt, hints in FORMAT_HINTS.items():
        if any(h in prompt_lower for h in hints):
            return fmt
    return "xlsx"

def _extract_max_results(prompt:str)->int:
    m=re.search(r"\b(\d{1,3})\s+results?\b", prompt, re.IGNORECASE)
    if m:
        try: return max(1, min(500, int(m.group(1))))
        except: pass
    return 25

def parse_task_prompt(prompt: str) -> TaskSpec:
    raw = (prompt or "").strip()
    prompt_lower = _normalize_geo_aliases_in_text(_normalize(raw))
    task_type = _extract_task_type(prompt_lower)
    entity_types = _extract_entity_types(prompt_lower)
    target_category = _extract_target_category(prompt_lower)
    geography = _extract_geography(prompt_lower)
    industry = _extract_focus_term(raw, prompt_lower, task_type)
    solution_keywords = _extract_solution_keywords(prompt_lower)
    domain_keywords = _extract_domain_keywords(prompt_lower)
    target_attributes = _extract_target_attributes(prompt_lower)
    return TaskSpec(
        raw_prompt=raw,
        task_type=task_type,
        target_entity_types=entity_types,
        target_category=target_category,
        industry=industry or ("oil and gas" if target_category=="service_company" else ""),
        solution_keywords=solution_keywords,
        domain_keywords=domain_keywords,
        commercial_intent=_extract_commercial_intent(prompt_lower),
        target_attributes=target_attributes,
        geography=geography,
        output=OutputSpec(format=_extract_output_format(prompt_lower), filename="results.xlsx"),
        credential_mode=CredentialMode(mode="free"),
        max_results=_extract_max_results(raw),
        mode="Balanced",
    )
