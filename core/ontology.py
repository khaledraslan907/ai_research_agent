from __future__ import annotations

"""
ontology.py
===========
Lightweight generic ontology and taxonomy helpers for the research agent.

Goals:
- stay backward-compatible with the current codebase
- support generic multi-industry search, not only oil & gas
- support bilingual English/Arabic alias matching where useful
- provide normalized concept extraction utilities that parsers and planners can reuse
"""

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Sequence


def normalize_label(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[\u0640\u200f\u200e]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class TaxonomyEntry:
    key: str
    aliases: List[str] = field(default_factory=list)
    weight: float = 1.0
    category: str = ""

    def all_terms(self) -> List[str]:
        return [self.key] + list(self.aliases or [])


ENTITY_TYPE_ALIASES: Dict[str, List[str]] = {
    "company": [
        "company", "companies", "vendor", "provider", "supplier", "firm", "contractor",
        "startup", "business", "enterprise", "شركة", "شركات", "مورد", "مزود", "مقاول",
    ],
    "paper": [
        "paper", "papers", "study", "studies", "article", "articles", "publication",
        "publications", "journal", "thesis", "literature", "ورقة", "اوراق", "أوراق",
        "بحث", "ابحاث", "أبحاث", "دراسة", "دراسات", "مقال", "مقالات",
    ],
    "person": [
        "person", "people", "profile", "profiles", "expert", "engineer", "manager",
        "director", "founder", "researcher", "consultant", "linkedin", "شخص", "أشخاص",
        "مهندس", "مدير", "باحث", "خبير", "لينكدان", "لينكد إن",
    ],
    "organization": ["organization", "organisation", "association", "university", "lab", "منظمة", "جامعة", "مختبر"],
    "event": ["event", "conference", "expo", "summit", "webinar", "exhibitor", "speaker", "مؤتمر", "معرض", "فعالية"],
    "product": ["product", "products", "tool", "software", "platform", "solution", "منتج", "أداة", "منصة", "حل"],
    "patent": ["patent", "patents", "براءة", "براءات"],
    "tender": ["tender", "rfp", "rfq", "procurement", "vendor list", "مناقصة", "عطاء", "توريد"],
    "dataset": ["dataset", "datasets", "data set", "benchmark", "بيانات", "مجموعة بيانات"],
    "news": ["news", "updates", "press release", "خبر", "أخبار", "اخبار"],
}


TARGET_CATEGORY_ALIASES: Dict[str, List[str]] = {
    "software_company": [
        "digital", "software", "saas", "platform", "analytics", "automation", "ai", "artificial intelligence",
        "machine learning", "iot", "scada", "cloud", "technology company", "tech company",
        "رقمي", "رقمية", "برمجيات", "منصة", "منصات", "ذكاء اصطناعي", "تحليلات", "أتمتة", "انترنت الاشياء",
    ],
    "service_company": [
        "service company", "service companies", "contractor", "contractors", "engineering services", "oilfield services",
        "inspection", "maintenance", "testing", "wireline", "slickline", "well logging",
        "شركة خدمات", "شركات خدمات", "خدمات هندسية", "مقاول", "صيانة", "تفتيش", "اختبار",
    ],
    "manufacturer": ["manufacturer", "manufacturing", "factory", "fabrication", "مصنع", "تصنيع", "تصنيع"],
    "consultancy": ["consulting", "consultancy", "advisor", "advisory", "استشارات", "استشاري"],
    "operator": ["operator", "operators", "مشغل", "مشغلين"],
    "general": [],
}


COMMERCIAL_INTENT_ALIASES: Dict[str, List[str]] = {
    "agent_or_distributor": [
        "agent", "agency", "distributor", "distribution", "representative", "representation",
        "local representative", "وكيل", "توكيل", "موزع", "توزيع", "ممثل محلي",
    ],
    "reseller": ["reseller", "resellers", "إعادة بيع", "ريسيلر"],
    "partner": ["partner", "partners", "channel partner", "alliance", "شريك", "شراكة"],
    "recruiting": ["hire", "hiring", "recruit", "recruitment", "وظائف", "توظيف"],
    "competitor_research": ["competitor", "competition", "منافس", "منافسين"],
    "supplier_discovery": ["supplier", "supplier discovery", "مورد", "مورّد"],
    "academic": ["research", "academic", "literature", "أكاديمي", "بحثي"],
    "general": [],
}


INDUSTRY_ALIASES: Dict[str, List[str]] = {
    "oil and gas": [
        "oil and gas", "oil & gas", "petroleum", "oilfield", "upstream", "midstream", "downstream",
        "النفط والغاز", "بترول", "البترول", "حقول النفط",
    ],
    "energy": ["energy", "power", "utilities", "الطاقة", "الكهرباء"],
    "manufacturing": ["manufacturing", "factory", "industrial", "food manufacturing", "manufacturing sector", "التصنيع", "صناعي"],
    "healthcare": ["healthcare", "medical", "hospital", "pharma", "life sciences", "الرعاية الصحية", "طبي", "دوائي"],
    "software": ["software", "saas", "enterprise software", "برمجيات", "برمجي"],
    "academia": ["academic", "research", "university", "lab", "أكاديمي", "جامعة", "مختبر"],
    "finance": ["finance", "banking", "fintech", "financial services", "تمويل", "بنوك", "فنتك"],
    "telecom": ["telecom", "telecommunications", "communications", "اتصالات", "اتصالات"],
    "construction": ["construction", "building", "infrastructure", "انشاءات", "بناء", "بنية تحتية"],
    "mining": ["mining", "metals", "minerals", "تعدين", "معادن"],
    "agriculture": ["agriculture", "agri", "farming", "زراعة"],
    "ccs": ["ccs", "carbon capture", "carbon capture and storage", "احتجاز الكربون"],
}


GENERIC_SOLUTION_TERMS: Dict[str, List[str]] = {
    "machine learning": ["machine learning", "ml", "تعلم الآلة"],
    "artificial intelligence": ["artificial intelligence", "ذكاء اصطناعي"],
    "ai": ["ai"],
    "analytics": ["analytics", "analytic", "تحليلات"],
    "monitoring": ["monitoring", "remote monitoring", "مراقبة"],
    "optimization": ["optimization", "optimisation", "تحسين"],
    "automation": ["automation", "automated", "أتمتة"],
    "iot": ["iot", "internet of things", "انترنت الاشياء", "إنترنت الأشياء"],
    "scada": ["scada"],
    "digital twin": ["digital twin", "توأم رقمي"],
    "predictive maintenance": ["predictive maintenance", "صيانة تنبؤية"],
}


OIL_GAS_DOMAIN_TERMS: Dict[str, List[str]] = {
    "wireline": ["wireline", "وايرلاين"],
    "slickline": ["slickline", "سليك لاين", "سلك لاين"],
    "e-line": ["e-line", "electric line"],
    "well logging": ["well logging", "logging", "تسجيل آبار", "قياسات الآبار"],
    "coiled tubing": ["coiled tubing", "كويلد تيوبنج"],
    "well testing": ["well testing", "اختبار الآبار"],
    "mud logging": ["mud logging", "تسجيل الطفلة"],
    "drilling": ["drilling", "حفر"],
    "drilling optimization": ["drilling optimization", "تحسين الحفر"],
    "esp": ["esp", "electrical submersible pump", "مضخة غاطسة كهربائية"],
    "artificial lift": ["artificial lift", "رفع صناعي"],
    "production optimization": ["production optimization", "تحسين الإنتاج"],
    "well surveillance": ["well surveillance", "مراقبة الآبار"],
    "multiphase metering": ["multiphase metering", "قياس متعدد الأطوار"],
    "flow assurance": ["flow assurance", "ضمان الانسياب"],
    "reservoir simulation": ["reservoir simulation", "محاكاة المكامن"],
    "reservoir modeling": ["reservoir modeling", "reservoir modelling", "نمذجة المكامن"],
    "pipeline inspection": ["pipeline inspection", "فحص خطوط الأنابيب"],
    "asset integrity": ["asset integrity", "سلامة الأصول"],
}


def _find_matches(text: str, alias_map: Dict[str, Sequence[str]]) -> List[str]:
    low = normalize_label(text)
    found: List[str] = []
    for canonical, aliases in alias_map.items():
        terms = [canonical] + list(aliases or [])
        if any(term and normalize_label(term) in low for term in terms):
            found.append(canonical)
    return found


def infer_entity_types(text: str) -> List[str]:
    matches = _find_matches(text, ENTITY_TYPE_ALIASES)
    return matches or ["company"]


def infer_target_category(text: str) -> str:
    scores: Dict[str, int] = {}
    low = normalize_label(text)
    for key, aliases in TARGET_CATEGORY_ALIASES.items():
        if key == "general":
            continue
        score = sum(1 for alias in aliases if normalize_label(alias) in low)
        if score:
            scores[key] = score
    if not scores:
        return "general"
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def infer_commercial_intent(text: str) -> str:
    scores: Dict[str, int] = {}
    low = normalize_label(text)
    for key, aliases in COMMERCIAL_INTENT_ALIASES.items():
        if key == "general":
            continue
        score = sum(1 for alias in aliases if normalize_label(alias) in low)
        if score:
            scores[key] = score
    if not scores:
        return "general"
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def infer_industries(text: str) -> List[str]:
    return _find_matches(text, INDUSTRY_ALIASES)


def extract_solution_keywords(text: str) -> List[str]:
    return _find_matches(text, GENERIC_SOLUTION_TERMS)


def extract_domain_keywords(text: str, industry: str = "") -> List[str]:
    found = _find_matches(text, OIL_GAS_DOMAIN_TERMS)
    if found:
        return found
    if normalize_label(industry) in {"oil and gas", "petroleum"}:
        return found
    return []


def collect_topic_terms(text: str, max_terms: int = 12) -> List[str]:
    low = normalize_label(text)
    tokens = re.findall(r"[\w\u0600-\u06FF][\w\-\u0600-\u06FF]{1,}", low)
    stop = {
        "find", "show", "search", "get", "companies", "company", "people", "papers", "paper", "profiles",
        "with", "and", "for", "the", "that", "this", "from", "inside", "outside", "operating", "working",
        "في", "من", "على", "عن", "الى", "إلى", "مع", "شركة", "شركات", "بحث", "ابحث", "أبحث",
    }
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in stop or len(tok) < 2:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def merge_unique(*groups: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        for item in group or []:
            s = str(item or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out
