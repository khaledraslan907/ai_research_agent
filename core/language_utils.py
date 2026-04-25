from __future__ import annotations

"""Helpers for multilingual prompt/query handling."""

import re
from typing import Dict, Iterable, List

from core.geography import COUNTRY_ALIASES, CITY_TO_COUNTRY, REGION_ALIASES


_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")

_ARABIC_CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
    "ـ": "", "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": "",
})


EN_AR_GLOSSARY: Dict[str, List[str]] = {
    "company": ["شركة", "شركات"],
    "companies": ["شركات"],
    "paper": ["بحث", "ورقة علمية"],
    "papers": ["أبحاث", "ابحاث", "دراسات"],
    "people": ["أشخاص", "كوادر"],
    "profile": ["ملف شخصي"],
    "service company": ["شركة خدمات", "شركات خدمات"],
    "software company": ["شركة برمجيات", "شركة تقنية", "شركة رقمية"],
    "oil and gas": ["النفط والغاز", "البترول"],
    "energy": ["الطاقة"],
    "research": ["بحث", "أبحاث", "دراسة"],
    "find": ["ابحث", "ابحث عن", "اعثر على"],
    "egypt": ["مصر"],
    "saudi arabia": ["السعودية", "المملكة العربية السعودية"],
    "united arab emirates": ["الإمارات", "الامارات", "الإمارات العربية المتحدة"],
    "wireline": ["وايرلاين"],
    "well logging": ["تسجيل الآبار", "قياسات الآبار"],
    "machine learning": ["تعلم الآلة"],
    "artificial intelligence": ["ذكاء اصطناعي"],
}


def normalize_arabic(text: str) -> str:
    s = str(text or "").strip()
    s = s.translate(_ARABIC_CHAR_MAP)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_multilingual_text(text: str) -> str:
    s = normalize_arabic(str(text or "").strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(str(text or "")))


def contains_latin(text: str) -> bool:
    return bool(_LATIN_RE.search(str(text or "")))


def detect_languages(text: str) -> List[str]:
    langs: List[str] = []
    if contains_latin(text):
        langs.append("en")
    if contains_arabic(text):
        langs.append("ar")
    return langs or ["en"]


def choose_query_languages(prompt: str, geography_countries: Iterable[str] | None = None) -> List[str]:
    langs = detect_languages(prompt)
    geos = {str(x or "").strip().lower() for x in (geography_countries or []) if str(x or "").strip()}
    arab_market = {"egypt", "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain", "iraq", "jordan", "lebanon"}
    if geos & arab_market:
        for lang in ("en", "ar"):
            if lang not in langs:
                langs.append(lang)
    return langs


def bilingual_variants(term: str) -> List[str]:
    base = normalize_multilingual_text(term)
    variants = [term.strip()] if str(term or "").strip() else []
    if base in EN_AR_GLOSSARY:
        variants.extend(EN_AR_GLOSSARY[base])
    else:
        # reverse lookup
        for en, ar_list in EN_AR_GLOSSARY.items():
            if base == normalize_multilingual_text(en) or any(base == normalize_multilingual_text(x) for x in ar_list):
                variants.append(en)
                variants.extend(ar_list)
                break
    # country / city / region aliases
    for canonical, aliases in COUNTRY_ALIASES.items():
        terms = [canonical] + list(aliases)
        if any(base == normalize_multilingual_text(x) for x in terms):
            variants.extend(terms)
            break
    for canonical, country in CITY_TO_COUNTRY.items():
        if base == normalize_multilingual_text(canonical):
            variants.append(canonical)
            variants.append(country)
    for region, countries in REGION_ALIASES.items():
        if base == normalize_multilingual_text(region):
            variants.append(region)
            variants.extend(countries[:5])
    # dedupe keep order
    out: List[str] = []
    seen = set()
    for v in variants:
        s = str(v or "").strip()
        key = normalize_multilingual_text(s)
        if not s or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def expand_terms_multilingual(terms: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for term in terms or []:
        for variant in bilingual_variants(str(term or "")):
            key = normalize_multilingual_text(variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(variant)
    return out
