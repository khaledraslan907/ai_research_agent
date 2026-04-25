"""
geography.py
============
Generic, bilingual geographic resolution for the AI Research Agent.

Goals:
- Keep backward compatibility with the existing project
- Support English + Arabic country / region / city aliases
- Resolve countries, regions, states, provinces, and major cities
- Make downstream geography filtering stricter and easier to debug
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def _normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


# ── Country aliases ────────────────────────────────────────────────────────
COUNTRY_ALIASES: Dict[str, List[str]] = {
    # Americas
    "usa": [
        "usa", "us", "u.s.", "u.s.a.", "united states", "united states of america",
        "america", "the us", "the usa", "the united states",
        "الولايات المتحدة", "الولايات المتحده", "الولايات المتحدة الامريكية",
        "الولايات المتحدة الأمريكية", "امريكا", "أمريكا",
    ],
    "canada": ["canada", "ca", "كندا"],
    "mexico": ["mexico", "méxico", "المكسيك"],
    "brazil": ["brazil", "brasil", "البرازيل"],
    "argentina": ["argentina", "الأرجنتين", "الارجنتين"],
    "chile": ["chile", "تشيلي"],
    "colombia": ["colombia", "كولومبيا"],
    "peru": ["peru", "perú", "بيرو"],
    "venezuela": ["venezuela", "فنزويلا"],
    "ecuador": ["ecuador", "الإكوادور", "الاكوادور"],
    "bolivia": ["bolivia", "بوليفيا"],
    "uruguay": ["uruguay", "أوروغواي", "اوروجواي"],
    "paraguay": ["paraguay", "باراغواي"],

    # Europe
    "united kingdom": [
        "uk", "u.k.", "united kingdom", "britain", "great britain",
        "england", "scotland", "wales", "northern ireland",
        "المملكة المتحدة", "بريطانيا", "بريطانيا العظمى", "انجلترا", "إنجلترا",
    ],
    "ireland": ["ireland", "eire", "ايرلندا", "إيرلندا"],
    "france": ["france", "فرنسا"],
    "germany": ["germany", "deutschland", "المانيا", "ألمانيا"],
    "italy": ["italy", "italia", "ايطاليا", "إيطاليا"],
    "spain": ["spain", "españa", "اسبانيا", "إسبانيا"],
    "portugal": ["portugal", "البرتغال"],
    "netherlands": ["netherlands", "holland", "the netherlands", "هولندا", "البلاد المنخفضة"],
    "belgium": ["belgium", "بلجيكا"],
    "switzerland": ["switzerland", "سويسرا"],
    "austria": ["austria", "النمسا"],
    "norway": ["norway", "النرويج"],
    "sweden": ["sweden", "السويد"],
    "denmark": ["denmark", "الدنمارك"],
    "finland": ["finland", "فنلندا"],
    "poland": ["poland", "بولندا"],
    "czech republic": ["czech republic", "czechia", "czech", "التشيك", "جمهورية التشيك"],
    "romania": ["romania", "رومانيا"],
    "greece": ["greece", "اليونان"],
    "turkey": ["turkey", "türkiye", "تركيا"],
    "hungary": ["hungary", "هنغاريا", "المجر"],
    "ukraine": ["ukraine", "أوكرانيا", "اوكرانيا"],
    "russia": ["russia", "russian federation", "روسيا", "الاتحاد الروسي"],
    "serbia": ["serbia", "صربيا"],
    "croatia": ["croatia", "كرواتيا"],
    "bulgaria": ["bulgaria", "بلغاريا"],
    "slovakia": ["slovakia", "سلوفاكيا"],
    "slovenia": ["slovenia", "سلوفينيا"],
    "estonia": ["estonia", "استونيا", "إستونيا"],
    "latvia": ["latvia", "لاتفيا"],
    "lithuania": ["lithuania", "ليتوانيا"],
    "luxembourg": ["luxembourg", "لوكسمبورغ"],
    "iceland": ["iceland", "آيسلندا", "ايسلندا"],

    # Middle East
    "saudi arabia": ["saudi arabia", "ksa", "kingdom of saudi arabia", "السعودية", "المملكة العربية السعودية"],
    "united arab emirates": [
        "uae", "u.a.e", "u.a.e.", "united arab emirates", "emirates", "the uae",
        "الامارات", "الإمارات", "الامارات العربية المتحدة", "الإمارات العربية المتحدة",
    ],
    "qatar": ["qatar", "قطر"],
    "oman": ["oman", "sultanate of oman", "عمان", "سلطنة عمان"],
    "kuwait": ["kuwait", "الكويت"],
    "bahrain": ["bahrain", "البحرين"],
    "iraq": ["iraq", "العراق"],
    "iran": ["iran", "إيران", "ايران"],
    "jordan": ["jordan", "الأردن", "الاردن"],
    "lebanon": ["lebanon", "لبنان"],
    "israel": ["israel", "إسرائيل", "اسرائيل"],
    "syria": ["syria", "سوريا"],
    "yemen": ["yemen", "اليمن"],

    # Africa
    "egypt": ["egypt", "arab republic of egypt", "مصر", "جمهورية مصر العربية"],
    "libya": ["libya", "ليبيا"],
    "algeria": ["algeria", "الجزائر"],
    "tunisia": ["tunisia", "تونس"],
    "morocco": ["morocco", "المغرب"],
    "south africa": ["south africa", "جنوب افريقيا", "جنوب أفريقيا"],
    "nigeria": ["nigeria", "نيجيريا"],
    "angola": ["angola", "أنغولا", "انغولا"],
    "kenya": ["kenya", "كينيا"],
    "ghana": ["ghana", "غانا"],
    "ethiopia": ["ethiopia", "إثيوبيا", "اثيوبيا"],
    "mozambique": ["mozambique", "موزمبيق"],
    "tanzania": ["tanzania", "تنزانيا"],
    "zambia": ["zambia", "زامبيا"],
    "cameroon": ["cameroon", "الكاميرون"],
    "senegal": ["senegal", "السنغال"],
    "ivory coast": ["ivory coast", "côte d'ivoire", "cote d'ivoire", "ساحل العاج"],

    # Asia-Pacific
    "india": ["india", "bharat", "الهند"],
    "pakistan": ["pakistan", "باكستان"],
    "bangladesh": ["bangladesh", "بنغلاديش"],
    "sri lanka": ["sri lanka", "سريلانكا"],
    "china": ["china", "prc", "people's republic of china", "الصين"],
    "japan": ["japan", "اليابان"],
    "south korea": ["south korea", "korea", "republic of korea", "كوريا الجنوبية"],
    "north korea": ["north korea", "كوريا الشمالية"],
    "taiwan": ["taiwan", "تايوان"],
    "hong kong": ["hong kong", "hk", "هونغ كونغ", "هونج كونج"],
    "singapore": ["singapore", "سنغافورة"],
    "malaysia": ["malaysia", "ماليزيا"],
    "indonesia": ["indonesia", "إندونيسيا", "اندونيسيا"],
    "thailand": ["thailand", "تايلاند"],
    "vietnam": ["vietnam", "viet nam", "فيتنام"],
    "philippines": ["philippines", "الفلبين"],
    "myanmar": ["myanmar", "burma", "ميانمار"],
    "cambodia": ["cambodia", "كمبوديا"],
    "laos": ["laos", "لاوس"],
    "mongolia": ["mongolia", "منغوليا"],
    "australia": ["australia", "أستراليا", "استراليا"],
    "new zealand": ["new zealand", "nz", "نيوزيلندا"],
    "papua new guinea": ["papua new guinea", "بابوا غينيا الجديدة"],
    "fiji": ["fiji", "فيجي"],

    # Central Asia / CIS
    "kazakhstan": ["kazakhstan", "كازاخستان"],
    "azerbaijan": ["azerbaijan", "أذربيجان", "اذربيجان"],
    "uzbekistan": ["uzbekistan", "أوزبكستان", "اوزبكستان"],
    "turkmenistan": ["turkmenistan", "تركمانستان"],
    "kyrgyzstan": ["kyrgyzstan", "قرغيزستان"],
    "tajikistan": ["tajikistan", "طاجيكستان"],
    "georgia": ["georgia", "جورجيا"],
    "armenia": ["armenia", "أرمينيا", "ارمينيا"],
}

# ── All 50 US states + DC + territories → canonical "usa" ───────────────
US_STATES: Dict[str, str] = {
    "alabama": "usa", "alaska": "usa", "arizona": "usa", "arkansas": "usa",
    "california": "usa", "colorado": "usa", "connecticut": "usa",
    "delaware": "usa", "florida": "usa", "georgia state": "usa",
    "hawaii": "usa", "idaho": "usa", "illinois": "usa", "indiana": "usa",
    "iowa": "usa", "kansas": "usa", "kentucky": "usa", "louisiana": "usa",
    "maine": "usa", "maryland": "usa", "massachusetts": "usa",
    "michigan": "usa", "minnesota": "usa", "mississippi": "usa",
    "missouri": "usa", "montana": "usa", "nebraska": "usa", "nevada": "usa",
    "new hampshire": "usa", "new jersey": "usa", "new mexico": "usa",
    "new york state": "usa", "new york": "usa",
    "north carolina": "usa", "north dakota": "usa", "ohio": "usa",
    "oklahoma": "usa", "oregon": "usa", "pennsylvania": "usa",
    "rhode island": "usa", "south carolina": "usa", "south dakota": "usa",
    "tennessee": "usa", "texas": "usa", "utah": "usa", "vermont": "usa",
    "virginia": "usa", "washington state": "usa", "west virginia": "usa",
    "wisconsin": "usa", "wyoming": "usa",
    "district of columbia": "usa", "washington dc": "usa", "washington d.c.": "usa",
    "puerto rico": "usa", "guam": "usa", "us virgin islands": "usa",
    "american samoa": "usa", "northern mariana islands": "usa",
    # safe abbreviations only
    "al": "usa", "ak": "usa", "az": "usa", "ar": "usa",
    "ct": "usa", "fl": "usa", "ga": "usa", "il": "usa", "ia": "usa",
    "ks": "usa", "ky": "usa", "la": "usa", "md": "usa",
    "mi": "usa", "mn": "usa", "mo": "usa", "ne": "usa", "nv": "usa",
    "nh": "usa", "nj": "usa", "nm": "usa", "ny": "usa", "nc": "usa",
    "nd": "usa", "oh": "usa", "pa": "usa", "ri": "usa", "sc": "usa",
    "sd": "usa", "tn": "usa", "tx": "usa", "ut": "usa", "vt": "usa",
    "va": "usa", "wv": "usa", "wi": "usa", "wy": "usa", "dc": "usa",
    # Arabic names
    "تكساس": "usa", "كاليفورنيا": "usa", "نيويورك": "usa", "فلوريدا": "usa",
}

# ── Canadian provinces and territories → "canada" ───────────────────────
CANADIAN_PROVINCES: Dict[str, str] = {
    "ontario": "canada", "quebec": "canada", "british columbia": "canada",
    "alberta": "canada", "manitoba": "canada", "saskatchewan": "canada",
    "nova scotia": "canada", "new brunswick": "canada",
    "newfoundland and labrador": "canada", "prince edward island": "canada",
    "northwest territories": "canada", "nunavut": "canada", "yukon": "canada",
    "qc": "canada", "bc": "canada", "ab": "canada", "mb": "canada",
    "sk": "canada", "ns": "canada", "nb": "canada", "nl": "canada", "pe": "canada",
    "اونتاريو": "canada", "كيبيك": "canada", "البرتا": "canada",
}

# ── Australian states → "australia" ─────────────────────────────────────
AUSTRALIAN_STATES: Dict[str, str] = {
    "new south wales": "australia", "victoria": "australia", "queensland": "australia",
    "western australia": "australia", "south australia": "australia", "tasmania": "australia",
    "northern territory": "australia", "australian capital territory": "australia",
    "nsw": "australia", "vic": "australia", "qld": "australia", "wa": "australia",
    "sa": "australia", "tas": "australia", "nt": "australia", "act": "australia",
}

# ── Major cities worldwide → country ──────────────────────────────────────
CITY_TO_COUNTRY: Dict[str, str] = {
    # USA
    "houston": "usa", "dallas": "usa", "new york city": "usa", "los angeles": "usa",
    "chicago": "usa", "denver": "usa", "oklahoma city": "usa", "austin": "usa",
    "san francisco": "usa", "seattle": "usa", "boston": "usa", "miami": "usa",
    "atlanta": "usa", "phoenix": "usa", "san diego": "usa", "las vegas": "usa",
    "portland": "usa", "nashville": "usa", "minneapolis": "usa", "st. louis": "usa",
    "detroit": "usa", "pittsburgh": "usa", "charlotte": "usa", "indianapolis": "usa",
    "columbus": "usa", "san jose": "usa", "san antonio": "usa", "jacksonville": "usa",
    "fort worth": "usa", "memphis": "usa", "louisville": "usa", "baltimore": "usa",
    "washington": "usa", "midland": "usa", "odessa": "usa", "abilene": "usa",

    # Canada
    "calgary": "canada", "toronto": "canada", "vancouver": "canada", "montreal": "canada",
    "ottawa": "canada", "edmonton": "canada", "winnipeg": "canada", "quebec city": "canada",
    "halifax": "canada",

    # Egypt
    "cairo": "egypt", "alexandria": "egypt", "giza": "egypt", "port said": "egypt",
    "suez": "egypt", "luxor": "egypt", "aswan": "egypt", "hurghada": "egypt",
    "القاهرة": "egypt", "الاسكندرية": "egypt", "الإسكندرية": "egypt", "الجيزة": "egypt",
    "بورسعيد": "egypt", "السويس": "egypt", "الأقصر": "egypt", "اسوان": "egypt", "أسوان": "egypt",

    # Saudi Arabia
    "riyadh": "saudi arabia", "dhahran": "saudi arabia", "jeddah": "saudi arabia",
    "khobar": "saudi arabia", "dammam": "saudi arabia", "jubail": "saudi arabia",
    "al khobar": "saudi arabia", "الرياض": "saudi arabia", "الدمام": "saudi arabia",
    "جدة": "saudi arabia", "الخبر": "saudi arabia", "الجبيل": "saudi arabia",

    # UAE / Gulf
    "dubai": "united arab emirates", "abu dhabi": "united arab emirates", "sharjah": "united arab emirates",
    "ajman": "united arab emirates", "dubai city": "united arab emirates",
    "دبي": "united arab emirates", "ابوظبي": "united arab emirates", "أبوظبي": "united arab emirates",
    "الشارقة": "united arab emirates", "عجمان": "united arab emirates",
    "doha": "qatar", "muscat": "oman", "kuwait city": "kuwait", "manama": "bahrain",
    "salalah": "oman", "الدوحة": "qatar", "مسقط": "oman", "الكويت": "kuwait", "المنامة": "bahrain",

    # Europe
    "london": "united kingdom", "birmingham": "united kingdom", "manchester": "united kingdom",
    "glasgow": "united kingdom", "edinburgh": "united kingdom", "bristol": "united kingdom",
    "paris": "france", "marseille": "france", "lyon": "france",
    "berlin": "germany", "munich": "germany", "frankfurt": "germany", "hamburg": "germany",
    "cologne": "germany", "düsseldorf": "germany", "milan": "italy", "rome": "italy",
    "naples": "italy", "turin": "italy", "madrid": "spain", "barcelona": "spain",
    "seville": "spain", "amsterdam": "netherlands", "rotterdam": "netherlands",
    "the hague": "netherlands", "brussels": "belgium", "antwerp": "belgium",
    "oslo": "norway", "bergen": "norway", "stavanger": "norway", "stockholm": "sweden",
    "gothenburg": "sweden", "malmö": "sweden", "copenhagen": "denmark", "aarhus": "denmark",
    "helsinki": "finland", "tampere": "finland", "warsaw": "poland", "kraków": "poland",
    "vienna": "austria", "graz": "austria", "zurich": "switzerland", "geneva": "switzerland",
    "bern": "switzerland", "lisbon": "portugal", "porto": "portugal", "dublin": "ireland",
    "cork": "ireland", "athens": "greece", "thessaloniki": "greece", "istanbul": "turkey",
    "ankara": "turkey", "izmir": "turkey", "moscow": "russia", "st. petersburg": "russia",
    "saint petersburg": "russia", "kyiv": "ukraine", "kharkiv": "ukraine", "bucharest": "romania",
    "cluj": "romania", "budapest": "hungary", "prague": "czech republic", "bratislava": "slovakia",
    "zagreb": "croatia", "sofia": "bulgaria", "belgrade": "serbia", "reykjavik": "iceland",
    "لندن": "united kingdom", "باريس": "france", "برلين": "germany", "ميونخ": "germany",
    "روما": "italy", "مدريد": "spain", "امستردام": "netherlands", "أمستردام": "netherlands",
    "اوسلو": "norway", "أوسلو": "norway", "اسطنبول": "turkey", "إسطنبول": "turkey",

    # Asia
    "mumbai": "india", "delhi": "india", "new delhi": "india", "bangalore": "india",
    "bengaluru": "india", "hyderabad": "india", "chennai": "india", "kolkata": "india",
    "pune": "india", "karachi": "pakistan", "lahore": "pakistan", "islamabad": "pakistan",
    "dhaka": "bangladesh", "beijing": "china", "shanghai": "china", "shenzhen": "china",
    "guangzhou": "china", "chengdu": "china", "wuhan": "china", "hong kong": "hong kong",
    "tokyo": "japan", "osaka": "japan", "nagoya": "japan", "yokohama": "japan", "kyoto": "japan",
    "seoul": "south korea", "busan": "south korea", "incheon": "south korea", "singapore": "singapore",
    "kuala lumpur": "malaysia", "penang": "malaysia", "jakarta": "indonesia", "surabaya": "indonesia",
    "bali": "indonesia", "bangkok": "thailand", "chiang mai": "thailand", "ho chi minh city": "vietnam",
    "hanoi": "vietnam", "manila": "philippines", "cebu": "philippines", "colombo": "sri lanka",
    "taipei": "taiwan", "بكين": "china", "شنغهاي": "china", "طوكيو": "japan", "سيول": "south korea",

    # Australia / NZ
    "sydney": "australia", "melbourne": "australia", "brisbane": "australia", "perth": "australia",
    "adelaide": "australia", "canberra": "australia", "auckland": "new zealand", "wellington": "new zealand",

    # CIS / Central Asia
    "baku": "azerbaijan", "nur-sultan": "kazakhstan", "astana": "kazakhstan", "almaty": "kazakhstan",
    "tashkent": "uzbekistan", "ashgabat": "turkmenistan", "tbilisi": "georgia", "yerevan": "armenia",
    "باكو": "azerbaijan", "استانا": "kazakhstan", "أستانا": "kazakhstan",

    # Africa
    "lagos": "nigeria", "abuja": "nigeria", "nairobi": "kenya", "mombasa": "kenya",
    "luanda": "angola", "johannesburg": "south africa", "cape town": "south africa",
    "durban": "south africa", "accra": "ghana", "addis ababa": "ethiopia",
    "dar es salaam": "tanzania", "kampala": "uganda", "dakar": "senegal", "abidjan": "ivory coast",
    "tripoli": "libya", "algiers": "algeria", "tunis": "tunisia", "casablanca": "morocco", "rabat": "morocco",

    # Latin America
    "são paulo": "brazil", "rio de janeiro": "brazil", "brasília": "brazil", "buenos aires": "argentina",
    "córdoba": "argentina", "santiago": "chile", "bogotá": "colombia", "medellín": "colombia",
    "lima": "peru", "caracas": "venezuela", "quito": "ecuador", "mexico city": "mexico",
    "guadalajara": "mexico", "monterrey": "mexico",

    # Iraq / Iran / Levant
    "baghdad": "iraq", "basra": "iraq", "kirkuk": "iraq", "tehran": "iran", "isfahan": "iran",
    "amman": "jordan", "beirut": "lebanon", "بغداد": "iraq", "البصرة": "iraq", "طهران": "iran", "عمان": "jordan", "بيروت": "lebanon",
}

REGION_ALIASES: Dict[str, List[str]] = {
    "europe": [
        "united kingdom", "france", "germany", "italy", "spain", "portugal",
        "netherlands", "belgium", "switzerland", "austria", "norway",
        "sweden", "denmark", "finland", "poland", "czech republic",
        "romania", "greece", "turkey", "hungary", "ukraine", "ireland",
        "serbia", "croatia", "bulgaria", "slovakia", "slovenia",
        "estonia", "latvia", "lithuania", "luxembourg", "iceland",
    ],
    "أوروبا": [
        "united kingdom", "france", "germany", "italy", "spain", "portugal",
        "netherlands", "belgium", "switzerland", "austria", "norway",
        "sweden", "denmark", "finland", "poland", "czech republic",
        "romania", "greece", "turkey", "hungary", "ukraine", "ireland",
        "serbia", "croatia", "bulgaria", "slovakia", "slovenia",
        "estonia", "latvia", "lithuania", "luxembourg", "iceland",
    ],
    "middle east": [
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen", "syria",
    ],
    "الشرق الأوسط": [
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen", "syria",
    ],
    "الشرق الاوسط": [
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen", "syria",
    ],
    "north africa": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "شمال أفريقيا": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "شمال افريقيا": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "mena": [
        "egypt", "libya", "algeria", "tunisia", "morocco",
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen",
    ],
    "gcc": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
    "الخليج": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
    "دول الخليج": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
    "مجلس التعاون الخليجي": ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"],
    "asia": [
        "india", "pakistan", "bangladesh", "china", "japan", "south korea",
        "singapore", "malaysia", "indonesia", "thailand", "vietnam", "philippines",
        "sri lanka", "taiwan", "hong kong", "myanmar", "cambodia",
    ],
    "africa": [
        "egypt", "libya", "algeria", "tunisia", "morocco",
        "south africa", "nigeria", "angola", "kenya", "ghana",
        "ethiopia", "mozambique", "tanzania", "cameroon", "senegal",
    ],
    "أفريقيا": [
        "egypt", "libya", "algeria", "tunisia", "morocco",
        "south africa", "nigeria", "angola", "kenya", "ghana",
        "ethiopia", "mozambique", "tanzania", "cameroon", "senegal",
    ],
    "north america": ["usa", "canada", "mexico"],
    "south america": ["brazil", "argentina", "chile", "colombia", "peru", "venezuela", "ecuador"],
    "latin america": [
        "mexico", "brazil", "argentina", "chile", "colombia", "peru",
        "venezuela", "ecuador", "bolivia", "uruguay", "paraguay",
    ],
    "cis": ["russia", "kazakhstan", "azerbaijan", "ukraine", "uzbekistan", "turkmenistan", "kyrgyzstan", "tajikistan", "georgia", "armenia"],
    "apac": [
        "australia", "new zealand", "india", "china", "japan", "south korea",
        "singapore", "malaysia", "indonesia", "thailand", "vietnam", "philippines",
    ],
    "nordics": ["norway", "sweden", "denmark", "finland", "iceland"],
    "scandinavia": ["norway", "sweden", "denmark"],
    "dach": ["germany", "austria", "switzerland"],
    "benelux": ["netherlands", "belgium", "luxembourg"],
    "balkans": ["serbia", "croatia", "bulgaria", "romania", "greece", "slovenia"],
}

# Combined sub-national lookup: state/province/territory → country
_SUBNATIONAL: Dict[str, str] = {}
_SUBNATIONAL.update(US_STATES)
_SUBNATIONAL.update(CANADIAN_PROVINCES)
_SUBNATIONAL.update(AUSTRALIAN_STATES)

# Human-friendly display names for canonical country codes used internally
COUNTRY_DISPLAY_NAMES: Dict[str, str] = {
    "usa": "United States",
    **{name: name.title() for name in COUNTRY_ALIASES.keys() if name != "usa"},
}
COUNTRY_DISPLAY_NAMES.update({
    "united kingdom": "United Kingdom",
    "united arab emirates": "United Arab Emirates",
    "czech republic": "Czech Republic",
    "saudi arabia": "Saudi Arabia",
    "south africa": "South Africa",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "new zealand": "New Zealand",
    "sri lanka": "Sri Lanka",
    "papua new guinea": "Papua New Guinea",
    "ivory coast": "Ivory Coast",
    "hong kong": "Hong Kong",
})


def normalize_geo_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("،", " ").replace("؛", " ").replace("/", " ")
    text = _normalize_arabic(text)
    return re.sub(r"\s+", " ", text).strip()


def _geo_pattern(token: str) -> str:
    token = normalize_geo_text(token)
    escaped = re.escape(token)
    if re.fullmatch(r"[a-z]{2}", token):
        return r"(?<![a-z])" + escaped + r"(?![a-z])"
    return r"(?<![\w])" + escaped + r"(?![\w])"


def normalize_country_name(name: str) -> str:
    """Resolve country name, alias, state, province, city → canonical country name."""
    text = normalize_geo_text(name)
    if not text:
        return ""

    # Direct country match
    for canonical, aliases in COUNTRY_ALIASES.items():
        if text == normalize_geo_text(canonical):
            return canonical
        for alias in aliases:
            if text == normalize_geo_text(alias):
                return canonical

    if text in _SUBNATIONAL:
        return _SUBNATIONAL[text]

    if text in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[text]

    return text  # pass-through if unknown


def humanize_country_name(name: str) -> str:
    canonical = normalize_country_name(name)
    return COUNTRY_DISPLAY_NAMES.get(canonical, canonical.replace("_", " ").title())


def expand_region_name(name: str) -> List[str]:
    """Expand a region name to its constituent countries."""
    text = normalize_geo_text(name)
    for region, countries in REGION_ALIASES.items():
        if normalize_geo_text(region) == text:
            return list(countries)
    return []


def all_country_names() -> List[str]:
    """All canonical country names."""
    return sorted(COUNTRY_ALIASES.keys())


def all_region_names() -> List[str]:
    return sorted({normalize_geo_text(k) for k in REGION_ALIASES.keys()})


def _collect_region_matches(text: str) -> Set[str]:
    low = normalize_geo_text(text)
    found: Set[str] = set()
    for region, countries in REGION_ALIASES.items():
        if re.search(_geo_pattern(region), low):
            found.update(countries)
    return found


def _collect_country_matches(text: str) -> Set[str]:
    low = normalize_geo_text(text)
    found: Set[str] = set()

    for canonical, aliases in COUNTRY_ALIASES.items():
        for alias in [canonical] + aliases:
            if re.search(_geo_pattern(alias), low):
                found.add(canonical)
                break

    for subnational, country in _SUBNATIONAL.items():
        if re.search(_geo_pattern(subnational), low):
            found.add(country)

    for city, country in CITY_TO_COUNTRY.items():
        if re.search(_geo_pattern(city), low):
            found.add(country)

    return found


def find_countries_in_text(text: str) -> List[str]:
    """
    Find all country references in text, including:
    - Country names and aliases
    - Region names / abbreviations (GCC, MENA, Europe → expanded to countries)
    - US states, Canadian provinces, Australian states
    - Major world cities
    Returns list of canonical country names.
    """
    found: Set[str] = set()
    found.update(_collect_region_matches(text))
    found.update(_collect_country_matches(text))
    return sorted(found)


def find_regions_in_text(text: str) -> List[str]:
    low = normalize_geo_text(text)
    found: Set[str] = set()
    for region in REGION_ALIASES.keys():
        if re.search(_geo_pattern(region), low):
            found.add(normalize_geo_text(region))
    return sorted(found)


def find_cities_in_text(text: str) -> List[str]:
    low = normalize_geo_text(text)
    found: List[str] = []
    seen = set()
    for city in sorted(CITY_TO_COUNTRY.keys(), key=lambda x: len(normalize_geo_text(x)), reverse=True):
        if city in seen:
            continue
        if re.search(_geo_pattern(city), low):
            norm_city = normalize_geo_text(city)
            if norm_city not in seen:
                seen.add(norm_city)
                found.append(city)
    return found


def find_first_country_in_text(text: str) -> str:
    matches = find_countries_in_text(text)
    return matches[0] if matches else ""


def contains_country_or_city(text: str, country_name: str) -> bool:
    """Return True if the text contains a reference to the given country (directly or via city/state)."""
    canonical = normalize_country_name(country_name)
    return canonical in find_countries_in_text(text)


def explain_geo_matches(text: str) -> Dict[str, List[str]]:
    """Helpful for debugging parser / verifier behavior."""
    return {
        "countries": find_countries_in_text(text),
        "regions": find_regions_in_text(text),
        "cities": find_cities_in_text(text),
    }
