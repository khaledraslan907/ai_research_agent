"""
geography.py
============
Complete geographic resolution for the AI Research Agent.

Covers:
- All countries with common aliases and abbreviations
- All 50 US states + DC + territories (maps to "usa")
- All Canadian provinces and territories (maps to "canada")
- All Australian states (maps to "australia")
- Major cities worldwide → country
- Region expansions (Europe, MENA, etc.)

This makes the agent universally useful — not just for oil & gas —
and correctly handles prompts like:
  "companies in Texas" → include: usa
  "companies in California" → include: usa
  "companies in Ontario" → include: canada
  "companies outside New York State" → exclude: usa
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


# ── Country aliases ────────────────────────────────────────────────────────
COUNTRY_ALIASES: Dict[str, List[str]] = {
    # Americas
    "usa": [
        "usa", "us", "u.s.", "u.s.a.", "united states", "united states of america",
        "america", "the us", "the usa", "the united states", "أمريكا", "الولايات المتحدة",
    ],
    "canada": ["canada", "ca"],
    "mexico": ["mexico", "méxico"],
    "brazil": ["brazil", "brasil"],
    "argentina": ["argentina"],
    "chile": ["chile"],
    "colombia": ["colombia"],
    "peru": ["peru", "perú"],
    "venezuela": ["venezuela"],
    "ecuador": ["ecuador"],
    "bolivia": ["bolivia"],
    "uruguay": ["uruguay"],
    "paraguay": ["paraguay"],
    # Europe
    "united kingdom": [
        "uk", "u.k.", "united kingdom", "britain", "great britain",
        "england", "scotland", "wales", "northern ireland", "بريطانيا", "المملكة المتحدة",
    ],
    "ireland": ["ireland", "eire"],
    "france": ["france"],
    "germany": ["germany", "deutschland"],
    "italy": ["italy", "italia"],
    "spain": ["spain", "españa"],
    "portugal": ["portugal"],
    "netherlands": ["netherlands", "holland", "the netherlands"],
    "belgium": ["belgium"],
    "switzerland": ["switzerland"],
    "austria": ["austria"],
    "norway": ["norway"],
    "sweden": ["sweden"],
    "denmark": ["denmark"],
    "finland": ["finland"],
    "poland": ["poland"],
    "czech republic": ["czech republic", "czechia", "czech"],
    "romania": ["romania"],
    "greece": ["greece"],
    "turkey": ["turkey", "türkiye"],
    "hungary": ["hungary"],
    "ukraine": ["ukraine"],
    "russia": ["russia", "russian federation"],
    "serbia": ["serbia"],
    "croatia": ["croatia"],
    "bulgaria": ["bulgaria"],
    "slovakia": ["slovakia"],
    "slovenia": ["slovenia"],
    "estonia": ["estonia"],
    "latvia": ["latvia"],
    "lithuania": ["lithuania"],
    "luxembourg": ["luxembourg"],
    "iceland": ["iceland"],
    # Middle East
    "saudi arabia": ["saudi arabia", "ksa", "kingdom of saudi arabia", "السعودية", "المملكة العربية السعودية"],
    "united arab emirates": [
        "uae", "u.a.e", "u.a.e.", "united arab emirates", "emirates",
        "the uae", "الإمارات", "الامارات", "الإمارات العربية المتحدة",
    ],
    "qatar": ["qatar", "قطر"],
    "oman": ["oman", "sultanate of oman", "عمان"],
    "kuwait": ["kuwait", "الكويت"],
    "bahrain": ["bahrain", "البحرين"],
    "iraq": ["iraq"],
    "iran": ["iran"],
    "jordan": ["jordan"],
    "lebanon": ["lebanon"],
    "israel": ["israel"],
    "syria": ["syria"],
    "yemen": ["yemen"],
    # Africa
    "egypt": ["egypt", "arab republic of egypt", "مصر", "جمهورية مصر العربية"],
    "libya": ["libya"],
    "algeria": ["algeria"],
    "tunisia": ["tunisia"],
    "morocco": ["morocco"],
    "south africa": ["south africa"],
    "nigeria": ["nigeria"],
    "angola": ["angola"],
    "kenya": ["kenya"],
    "ghana": ["ghana"],
    "ethiopia": ["ethiopia"],
    "mozambique": ["mozambique"],
    "tanzania": ["tanzania"],
    "zambia": ["zambia"],
    "cameroon": ["cameroon"],
    "senegal": ["senegal"],
    "ivory coast": ["ivory coast", "côte d'ivoire", "cote d'ivoire"],
    # Asia-Pacific
    "india": ["india", "bharat"],
    "pakistan": ["pakistan"],
    "bangladesh": ["bangladesh"],
    "sri lanka": ["sri lanka"],
    "china": ["china", "prc", "people's republic of china"],
    "japan": ["japan"],
    "south korea": ["south korea", "korea", "republic of korea"],
    "north korea": ["north korea"],
    "taiwan": ["taiwan"],
    "hong kong": ["hong kong", "hk"],
    "singapore": ["singapore"],
    "malaysia": ["malaysia"],
    "indonesia": ["indonesia"],
    "thailand": ["thailand"],
    "vietnam": ["vietnam", "viet nam"],
    "philippines": ["philippines"],
    "myanmar": ["myanmar", "burma"],
    "cambodia": ["cambodia"],
    "laos": ["laos"],
    "mongolia": ["mongolia"],
    "australia": ["australia"],
    "new zealand": ["new zealand", "nz"],
    "papua new guinea": ["papua new guinea"],
    "fiji": ["fiji"],
    # Central Asia / CIS
    "kazakhstan": ["kazakhstan"],
    "azerbaijan": ["azerbaijan"],
    "uzbekistan": ["uzbekistan"],
    "turkmenistan": ["turkmenistan"],
    "kyrgyzstan": ["kyrgyzstan"],
    "tajikistan": ["tajikistan"],
    "georgia": ["georgia"],
    "armenia": ["armenia"],
}

# ── All 50 US states + DC + territories → canonical "usa" ─────────────────
US_STATES: Dict[str, str] = {
    # Full names
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
    # Territories
    "puerto rico": "usa", "guam": "usa", "us virgin islands": "usa",
    "american samoa": "usa", "northern mariana islands": "usa",
    # 2-letter abbreviations — ONLY safe ones (not common English words)
    # Excluded: "in", "or", "ok", "ma", "hi", "me", "de", "id", "co", "ms", "mt"
    # as these are common English words causing false positives
    "al": "usa", "ak": "usa", "az": "usa", "ar": "usa",
    "ct": "usa", "fl": "usa", "ga": "usa",
    "il": "usa", "ia": "usa",
    "ks": "usa", "ky": "usa", "la": "usa", "md": "usa",
    "mi": "usa", "mn": "usa", "mo": "usa",
    "ne": "usa", "nv": "usa", "nh": "usa", "nj": "usa",
    "nm": "usa", "ny": "usa", "nc": "usa", "nd": "usa", "oh": "usa",
    "pa": "usa", "ri": "usa", "sc": "usa",
    "sd": "usa", "tn": "usa", "tx": "usa", "ut": "usa", "vt": "usa",
    "va": "usa", "wv": "usa", "wi": "usa", "wy": "usa",
    "dc": "usa",
}

# ── Canadian provinces and territories → "canada" ─────────────────────────
CANADIAN_PROVINCES: Dict[str, str] = {
    "ontario": "canada", "quebec": "canada", "british columbia": "canada",
    "alberta": "canada", "manitoba": "canada", "saskatchewan": "canada",
    "nova scotia": "canada", "new brunswick": "canada",
    "newfoundland and labrador": "canada", "prince edward island": "canada",
    "northwest territories": "canada", "nunavut": "canada",
    "yukon": "canada",
    # Abbreviations — only unambiguous ones
    "qc": "canada", "bc": "canada", "ab": "canada",
    "mb": "canada", "sk": "canada", "ns": "canada", "nb": "canada",
    "nl": "canada", "pe": "canada",
}

# ── Australian states → "australia" ───────────────────────────────────────
AUSTRALIAN_STATES: Dict[str, str] = {
    "new south wales": "australia", "victoria": "australia",
    "queensland": "australia", "western australia": "australia",
    "south australia": "australia", "tasmania": "australia",
    "northern territory": "australia",
    "australian capital territory": "australia",
    "nsw": "australia", "vic": "australia", "qld": "australia",
    "wa": "australia", "sa": "australia", "tas": "australia",
    "nt": "australia", "act": "australia",
}

# ── Major cities worldwide → country ─────────────────────────────────────
CITY_TO_COUNTRY: Dict[str, str] = {
    # USA major cities
    "houston": "usa", "dallas": "usa", "new york city": "usa",
    "los angeles": "usa", "chicago": "usa", "denver": "usa",
    "oklahoma city": "usa", "austin": "usa", "san francisco": "usa",
    "seattle": "usa", "boston": "usa", "miami": "usa", "atlanta": "usa",
    "phoenix": "usa", "san diego": "usa", "las vegas": "usa",
    "portland": "usa", "nashville": "usa", "minneapolis": "usa",
    "st. louis": "usa", "detroit": "usa", "pittsburgh": "usa",
    "charlotte": "usa", "indianapolis": "usa", "columbus": "usa",
    "san jose": "usa", "san antonio": "usa", "jacksonville": "usa",
    "fort worth": "usa", "memphis": "usa", "louisville": "usa",
    "baltimore": "usa", "washington": "usa",
    "midland": "usa", "odessa": "usa", "abilene": "usa",
    # Canada
    "calgary": "canada", "toronto": "canada", "vancouver": "canada",
    "montreal": "canada", "ottawa": "canada", "edmonton": "canada",
    "winnipeg": "canada", "quebec city": "canada", "halifax": "canada",
    # Egypt
    "cairo": "egypt", "alexandria": "egypt", "giza": "egypt",
    "port said": "egypt", "suez": "egypt", "luxor": "egypt",
    "aswan": "egypt", "hurghada": "egypt",
    # Saudi Arabia
    "riyadh": "saudi arabia", "dhahran": "saudi arabia", "jeddah": "saudi arabia",
    "khobar": "saudi arabia", "dammam": "saudi arabia", "jubail": "saudi arabia",
    "al khobar": "saudi arabia",
    # UAE
    "dubai": "united arab emirates", "abu dhabi": "united arab emirates",
    "sharjah": "united arab emirates", "ajman": "united arab emirates",
    # Gulf
    "doha": "qatar", "muscat": "oman", "kuwait city": "kuwait",
    "manama": "bahrain", "salalah": "oman",
    # Europe
    "london": "united kingdom", "birmingham": "united kingdom",
    "manchester": "united kingdom", "glasgow": "united kingdom",
    "edinburgh": "united kingdom", "bristol": "united kingdom",
    "paris": "france", "marseille": "france", "lyon": "france",
    "berlin": "germany", "munich": "germany", "frankfurt": "germany",
    "hamburg": "germany", "cologne": "germany", "düsseldorf": "germany",
    "milan": "italy", "rome": "italy", "naples": "italy", "turin": "italy",
    "madrid": "spain", "barcelona": "spain", "seville": "spain",
    "amsterdam": "netherlands", "rotterdam": "netherlands", "the hague": "netherlands",
    "brussels": "belgium", "antwerp": "belgium",
    "oslo": "norway", "bergen": "norway", "stavanger": "norway",
    "stockholm": "sweden", "gothenburg": "sweden", "malmö": "sweden",
    "copenhagen": "denmark", "aarhus": "denmark",
    "helsinki": "finland", "tampere": "finland",
    "warsaw": "poland", "kraków": "poland",
    "vienna": "austria", "graz": "austria",
    "zurich": "switzerland", "geneva": "switzerland", "bern": "switzerland",
    "lisbon": "portugal", "porto": "portugal",
    "dublin": "ireland", "cork": "ireland",
    "athens": "greece", "thessaloniki": "greece",
    "istanbul": "turkey", "ankara": "turkey", "izmir": "turkey",
    "moscow": "russia", "st. petersburg": "russia", "saint petersburg": "russia",
    "kyiv": "ukraine", "kharkiv": "ukraine",
    "bucharest": "romania", "cluj": "romania",
    "budapest": "hungary", "prague": "czech republic",
    "bratislava": "slovakia", "zagreb": "croatia",
    "sofia": "bulgaria", "belgrade": "serbia",
    "reykjavik": "iceland",
    # Asia
    "mumbai": "india", "delhi": "india", "new delhi": "india",
    "bangalore": "india", "bengaluru": "india", "hyderabad": "india",
    "chennai": "india", "kolkata": "india", "pune": "india",
    "karachi": "pakistan", "lahore": "pakistan", "islamabad": "pakistan",
    "dhaka": "bangladesh",
    "beijing": "china", "shanghai": "china", "shenzhen": "china",
    "guangzhou": "china", "chengdu": "china", "wuhan": "china",
    "hong kong": "hong kong",
    "tokyo": "japan", "osaka": "japan", "nagoya": "japan",
    "yokohama": "japan", "kyoto": "japan",
    "seoul": "south korea", "busan": "south korea", "incheon": "south korea",
    "singapore": "singapore",
    "kuala lumpur": "malaysia", "penang": "malaysia",
    "jakarta": "indonesia", "surabaya": "indonesia", "bali": "indonesia",
    "bangkok": "thailand", "chiang mai": "thailand",
    "ho chi minh city": "vietnam", "hanoi": "vietnam",
    "manila": "philippines", "cebu": "philippines",
    "colombo": "sri lanka",
    "taipei": "taiwan",
    # Australia / NZ
    "sydney": "australia", "melbourne": "australia", "brisbane": "australia",
    "perth": "australia", "adelaide": "australia", "canberra": "australia",
    "auckland": "new zealand", "wellington": "new zealand",
    # CIS / Central Asia
    "baku": "azerbaijan", "nur-sultan": "kazakhstan",
    "astana": "kazakhstan", "almaty": "kazakhstan",
    "tashkent": "uzbekistan", "ashgabat": "turkmenistan",
    "tbilisi": "georgia", "yerevan": "armenia",
    # Africa
    "lagos": "nigeria", "abuja": "nigeria",
    "nairobi": "kenya", "mombasa": "kenya",
    "luanda": "angola", "johannesburg": "south africa",
    "cape town": "south africa", "durban": "south africa",
    "accra": "ghana", "addis ababa": "ethiopia",
    "dar es salaam": "tanzania", "kampala": "uganda",
    "dakar": "senegal", "abidjan": "ivory coast",
    "tripoli": "libya", "algiers": "algeria", "tunis": "tunisia",
    "casablanca": "morocco", "rabat": "morocco",
    # Latin America
    "são paulo": "brazil", "rio de janeiro": "brazil", "brasília": "brazil",
    "buenos aires": "argentina", "córdoba": "argentina",
    "santiago": "chile", "bogotá": "colombia", "medellín": "colombia",
    "lima": "peru", "caracas": "venezuela", "quito": "ecuador",
    "mexico city": "mexico", "guadalajara": "mexico", "monterrey": "mexico",
    # Iraq / Iran
    "baghdad": "iraq", "basra": "iraq", "kirkuk": "iraq",
    "tehran": "iran", "isfahan": "iran",
    # Jordan / Lebanon
    "amman": "jordan", "beirut": "lebanon",
}



ARABIC_REGION_ALIASES: Dict[str, str] = {
    "الخليج": "gcc",
    "دول الخليج": "gcc",
    "اوروبا": "europe",
    "أوروبا": "europe",
    "الشرق الاوسط": "middle east",
    "الشرق الأوسط": "middle east",
    "شمال افريقيا": "north africa",
    "شمال أفريقيا": "north africa",
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
    "middle east": [
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen", "syria",
    ],
    "north africa": ["egypt", "libya", "algeria", "tunisia", "morocco"],
    "mena": [
        "egypt", "libya", "algeria", "tunisia", "morocco",
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait",
        "bahrain", "iraq", "jordan", "lebanon", "israel", "iran", "yemen",
    ],
    "gcc": [
        "saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain",
    ],
    "asia": [
        "india", "pakistan", "bangladesh", "china", "japan", "south korea",
        "singapore", "malaysia", "indonesia", "thailand", "vietnam", "philippines",
        "sri lanka", "taiwan", "hong kong", "myanmar", "cambodia",
    ],
    "southeast asia": [
        "singapore", "malaysia", "indonesia", "thailand", "vietnam",
        "philippines", "myanmar", "cambodia", "laos",
    ],
    "africa": [
        "egypt", "libya", "algeria", "tunisia", "morocco",
        "south africa", "nigeria", "angola", "kenya", "ghana",
        "ethiopia", "mozambique", "tanzania", "cameroon", "senegal",
    ],
    "sub-saharan africa": [
        "south africa", "nigeria", "angola", "kenya", "ghana",
        "ethiopia", "mozambique", "tanzania",
    ],
    "north america": ["usa", "canada", "mexico"],
    "south america": ["brazil", "argentina", "chile", "colombia", "peru", "venezuela", "ecuador"],
    "latin america": [
        "mexico", "brazil", "argentina", "chile", "colombia", "peru",
        "venezuela", "ecuador", "bolivia", "uruguay", "paraguay",
    ],
    "cis": ["russia", "kazakhstan", "azerbaijan", "ukraine", "uzbekistan",
            "turkmenistan", "kyrgyzstan", "tajikistan", "georgia", "armenia"],
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



# Hotfix aliases for Arabic geography resolution
for _canonical, _aliases in {
    "egypt": ["مصر", "جمهوريه مصر العربيه", "جمهورية مصر العربية"],
    "saudi arabia": ["السعوديه", "السعودية", "المملكه العربيه السعوديه", "المملكة العربية السعودية"],
    "united arab emirates": ["الامارات", "الإمارات", "دوله الامارات العربيه المتحده", "دولة الإمارات العربية المتحدة"],
    "qatar": ["قطر"],
    "oman": ["عمان", "سلطنه عمان", "سلطنة عمان"],
    "kuwait": ["الكويت"],
    "bahrain": ["البحرين"],
    "germany": ["المانيا", "ألمانيا"],
    "norway": ["النرويج"],
    "united kingdom": ["بريطانيا", "المملكه المتحده", "المملكة المتحدة", "انجلترا", "إنجلترا"],
    "united states": ["امريكا", "أمريكا", "الولايات المتحده", "الولايات المتحدة", "الولايات المتحدة الامريكية", "الولايات المتحدة الأمريكية"],
}.items():
    bucket = COUNTRY_ALIASES.setdefault(_canonical, [])
    for _alias in _aliases:
        if _alias not in bucket:
            bucket.append(_alias)

for _region, _aliases in {
    "gcc": ["الخليج", "دول الخليج", "مجلس التعاون الخليجي"],
    "mena": ["الشرق الاوسط وشمال افريقيا", "الشرق الأوسط وشمال أفريقيا"],
    "middle east": ["الشرق الاوسط", "الشرق الأوسط"],
    "north africa": ["شمال افريقيا", "شمال أفريقيا"],
}.items():
    bucket = REGION_ALIASES.setdefault(_region, [])
    for _alias in _aliases:
        if _alias not in bucket:
            bucket.append(_alias)

CITY_TO_COUNTRY.update({
    "القاهره": "egypt",
    "القاهرة": "egypt",
    "الاسكندريه": "egypt",
    "الاسكندرية": "egypt",
    "الإسكندرية": "egypt",
    "السويس": "egypt",
    "دبي": "united arab emirates",
    "ابو ظبي": "united arab emirates",
    "أبو ظبي": "united arab emirates",
    "الرياض": "saudi arabia",
    "الدمام": "saudi arabia",
    "الدوحه": "qatar",
    "الدوحة": "qatar",
    "مسقط": "oman",
    "مدينه الكويت": "kuwait",
    "مدينة الكويت": "kuwait",
    "المنامه": "bahrain",
    "المنامة": "bahrain",
})


# ── Arabic aliases / region variants ───────────────────────────────────────
COUNTRY_ALIASES.setdefault("egypt", []).extend(["مصر", "جمهوريه مصر العربيه"])
COUNTRY_ALIASES.setdefault("saudi arabia", []).extend(["السعوديه", "المملكه العربيه السعوديه"])
COUNTRY_ALIASES.setdefault("united arab emirates", []).extend(["الامارات", "الإمارات", "الامارات العربيه المتحده"])
COUNTRY_ALIASES.setdefault("qatar", []).extend(["قطر"])
COUNTRY_ALIASES.setdefault("oman", []).extend(["عمان", "سلطنه عمان"])
COUNTRY_ALIASES.setdefault("kuwait", []).extend(["الكويت"])
COUNTRY_ALIASES.setdefault("bahrain", []).extend(["البحرين"])
COUNTRY_ALIASES.setdefault("germany", []).extend(["المانيا", "ألمانيا"])
COUNTRY_ALIASES.setdefault("norway", []).extend(["النرويج"])
COUNTRY_ALIASES.setdefault("united kingdom", []).extend(["بريطانيا", "المملكه المتحده", "انجلترا"])
COUNTRY_ALIASES.setdefault("usa", []).extend(["امريكا", "أمريكا", "الولايات المتحده", "الولايات المتحده الامريكيه"])

REGION_ALIASES.setdefault("gcc", ["saudi arabia", "united arab emirates", "qatar", "oman", "kuwait", "bahrain"])
REGION_ALIASES["gcc"] = list(dict.fromkeys(REGION_ALIASES["gcc"] + ["الخليج", "دول الخليج", "مجلس التعاون الخليجي"]))
REGION_ALIASES.setdefault("middle east", [])
REGION_ALIASES["middle east"] = list(dict.fromkeys(REGION_ALIASES["middle east"] + ["الشرق الاوسط", "الشرق الأوسط"]))
REGION_ALIASES.setdefault("north africa", [])
REGION_ALIASES["north africa"] = list(dict.fromkeys(REGION_ALIASES["north africa"] + ["شمال افريقيا", "شمال أفريقيا"]))
REGION_ALIASES.setdefault("mena", [])
REGION_ALIASES["mena"] = list(dict.fromkeys(REGION_ALIASES["mena"] + ["الشرق الاوسط وشمال افريقيا", "الشرق الأوسط وشمال أفريقيا"]))

CITY_TO_COUNTRY.update({
    "القاهره": "egypt",
    "الاسكندريه": "egypt",
    "الاسكندرية": "egypt",
    "السويس": "egypt",
    "دبي": "united arab emirates",
    "ابو ظبي": "united arab emirates",
    "الرياض": "saudi arabia",
    "الدمام": "saudi arabia",
    "الدوحه": "qatar",
    "مسقط": "oman",
    "مدينه الكويت": "kuwait",
    "المنامه": "bahrain",
})


# Combined sub-national lookup: state/province/territory → country
_SUBNATIONAL: Dict[str, str] = {}
_SUBNATIONAL.update(US_STATES)
_SUBNATIONAL.update(CANADIAN_PROVINCES)
_SUBNATIONAL.update(AUSTRALIAN_STATES)


def normalize_geo_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return ARABIC_REGION_ALIASES.get(text, text)


def normalize_country_name(name: str) -> str:
    """Resolve country name, alias, state, province, city → canonical country name."""
    text = normalize_geo_text(name)
    if not text:
        return ""

    # Direct country match
    for canonical, aliases in COUNTRY_ALIASES.items():
        if text == canonical:
            return canonical
        for alias in aliases:
            if text == normalize_geo_text(alias):
                return canonical

    # Sub-national (state/province/territory)
    if text in _SUBNATIONAL:
        return _SUBNATIONAL[text]

    # City
    if text in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[text]

    return text  # pass-through if unknown


def expand_region_name(name: str) -> List[str]:
    """Expand a region name to its constituent countries."""
    text = normalize_geo_text(name)
    return REGION_ALIASES.get(text, [])


def all_country_names() -> List[str]:
    """All canonical country names."""
    return sorted(COUNTRY_ALIASES.keys())


def find_countries_in_text(text: str) -> List[str]:
    """
    Find all country references in text, including:
    - Country names and aliases
    - Region names / abbreviations (GCC, MENA, Europe → expanded to countries)
    - US states, Canadian provinces, Australian states
    - Major world cities
    Returns list of canonical country names.
    """
    low = normalize_geo_text(text)
    found: Set[str] = set()

    # Region aliases first (e.g. "gcc", "mena", "europe")
    for region, countries in REGION_ALIASES.items():
        if re.search(r"\b" + re.escape(region) + r"\b", low):
            found.update(countries)

    # Country names and aliases
    for canonical, aliases in COUNTRY_ALIASES.items():
        for alias in [canonical] + aliases:
            if re.search(r"\b" + re.escape(normalize_geo_text(alias)) + r"\b", low):
                found.add(canonical)
                break

    # Sub-national regions (states, provinces) — 2-letter only as word boundary
    for subnational, country in _SUBNATIONAL.items():
        if len(subnational) <= 2:
            # Strict: must be surrounded by non-alpha OR be standalone
            if re.search(r"(?<![a-zA-Z])" + re.escape(subnational) + r"(?![a-zA-Z])", low):
                found.add(country)
        else:
            if re.search(r"\b" + re.escape(subnational) + r"\b", low):
                found.add(country)

    # Cities
    for city, country in CITY_TO_COUNTRY.items():
        if re.search(r"\b" + re.escape(normalize_geo_text(city)) + r"\b", low):
            found.add(country)

    return sorted(found)


def find_first_country_in_text(text: str) -> str:
    matches = find_countries_in_text(text)
    return matches[0] if matches else ""


def contains_country_or_city(text: str, country_name: str) -> bool:
    """Return True if the text contains a reference to the given country (directly or via city/state)."""
    low = normalize_geo_text(text)
    canonical = normalize_country_name(country_name)
    if canonical in find_countries_in_text(low):
        return True
    return False
