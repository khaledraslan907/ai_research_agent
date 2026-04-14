"""
keyword_expander.py
===================
Proposes related keywords to widen the user's search.

Given a prompt like "Find oil gas service companies in Egypt",
produces expansion suggestions such as:
  - Industry synonyms: "petroleum", "upstream", "oilfield services"
  - Company type variants: "EPC contractor", "drilling services", "well services"
  - Location variants: "Cairo", "Alexandria", "Egyptian"
  - Attribute variants: "email", "phone", "contact"

The user can then select which expansions to include before running.
Uses LLM when available; falls back to a comprehensive rule-based dictionary.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Rule-based expansion dictionaries
# ---------------------------------------------------------------------------

# Maps topic keywords to domain-specific synonyms and related terms
TOPIC_EXPANSIONS: Dict[str, List[str]] = {
    # Oil & Gas
    "oil gas":          ["petroleum", "hydrocarbons", "upstream", "downstream", "oilfield"],
    "oil and gas":      ["petroleum", "hydrocarbons", "upstream", "downstream", "oilfield"],
    "oilfield":         ["oilfield services", "well services", "drilling services", "completion"],
    "oilfield service": ["well services", "drilling contractor", "completion services",
                         "wireline", "cementing", "stimulation", "coiled tubing",
                         "mud logging", "directional drilling", "ESP services"],
    "upstream":         ["exploration", "production", "E&P", "drilling", "reservoir",
                         "well completion", "artificial lift"],
    "downstream":       ["refinery", "petrochemical", "LNG", "pipeline", "storage"],
    "drilling":         ["rotary drilling", "directional drilling", "horizontal drilling",
                         "drill bit", "drill string", "BHA", "MWD", "LWD"],
    "petroleum engineer":   ["reservoir engineer", "drilling engineer", "production engineer",
                              "completion engineer", "wellsite engineer"],
    "digital oil gas":  ["digital oilfield", "IIoT energy", "Industry 4.0 oil gas",
                         "digital transformation energy", "upstream software",
                         "production optimization software", "SCADA oil gas"],
    "esp":              ["electrical submersible pump", "artificial lift", "downhole pump",
                         "submersible motor pump"],
    "pipeline":         ["pipeline integrity", "pipeline inspection", "pig",
                         "flow assurance", "pipeline operations"],

    # Energy general
    "energy":           ["power", "utilities", "electricity", "renewables", "fossil fuel"],
    "renewable energy": ["solar energy", "wind energy", "green energy", "clean energy",
                         "photovoltaic", "wind turbine", "energy storage"],
    "solar":            ["photovoltaic", "PV", "solar panel", "solar farm", "solar power"],
    "wind":             ["wind turbine", "wind farm", "offshore wind", "onshore wind"],

    # Technology
    "software":         ["SaaS", "platform", "cloud software", "enterprise software",
                         "digital solution", "technology platform"],
    "digital":          ["digital transformation", "Industry 4.0", "IoT", "AI",
                         "machine learning", "automation", "cloud computing"],
    "analytics":        ["data analytics", "big data", "business intelligence", "BI",
                         "predictive analytics", "real-time monitoring"],
    "ai":               ["artificial intelligence", "machine learning", "deep learning",
                         "neural network", "predictive", "NLP"],
    "cybersecurity":    ["information security", "network security", "OT security",
                         "industrial cybersecurity", "cyber defense", "SCADA security"],
    "fintech":          ["financial technology", "payments", "banking software",
                         "insurtech", "regtech", "blockchain"],
    "healthcare":       ["medical", "health tech", "pharma", "clinical", "hospital",
                         "telemedicine", "medtech", "life sciences"],

    # Engineering & Industrial
    "engineering":      ["EPC", "technical services", "consulting engineering",
                         "project management", "FEED", "detailed design"],
    "manufacturing":    ["industrial manufacturing", "production", "assembly",
                         "process manufacturing", "OEM"],
    "construction":     ["building", "infrastructure", "EPC contractor",
                         "civil engineering", "structural"],
    "maintenance":      ["MRO", "predictive maintenance", "preventive maintenance",
                         "asset integrity", "condition monitoring"],

    # People / HR
    "engineer":         ["engineering professional", "technical specialist",
                         "petroleum engineer", "mechanical engineer", "electrical engineer",
                         "process engineer", "control engineer"],
    "manager":          ["management professional", "project manager", "operations manager",
                         "technical manager", "business development manager"],
    "hr":               ["human resources", "talent acquisition", "recruitment",
                         "people operations", "workforce management"],
    "director":         ["VP", "vice president", "executive", "head of", "general manager"],

    # Geography helpers
    "egypt":            ["cairo", "alexandria", "EGPC", "Egyptian", "Nile delta"],
    "saudi arabia":     ["KSA", "saudi", "aramco", "SABIC", "riyadh"],
    "uae":              ["dubai", "abu dhabi", "emirates", "ADNOC"],
    "usa":              ["united states", "american", "US-based", "houston", "texas"],
}

# Position level filters for people search
SENIORITY_LEVELS: Dict[str, List[str]] = {
    "engineer":  ["engineer", "engineering", "specialist", "analyst", "technologist",
                  "technical", "geologist", "geophysicist", "physicist"],
    "manager":   ["manager", "supervisor", "team lead", "project manager", "section head",
                  "superintendent", "foreman", "coordinator"],
    "director":  ["director", "vice president", "VP", "head of", "general manager", "GM",
                  "C-suite", "CEO", "COO", "CTO", "president"],
    "hr":        ["HR", "human resources", "talent", "recruitment", "recruiter",
                  "people", "workforce", "learning development", "L&D"],
    "executive": ["executive", "C-level", "leadership", "board", "managing director", "MD"],
}


def _dedup(lst: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in lst:
        key = item.lower().strip()
        if key not in seen and key:
            seen.add(key)
            result.append(item)
    return result


def expand_keywords(
    topic: str,
    entity_type: str = "company",
    task_type: str = "entity_discovery",
    geography: Optional[List[str]] = None,
    job_positions: Optional[List[str]] = None,
    llm=None,
) -> Dict[str, List[str]]:
    """
    Generate keyword expansion suggestions organized by category.

    Returns:
        {
          "synonyms":     ["petroleum", "hydrocarbons", ...],
          "sub_sectors":  ["upstream", "drilling services", ...],
          "company_types":["EPC contractor", "oilfield services", ...],
          "geo_variants": ["Cairo", "Egyptian market", ...],
          "job_variants": ["petroleum engineer", "reservoir engineer", ...],
          "search_operators": ['site:linkedin.com "oil gas"', ...],
        }
    """
    topic_lower = (topic or "").lower().strip()

    # Try LLM first for richer expansions
    if llm and llm.is_available():
        try:
            return _expand_with_llm(topic, entity_type, task_type, geography, job_positions, llm)
        except Exception:
            pass

    return _expand_rule_based(topic_lower, entity_type, task_type, geography, job_positions)


def _expand_with_llm(
    topic: str,
    entity_type: str,
    task_type: str,
    geography: Optional[List[str]],
    job_positions: Optional[List[str]],
    llm,
) -> Dict[str, List[str]]:
    geo_str = ", ".join(geography or []) or "global"
    pos_str = ", ".join(job_positions or []) or "any"

    prompt = f"""You are a search expert. The user wants to find: "{topic}" ({entity_type})
Geography: {geo_str}
Job positions filter: {pos_str}
Task: {task_type}

Generate keyword expansions to widen the search. Return ONLY this JSON:
{{
  "synonyms": ["alternative term 1", "alternative term 2", ...],
  "sub_sectors": ["specific sub-sector 1", "sub-sector 2", ...],
  "company_types": ["company type 1", "company type 2", ...],
  "geo_variants": ["location variant 1", "location variant 2", ...],
  "job_variants": ["job title 1", "job title 2", ...],
  "industry_codes": ["SIC code description 1", "NAICS category", ...]
}}

Rules:
- synonyms: 4-8 alternative terms for the main topic
- sub_sectors: 4-8 specific sub-segments within the industry
- company_types: 4-6 types of companies in this space
- geo_variants: 3-5 geographic variations (cities, regions, abbreviations)
- job_variants: 5-10 specific job titles relevant to the search (only if searching for people)
- industry_codes: 2-4 industry classification terms
- Be specific and practical — these become actual search query keywords
- Do NOT include generic words like "company", "industry", "business"
"""
    result = llm.generate_json(prompt, timeout=30)
    if result and isinstance(result, dict):
        # Ensure all keys exist
        for key in ["synonyms", "sub_sectors", "company_types", "geo_variants",
                    "job_variants", "industry_codes"]:
            if key not in result:
                result[key] = []
        return result

    return _expand_rule_based(topic.lower(), entity_type, task_type, geography, job_positions)


def _expand_rule_based(
    topic_lower: str,
    entity_type: str,
    task_type: str,
    geography: Optional[List[str]],
    job_positions: Optional[List[str]],
) -> Dict[str, List[str]]:
    """Rule-based expansion covering common industries."""
    synonyms: List[str]       = []
    sub_sectors: List[str]    = []
    company_types: List[str]  = []
    geo_variants: List[str]   = []
    job_variants: List[str]   = []
    industry_codes: List[str] = []

    # Find matching topic expansions
    for key, expansions in TOPIC_EXPANSIONS.items():
        if key in topic_lower:
            synonyms.extend(expansions[:4])

    # Industry-specific company types
    if any(w in topic_lower for w in ["oil", "gas", "petroleum", "oilfield", "upstream"]):
        company_types = [
            "oilfield services company", "drilling contractor", "EPC contractor",
            "well services provider", "production services company",
            "subsurface solutions company", "integrated energy company",
        ]
        sub_sectors = [
            "upstream exploration", "drilling services", "completion services",
            "well intervention", "production optimization", "pipeline services",
            "offshore services", "onshore services",
        ]
        industry_codes = [
            "SIC 1311 crude petroleum gas", "SIC 1381 drilling oil gas wells",
            "NAICS 211 oil gas extraction", "NAICS 213 support activities mining",
        ]
    elif any(w in topic_lower for w in ["software", "digital", "saas", "platform", "tech"]):
        company_types = [
            "SaaS company", "technology vendor", "software developer",
            "digital solutions provider", "cloud platform company",
        ]
        sub_sectors = [
            "enterprise software", "cloud computing", "data analytics",
            "AI/ML solutions", "IoT platform",
        ]
    elif any(w in topic_lower for w in ["renewable", "solar", "wind", "clean energy"]):
        company_types = [
            "solar developer", "wind energy company", "EPC solar",
            "energy storage provider", "green energy startup",
        ]
        sub_sectors = [
            "utility-scale solar", "distributed solar", "offshore wind",
            "onshore wind", "battery storage", "green hydrogen",
        ]
    elif any(w in topic_lower for w in ["healthcare", "medical", "pharma", "health"]):
        company_types = [
            "hospital", "pharmaceutical company", "medical device company",
            "health tech startup", "clinical research organization",
        ]
        sub_sectors = [
            "diagnostics", "therapeutics", "medical devices",
            "digital health", "telemedicine",
        ]
    elif any(w in topic_lower for w in ["fintech", "financial", "banking", "payment"]):
        company_types = [
            "fintech startup", "digital bank", "payment processor",
            "insurtech company", "robo-advisor",
        ]
        sub_sectors = [
            "digital payments", "lending", "wealthtech",
            "insurtech", "regtech", "blockchain",
        ]

    # Geography variants
    geo = geography or []
    for g in geo:
        g_lower = g.lower()
        geo_variants.extend(TOPIC_EXPANSIONS.get(g_lower, []))

    # Job variants for people search
    if entity_type == "person" or task_type == "people_search":
        for pos in (job_positions or []):
            pos_lower = pos.lower()
            for key, titles in SENIORITY_LEVELS.items():
                if key in pos_lower:
                    job_variants.extend(titles)

        # Add industry-specific job titles
        if any(w in topic_lower for w in ["oil", "gas", "petroleum"]):
            job_variants.extend([
                "petroleum engineer", "reservoir engineer", "drilling engineer",
                "production engineer", "completion engineer", "wellsite engineer",
                "facilities engineer", "process engineer", "HSE manager",
                "operations manager", "project manager", "technical director",
                "business development manager", "HR manager", "talent acquisition",
            ])

    # Deduplicate and clean
    return {
        "synonyms":       _dedup(synonyms)[:8],
        "sub_sectors":    _dedup(sub_sectors)[:8],
        "company_types":  _dedup(company_types)[:6],
        "geo_variants":   _dedup(geo_variants)[:5],
        "job_variants":   _dedup(job_variants)[:12],
        "industry_codes": _dedup(industry_codes)[:4],
    }


def build_expanded_queries(
    base_topic: str,
    selected_expansions: List[str],
    geography: Optional[List[str]] = None,
    entity_type: str = "company",
) -> List[str]:
    """
    Combine base topic with selected expansion terms to produce
    additional search queries.
    """
    queries = []
    geo_suffix = " ".join(geography[:1]) if geography else ""

    for term in selected_expansions:
        if entity_type == "person":
            queries.append(f'{term} {geo_suffix}'.strip())
        else:
            queries.append(f'{term} {entity_type} {geo_suffix}'.strip())

    # Also combine some expansions with the base topic
    for term in selected_expansions[:3]:
        queries.append(f'{base_topic} {term} {geo_suffix}'.strip())

    return [re.sub(r"\s+", " ", q).strip() for q in queries if q.strip()]
