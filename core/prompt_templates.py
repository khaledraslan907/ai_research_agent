"""
prompt_templates.py
===================
Single source of truth for every LLM prompt in the agent.

Why this file exists:
  - Prompts scattered across provider files are impossible to tune
  - Version-tagged prompts let you A/B test improvements
  - Jinja-style {placeholders} make substitution explicit and auditable
  - All prompts follow the same output contract: valid JSON, no markdown fences

Editing guide:
  - Each prompt ends with "Return ONLY valid JSON. No explanation."
  - Keep prompts short — Groq/Gemini free tiers have token limits
  - Use UPPERCASE for critical rules inside prompts (they stand out to LLMs)
  - Test changes with: python -m core.prompt_templates
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# INTENT PARSING
# Converts raw user prompt → structured TaskSpec JSON
# Used by: core/llm_task_parser.py
# ---------------------------------------------------------------------------

INTENT_PARSE_PROMPT = """You are an AI research agent that understands what users want to find.

User request: "{prompt}"

Extract the search intent. Return ONLY this JSON (no markdown, no explanation):
{{
  "task_type": "entity_discovery" | "document_research" | "entity_enrichment" | "similar_entity_expansion" | "market_research",
  "entity_type": "company" | "paper" | "person" | "organization" | "event" | "product",
  "entity_category": "service_company" | "software_company" | "general",
  "topic": "<the core subject being searched, e.g. oil and gas, CCS, renewable energy>",
  "include_countries": ["list of countries to search IN, empty if none"],
  "exclude_countries": ["list of countries to EXCLUDE from HQ/headquarters, empty if none"],
  "exclude_presence_countries": ["countries where company must NOT have offices, empty if none"],
  "attributes_wanted": ["website","email","phone","linkedin","summary","hq_country"],
  "output_format": "xlsx" | "csv" | "json" | "pdf",
  "max_results": <number, default 25>,
  "confidence": <0-100 how confident you are in this parse>
}}

RULES:
- topic = the INDUSTRY or SUBJECT ONLY. Strip out: country names, "email", "phone", "contact", "number", "with", "without", output format words
- For "Find service companies in Egypt working in oil and gas": topic = "oil and gas", include_countries = ["egypt"]
- For "Find papers about CCS in Europe": topic = "CCS", task_type = "document_research"
- For "Find digital oil gas companies outside USA and Egypt with email": topic = "oil and gas", entity_category = "software_company", exclude_countries = ["usa", "egypt"]
- NEVER put country names in the topic field
- NEVER put attribute words (email, phone, contact) in the topic field
- output_format default is "xlsx" unless user says csv/json/pdf
- attributes_wanted: always include "website". Add "email" if user mentions email/contact. Add "phone" if user mentions phone/number/contact. Add "linkedin" if explicitly mentioned."""


# ---------------------------------------------------------------------------
# QUERY PLANNING
# Converts TaskSpec → diverse search queries per provider
# Used by: core/llm_query_planner.py
# ---------------------------------------------------------------------------

QUERY_PLAN_PROMPT = """You are a search expert. Generate search queries to find: {topic_description}

Task: {task_type}
Entity: {entity_category} {entity_type}
Include countries (search IN these): {include_countries}
Exclude countries (NOT headquartered there): {exclude_countries}
Topic words to use: {topic_words}

Generate search queries. Return ONLY this JSON:
{{
  "ddg": [
    "<query 1>",
    "<query 2>",
    "<query 3>"
  ],
  "exa": [
    "<query 1>",
    "<query 2>"
  ],
  "tavily": [
    "<query 1>"
  ]
}}

RULES:
- DDG: keyword style, 3-6 words, NO negative operators (-"usa") in queries
- Exa: full natural language sentences describing the ideal result
- Tavily: questions or phrases
- NEVER repeat the same query across providers
- Each query must take a DIFFERENT angle (industry term, technology, market segment)
- CRITICAL: If include_countries is "any", do NOT invent or fixate on a specific country.
  Generate GLOBAL queries like "oil gas software company Europe" or "digital oilfield analytics UK Norway"
  NOT queries fixated on one country you guessed.
- If include_countries lists specific countries, use those countries in queries
- For paper/document searches: always include "site:pubmed" or "site:scholar" or "pdf" variations"""


# ---------------------------------------------------------------------------
# RESULT RE-RANKING
# LLM scores each candidate for relevance to the original request
# Used by: core/llm_ranker.py
# ---------------------------------------------------------------------------

RERANK_PROMPT = """You are a precise B2B research analyst. Score these search results for relevance.

User wanted: {user_request}
Topic: {topic}
Entity type: {entity_type}
Exclude HQ countries: {exclude_countries}

Results to score (index | name | domain | description):
{candidates}

Return ONLY a JSON array — one object per result, same order:
[
  {{"index": 0, "score": 0-10, "keep": true/false, "reason": "one sentence"}},
  ...
]

SCORING RULES:
- score 8-10: Perfect match. Clearly a {entity_type} in the {topic} sector.
- score 5-7: Likely relevant but missing some signals.
- score 2-4: Tangentially related, probably not what user wants.
- score 0-1: Completely irrelevant (football, real estate, guns, news, directories).
- keep=true if score >= 5
- keep=false if: it is a directory/list page, news article, job board, or completely off-topic
- If exclude_countries contains "usa" and the company is clearly US-based, score -= 3
- If exclude_countries contains "egypt" and company is clearly Egypt-based, score -= 3
- A company MENTIONING an excluded country as a client does NOT count as being based there"""


# ---------------------------------------------------------------------------
# GEO VERIFICATION
# Determines HQ country for ambiguous records
# Used by: core/geo_verifier.py
# ---------------------------------------------------------------------------

GEO_VERIFY_PROMPT = """You are a corporate intelligence analyst.

Company: {company_name}
Website: {website}
Description: {description}
Page text (first 800 chars): {page_text}

Determine where this company is headquartered. Return ONLY this JSON:
{{
  "hq_country": "<lowercase country name or empty string if unknown>",
  "confidence": <0-100>,
  "evidence": "<what text led you to this conclusion>",
  "has_usa_presence": <true/false — office or branch in USA>,
  "has_egypt_presence": <true/false — office or branch in Egypt>
}}

RULES:
- hq_country = where the COMPANY itself is based, NOT where it serves clients
- A UK company serving US clients is NOT a US company
- Mentions of "USA operations" or "Egyptian clients" do NOT mean the company is based there
- Only set has_usa_presence=true if there is explicit mention of a US office/branch
- If you cannot determine with confidence >= 40, return empty hq_country"""


# ---------------------------------------------------------------------------
# PAGE CLASSIFICATION
# Determines if a scraped page is a real company in the right sector
# Used by: providers/local_llm_provider.py
# ---------------------------------------------------------------------------

PAGE_CLASSIFY_PROMPT = """Classify this web page for B2B research.

Sector: {sector}
Title: {title}
URL: {url}
Text (first 3000 chars): {text}

Return ONLY this JSON:
{{
  "page_type": "company" | "directory" | "blog" | "media" | "irrelevant" | "unknown",
  "is_relevant": true/false,
  "confidence": 0-100,
  "reason": "one sentence"
}}

RULES:
- page_type="company" ONLY if this is a real company's own website in the {sector} sector
- page_type="directory" if it lists multiple companies (Top 10, Best companies, etc.)
- page_type="media" if it is a news article, press release, or magazine
- page_type="blog" if it is an opinion piece, tutorial, or personal blog
- page_type="irrelevant" if completely unrelated to {sector}
- is_relevant=true ONLY for actual companies/vendors in {sector}
- Do NOT mark directory/list pages as relevant even if they mention {sector}"""


# ---------------------------------------------------------------------------
# CONTACT EXTRACTION
# Extracts structured contact info from unstructured page text
# Used by: providers/structured_data_extractor.py
# ---------------------------------------------------------------------------

CONTACT_EXTRACT_PROMPT = """Extract contact information from this company page.

Company: {company_name}
URL: {url}
Page text: {text}

Return ONLY this JSON:
{{
  "emails": ["list of email addresses found, empty list if none"],
  "phones": ["list of phone numbers found, empty list if none"],
  "linkedin_url": "<LinkedIn company page URL or empty string>",
  "contact_page_url": "<URL of contact page if found, empty string if not>",
  "hq_address": "<physical address if found, empty string if not>"
}}

RULES:
- Only include REAL emails (not example@domain.com, info@sentry.io, etc.)
- Phone numbers should include country code if present
- LinkedIn must be a company page (linkedin.com/company/...) not a personal profile
- If multiple emails found, list all of them"""
