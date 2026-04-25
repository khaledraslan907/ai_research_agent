from __future__ import annotations


INTENT_PARSE_PROMPT = """You are an AI research agent that understands what users want to find.

User request: "{prompt}"

Extract the search intent. Return ONLY this JSON (no markdown, no explanation):
{{
  "task_type": "entity_discovery" | "document_research" | "entity_enrichment" | "similar_entity_expansion" | "market_research" | "people_search",
  "entity_type": "company" | "paper" | "person" | "organization" | "event" | "product",
  "entity_category": "service_company" | "software_company" | "general",
  "topic": "<industry or subject only, e.g. oil and gas, CCS, renewable energy>",
  "solution_keywords": ["technical phrases explicitly present in the request"],
  "domain_keywords": ["domain or equipment phrases explicitly present in the request"],
  "commercial_intent": "general" | "agent_or_distributor" | "reseller" | "partner",
  "include_countries": ["countries or regions to search IN, empty if none"],
  "exclude_countries": ["countries to exclude by HQ / headquarters, empty if none"],
  "exclude_presence_countries": ["countries where company must NOT have offices / branches / subsidiaries / local entities / presence"],
  "attributes_wanted": ["website","email","phone","linkedin","summary","hq_country","presence_countries"],
  "output_format": "xlsx" | "csv" | "json" | "pdf",
  "max_results": <number, default 25>,
  "confidence": <0-100>
}}

RULES:
- topic = industry or subject only
- NEVER put country names in topic
- NEVER put contact/output words in topic
- entity_category = "software_company" for digital/software/AI/analytics/automation/platform/SaaS vendors
- entity_category = "service_company" for contractors, service companies, engineering firms, inspection, maintenance, wireline, logging, intervention, fabrication, rental, chemicals, offshore services, manpower, or field services
- Parse both English and Arabic requests
- If the request is Arabic, still return normalized English field values in the JSON
- Countries may be written in Arabic; normalize them to English canonical names
- Preserve niche phrases when the request is more specific than the broad industry

CRITICAL RULES FOR solution_keywords:
- solution_keywords must contain ONLY technical phrases explicitly written by the user
- DO NOT infer related keywords not mentioned by the user
- DO NOT expand one keyword into a broader taxonomy

CRITICAL RULES FOR domain_keywords:
- domain_keywords must capture domain-specific product/use-case phrases explicitly written by the user
- Keep equipment names, service lines, and niche phrases here
- Do NOT infer domain_keywords that the user did not explicitly mention

COMMERCIAL INTENT RULES:
- commercial_intent = "agent_or_distributor" when user wants agent, distributor, local representative, representation
- commercial_intent = "reseller" for reseller intent
- commercial_intent = "partner" for partner/channel-partner intent

GEOGRAPHY RULES:
- If the user says "in Egypt", use include_countries=["egypt"]
- If the user says "outside Egypt" or "no offices in Egypt", use exclude_presence_countries=["egypt"]
- "operate outside", "no offices in", "no branches in", "no subsidiaries in", "no local entities in", "exclude Egypt presence", "exclude USA presence" should map to exclude_presence_countries
- If user names a region, keep it as include_countries region text and let the parser expand it later when needed

OUTPUT RULES:
- output_format default is "xlsx" unless user says csv/json/pdf
- attributes_wanted must always include "website"
- add "email" if user mentions email/contact
- add "phone" if user mentions phone/number/contact
- add "linkedin" only if explicitly mentioned
"""


QUERY_PLAN_PROMPT = """You are a search expert. Generate search queries to find: {topic_description}

Task: {task_type}
Entity: {entity_category} {entity_type}
Include countries (search IN these): {include_countries}
Exclude HQ countries: {exclude_countries}
Exclude presence countries: {exclude_presence_countries}
Solution keywords: {solution_keywords}
Domain keywords: {domain_keywords}
Commercial intent: {commercial_intent}
Topic words to use: {topic_words}

Generate search queries. Return ONLY this JSON:
{{
  "ddg": ["<query 1>", "<query 2>", "<query 3>", "<query 4>"],
  "exa": ["<query 1>", "<query 2>", "<query 3>"],
  "tavily": ["<query 1>", "<query 2>"],
  "serpapi": ["<query 1>", "<query 2>"]
}}

RULES:
- DDG: keyword style, 3-10 words, avoid long sentences
- Exa: full natural-language ideal-result descriptions
- Tavily: questions or short natural phrases
- SerpApi: Google-style precise search strings
- NEVER repeat the same query across providers
- Each provider should take different angles
- Use solution_keywords explicitly when provided
- Use domain_keywords explicitly when provided
- If commercial_intent is agent_or_distributor / reseller / partner, include queries about distributors, resellers, representatives, channel partners, partner programs
- If include_countries is "any", do not invent one specific country
- If include_countries lists countries/regions, use them naturally in some queries
- If the market is Arabic-speaking or the request is Arabic, include at least one Arabic query in DDG and optionally SerpApi
- For paper/document searches: include academic / site / pdf / DOI / publisher variations
- For people searches: bias toward profile or team-page style queries
- Prefer real company / vendor / research / profile queries, not news, rankings, jobs, or directories
"""


RERANK_PROMPT = """You are a precise research analyst. Score these search results for relevance.

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
- score 8-10: perfect match
- score 5-7: likely relevant but incomplete
- score 2-4: weak / tangential
- score 0-1: irrelevant
- keep=true if score >= 5
- keep=false for directories, rankings, news, blogs, job boards, or irrelevant pages
- A company mentioning an excluded country as a client does NOT mean it is based there
"""


GEO_VERIFY_PROMPT = """You are a corporate intelligence analyst.

Company: {company_name}
Website: {website}
Description: {description}
Page text (first 1200 chars): {page_text}

Determine where this company is headquartered and where it has actual presence. Return ONLY this JSON:
{{
  "hq_country": "<lowercase country name or empty string>",
  "confidence": <0-100>,
  "evidence": "<short evidence>",
  "has_usa_presence": <true/false>,
  "has_egypt_presence": <true/false>,
  "presence_countries": ["countries with explicit office/branch/subsidiary/local-presence evidence"]
}}

RULES:
- hq_country = where the COMPANY itself is based
- serving clients in a country does NOT mean HQ there
- only set presence fields true if there is explicit office/branch/subsidiary/local-entity evidence
- if confidence < 40, return empty hq_country
"""


PAGE_CLASSIFY_PROMPT = """Classify this web page for research.

Sector: {sector}
Title: {title}
URL: {url}
Text (first 3000 chars): {text}

Return ONLY this JSON:
{{
  "page_type": "company" | "directory" | "blog" | "media" | "document" | "irrelevant" | "unknown",
  "is_relevant": true/false,
  "confidence": 0-100,
  "reason": "one sentence"
}}

RULES:
- page_type="company" ONLY if this is a real company's own website
- page_type="directory" if it lists multiple companies or contacts
- page_type="media" if it is news / magazine / press release
- page_type="blog" if opinion / tutorial / personal blog
- page_type="document" if it is a PDF/report/brochure/resource page with useful entity evidence
- is_relevant=true ONLY for actual entities/vendors/documents in the sector
"""


CONTACT_EXTRACT_PROMPT = """Extract contact information from this company page.

Company: {company_name}
URL: {url}
Page text: {text}

Return ONLY this JSON:
{{
  "emails": ["list of emails"],
  "phones": ["list of phone numbers"],
  "linkedin_url": "<linkedin company page url or empty string>",
  "contact_page_url": "<contact page url or empty string>",
  "hq_address": "<physical address or empty string>"
}}

RULES:
- include only real-looking emails
- phone numbers should preserve country code if present
- linkedin must be company page, not personal profile
- if multiple emails found, list all
"""
