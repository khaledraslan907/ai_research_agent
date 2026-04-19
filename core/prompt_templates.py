from __future__ import annotations


INTENT_PARSE_PROMPT = """You are an AI research agent that understands what users want to find.

User request: "{prompt}"

Extract the search intent. Return ONLY this JSON (no markdown, no explanation):
{{
  "task_type": "entity_discovery" | "document_research" | "entity_enrichment" | "similar_entity_expansion" | "market_research" | "people_search",
  "entity_type": "company" | "paper" | "person" | "organization" | "event" | "product",
  "entity_category": "service_company" | "software_company" | "general",
  "topic": "<industry or subject only, e.g. oil and gas, CCS, renewable energy>",
  "solution_keywords": ["machine learning","artificial intelligence","ai","analytics","monitoring","optimization","automation","iot","scada","digital twin","predictive maintenance"],
  "domain_keywords": ["esp","virtual flow metering","well performance","artificial lift","production optimization","well surveillance","multiphase metering","flow assurance","production monitoring","reservoir simulation","reservoir modeling","drilling optimization","production engineering"],
  "commercial_intent": "general" | "agent_or_distributor" | "reseller" | "partner",
  "include_countries": ["countries to search IN, empty if none"],
  "exclude_countries": ["countries to exclude by HQ / headquarters, empty if none"],
  "exclude_presence_countries": ["countries where company must NOT have offices / branches / subsidiaries / local entities / presence"],
  "attributes_wanted": ["website","email","phone","linkedin","summary","hq_country","presence_countries"],
  "output_format": "xlsx" | "csv" | "json" | "pdf",
  "max_results": <number, default 25>,
  "confidence": <0-100>
}}

RULES:
- topic = INDUSTRY or SUBJECT ONLY
- NEVER put country names in topic
- NEVER put contact/output words in topic
- entity_category = "software_company" for digital/software/AI/analytics/automation/platform/SaaS vendors

CRITICAL RULES FOR solution_keywords:
- solution_keywords must contain ONLY technical phrases EXPLICITLY WRITTEN in the user's request
- DO NOT infer related keywords not mentioned by the user
- DO NOT expand one keyword into a broader taxonomy
- If the user says "machine learning", return "machine learning"
- If the user says "AI", return "ai"
- If the user says "artificial intelligence", return "artificial intelligence"
- If the user says both "machine learning" and "AI", return both
- Do NOT add analytics, monitoring, optimization, automation, IoT, SCADA, digital twin, or predictive maintenance unless they are explicitly present in the request

CRITICAL RULES FOR domain_keywords:
- domain_keywords must capture domain-specific product/use-case phrases explicitly written by the user
- Keep equipment names and niche oil-and-gas phrases here
- Examples: ESP, virtual flow metering, well performance, artificial lift
- Do NOT force these into topic if the broad industry is already clear
- Do NOT infer domain_keywords that the user did not explicitly mention

COMMERCIAL INTENT RULES:
- commercial_intent = "agent_or_distributor" when user wants agent, distributor, local representative, representation
- commercial_intent = "reseller" for reseller intent
- commercial_intent = "partner" for partner/channel-partner intent

GEOGRAPHY RULES:
- For "Find service companies in Egypt working in oil and gas": topic = "oil and gas", include_countries = ["egypt"]
- For "Find papers about CCS in Europe": topic = "CCS", task_type = "document_research"
- For "Find digital oil gas companies outside USA and Egypt with email": topic = "oil and gas", entity_category = "software_company"
- For "companies that do not have offices in Egypt or the United States": use exclude_presence_countries, NOT include_countries
- Phrases like "no offices in", "no branches in", "no subsidiaries in", "no local entities in", "exclude Egypt presence", "exclude USA presence", "operate outside" should map to exclude_presence_countries

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
- DDG: keyword style, 3-8 words, avoid long sentences
- Exa: full natural-language ideal-result descriptions
- Tavily: questions or short natural phrases
- SerpApi: Google-style precise search strings
- NEVER repeat the same query across providers
- Each provider should take DIFFERENT angles
- Use solution_keywords explicitly when provided
- Use domain_keywords explicitly when provided
- If commercial_intent is agent_or_distributor / reseller / partner, include queries about distributors, resellers, representatives, channel partners, partner programs
- If include_countries is "any", DO NOT invent one specific country
- If include_countries lists countries/regions, use them naturally in some queries
- If exclude_presence_countries exists, bias toward companies operating in Europe / GCC / Canada / Australia / Asia-Pacific and avoid phrasing that centers excluded countries
- For paper/document searches: include academic/site/pdf variations
- Prefer real company / vendor queries, not news, rankings, jobs, or directories
"""


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
Page text (first 800 chars): {page_text}

Determine where this company is headquartered. Return ONLY this JSON:
{{
  "hq_country": "<lowercase country name or empty string>",
  "confidence": <0-100>,
  "evidence": "<short evidence>",
  "has_usa_presence": <true/false>,
  "has_egypt_presence": <true/false>
}}

RULES:
- hq_country = where the COMPANY itself is based
- serving clients in a country does NOT mean HQ there
- only set presence flags true if there is explicit office/branch/subsidiary/local-entity evidence
- if confidence < 40, return empty hq_country
"""


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
- page_type="company" ONLY if this is a real company's own website
- page_type="directory" if it lists multiple companies
- page_type="media" if it is news / magazine / press release
- page_type="blog" if opinion / tutorial / personal blog
- is_relevant=true ONLY for actual companies/vendors in the sector
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
