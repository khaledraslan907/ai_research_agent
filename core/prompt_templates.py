from __future__ import annotations

INTENT_PARSE_PROMPT = """You are an AI research agent that understands what users want to find.

User request: "{prompt}"

Extract the search intent. Return ONLY this JSON (no markdown, no explanation):
{
  "task_type": "entity_discovery" | "document_research" | "entity_enrichment" | "similar_entity_expansion" | "market_research" | "people_search",
  "entity_type": "company" | "paper" | "person" | "organization" | "event" | "product",
  "entity_category": "service_company" | "software_company" | "general",
  "topic": "<industry or subject only>",
  "solution_keywords": ["machine learning","artificial intelligence","ai","analytics","monitoring","optimization","automation","iot","scada","digital twin","predictive maintenance"],
  "domain_keywords": ["wireline","well logging","pipeline inspection"],
  "commercial_intent": "general" | "agent_or_distributor" | "reseller" | "partner",
  "include_countries": ["countries to search IN, empty if none"],
  "exclude_countries": ["countries to exclude by HQ / headquarters, empty if none"],
  "exclude_presence_countries": ["countries where company must NOT have offices / branches / subsidiaries / local entities / presence"],
  "attributes_wanted": ["website","email","phone","linkedin","summary","hq_country","presence_countries"],
  "output_format": "xlsx" | "csv" | "json" | "pdf",
  "max_results": 25,
  "confidence": 0
}

RULES:
- topic = INDUSTRY or SUBJECT ONLY
- NEVER put country names in topic
- NEVER put output words like website/email/phone/names into topic
- For "Find software companies in food manufacturing in Germany with website and email":
  - entity_category = "software_company"
  - topic = "food manufacturing"
  - include_countries = ["germany"]
  - attributes_wanted = ["website","email"]
  - DO NOT put "software" into solution_keywords unless it is a technical capability
- For "Find EGYPS exhibitors related to wireline and well logging and extract company names and websites":
  - task_type = "market_research"
  - entity_type = "company"
  - entity_category = "service_company"
  - topic = "wireline well logging"
  - include_countries = ["egypt"]
  - domain_keywords = ["wireline","well logging"]
  - attributes_wanted = ["website"]
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

Return ONLY JSON:
{"ddg": ["<query 1>", "<query 2>"], "exa": ["<query 1>"], "tavily": ["<query 1>"], "serpapi": ["<query 1>"]}

RULES:
- Prefer niche/vertical phrases over broad generic ERP/consulting phrases
- For food manufacturing software, use food processing / MES / factory software variants
- For EGYPS exhibitor searches, use exhibitor/event/site-specific queries
"""
RERANK_PROMPT = ""
GEO_VERIFY_PROMPT = ""
PAGE_CLASSIFY_PROMPT = ""
CONTACT_EXTRACT_PROMPT = ""
