from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from core.budget import BudgetManager
from core.cache import CacheManager
from core.company_index import CompanyIndex
from core.config import (
    EXCLUDED_DOMAINS, KNOWN_USA_DOMAINS, KNOWN_EGYPT_DOMAINS,
    DIRECTORY_TITLE_PATTERNS,
)
from core.dedup import deduplicate_companies
from core.export_manager import export_records
from core.free_llm_client import FreeLLMClient
from core.llm_query_planner import plan_queries
from core.llm_ranker import rerank_records
from core.models import (
    CompanyRecord, ProviderSettings, SearchBudget, SearchResult, SearchSpec,
)
from core.task_models import TaskSpec, ExecutionPlan
from core.plan_builder import build_execution_plan
from core.provider_resolver import resolve_provider_keys
from core.query_builder import QueryBuilder
from core.scoring import score_records
from core.utils import extract_domain
from core.geography import contains_country_or_city, humanize_country_name

from providers.ddg_provider import DDGProvider
from providers.exa_provider import ExaProvider
from providers.local_llm_provider import LocalLLMProvider
from providers.serpapi_provider import SerpApiProvider
from providers.structured_data_extractor import StructuredDataExtractor
from providers.tavily_provider import TavilyProvider
from providers.website_scraper import WebsiteScraper


# Title-based directory detection (compiled once)
_DIR_PATTERNS = [re.compile(p, re.I) for p in DIRECTORY_TITLE_PATTERNS]


def _is_directory_title(title: str) -> bool:
    """Return True if the page title looks like a blog/list/directory, not a company."""
    t = (title or "").strip().lower()
    return any(p.search(t) for p in _DIR_PATTERNS)


class SearchOrchestrator:

    def __init__(self):
        self.cache = CacheManager()
        self.company_index = CompanyIndex()
        self.struct_extractor = StructuredDataExtractor()

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run_task(
        self,
        task_spec: TaskSpec,
        provider_settings: Optional[ProviderSettings] = None,
        uploaded_df: Optional[pd.DataFrame] = None,
        budget_overrides: Optional[dict] = None,
        min_confidence_score: int = 35,
        user_keys: Optional[dict] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:

        provider_settings = provider_settings or ProviderSettings()
        budget_overrides = budget_overrides or {}
        user_keys = user_keys or {}
        logs: List[str] = []

        def _log(msg: str):
            logs.append(msg)
            if progress_callback:
                progress_callback(msg)

        resolved_keys = resolve_provider_keys(task_spec.credential_mode.mode, user_keys)

        llm_client = FreeLLMClient(
            groq_api_key=resolved_keys.groq_api_key,
            gemini_api_key=resolved_keys.gemini_api_key,
            openrouter_api_key=resolved_keys.openrouter_api_key,
            anthropic_api_key=resolved_keys.anthropic_api_key,
            openai_api_key=resolved_keys.openai_api_key,
        )
        if llm_client.is_available():
            _log(f"LLM backends available: {', '.join(llm_client.available_backends())}")
        else:
            _log("No LLM backends — keyword-only mode")

        # disable providers without keys
        if task_spec.credential_mode.mode == "free":
            if not resolved_keys.exa_api_key:
                provider_settings.use_exa = False
            if not resolved_keys.tavily_api_key:
                provider_settings.use_tavily = False
            if not resolved_keys.serpapi_key:
                provider_settings.use_serpapi = False
            if not resolved_keys.firecrawl_api_key:
                provider_settings.use_firecrawl = False

        if task_spec.mode.lower() == "fast":
            provider_settings.use_llm_parser = False
            task_spec.use_cloud_llm = False

        ddg = DDGProvider()
        exa = ExaProvider(api_key=resolved_keys.exa_api_key)
        tavily = TavilyProvider(api_key=resolved_keys.tavily_api_key)
        serpapi = SerpApiProvider(api_key=resolved_keys.serpapi_key)
        scraper = WebsiteScraper(
            use_firecrawl=provider_settings.use_firecrawl,
            firecrawl_api_key=resolved_keys.firecrawl_api_key,
        )
        local_llm = LocalLLMProvider(
            groq_api_key=resolved_keys.groq_api_key,
            gemini_api_key=resolved_keys.gemini_api_key,
            openai_api_key=resolved_keys.openai_api_key,
            anthropic_api_key=resolved_keys.anthropic_api_key,
        )

        execution_plan = build_execution_plan(task_spec, provider_settings)
        _log(f"Strategy: {execution_plan.strategy_name} | Mode: {task_spec.mode}")

        if provider_settings.use_uploaded_seed_dedupe and uploaded_df is not None and not uploaded_df.empty:
            self.company_index.load_dataframe(uploaded_df)
            s = self.company_index.summary()
            _log(f"Loaded seed: {s['known_companies']} companies, {s['known_domains']} domains.")

        budget = self._build_budget(
            task_spec.mode,
            provider_settings,
            budget_overrides,
            execution_plan,
            task_spec.max_results,
        )
        budget_manager = BudgetManager(budget)

        all_planned_queries = plan_queries(task_spec, llm=llm_client)
        _log(
            f"Queries planned — "
            f"DDG:{len(all_planned_queries.get('ddg', []))} "
            f"Exa:{len(all_planned_queries.get('exa', []))} "
            f"Tavily:{len(all_planned_queries.get('tavily', []))} "
            f"SerpApi:{len(all_planned_queries.get('serpapi', []))}"
        )

        if getattr(task_spec, "solution_keywords", None):
            _log(f"Solution keywords: {', '.join(task_spec.solution_keywords)}")
        if getattr(task_spec, "domain_keywords", None):
            _log(f"Domain keywords: {', '.join(task_spec.domain_keywords)}")
        if getattr(task_spec, "commercial_intent", "general") != "general":
            _log(f"Commercial intent: {task_spec.commercial_intent}")

        all_search_results: List[SearchResult] = []
        records: List[CompanyRecord] = []
        seen_domains: Set[str] = set()

        # Stage 1 — DDG
        if provider_settings.use_ddg and execution_plan.use_ddg:
            ddg_queries = all_planned_queries.get("ddg", [])[:execution_plan.max_queries_per_provider.get("ddg", 5)]
            _log(f"Stage 1 DDG: {len(ddg_queries)} queries.")
            ddg_results = self._run_provider_queries(ddg, "ddg", ddg_queries, budget_manager, 10, logs)
            all_search_results.extend(ddg_results)
            records.extend(self._results_to_records(
                ddg_results, seen_domains, budget_manager, logs,
                task_spec, scraper, provider_settings, local_llm,
                execution_plan.max_candidates_to_process,
                execution_plan.use_local_llm_classifier,
            ))

        # Stage 2 — Exa
        if provider_settings.use_exa and execution_plan.use_exa and exa.is_available():
            exa_cap = execution_plan.max_queries_per_provider.get("exa", 4)
            exa_queries = all_planned_queries.get("exa", [])[:exa_cap]
            _log(f"Stage 2 Exa: {len(exa_queries)} queries (cap={exa_cap}).")
            exa_max = 15 if (task_spec.max_results or 25) > 50 else 10

            if task_spec.task_type == "people_search":
                linkedin_qs = [q for q in exa_queries if q.family == "linkedin_exa"]
                normal_qs = [q for q in exa_queries if q.family != "linkedin_exa"]
                exa_results = []
                if linkedin_qs:
                    _log(f"  Exa LinkedIn-scoped: {len(linkedin_qs)} queries → linkedin.com/in only")
                    exa_results.extend(
                        self._run_exa_linkedin_queries(exa, linkedin_qs, budget_manager, exa_max, logs)
                    )
                if normal_qs:
                    _log(f"  Skipping {len(normal_qs)} non-linkedin EXA queries for people_search")
            else:
                exa_results = self._run_provider_queries(exa, "exa", exa_queries, budget_manager, exa_max, logs)

            all_search_results.extend(exa_results)
            records.extend(self._results_to_records(
                exa_results, seen_domains, budget_manager, logs,
                task_spec, scraper, provider_settings, local_llm,
                execution_plan.max_candidates_to_process,
                execution_plan.use_local_llm_classifier,
            ))

        # Stage 2b — Exa find_similar
        if execution_plan.use_exa_find_similar and exa.is_available() and records:
            seed_urls = [r.website for r in sorted(records, key=lambda x: x.confidence_score, reverse=True) if r.website][:3]
            _log(f"Stage 2b Exa find_similar: {len(seed_urls)} seeds.")
            for seed_url in seed_urls:
                if not budget_manager.can_use_provider("exa"):
                    break
                cached = self.cache.get_query("exa_similar", seed_url, 5)
                if cached:
                    similar_results = [SearchResult(**item) for item in cached]
                else:
                    similar_results = exa.find_similar(seed_url, max_results=5)
                    budget_manager.register_search_call("exa")
                    self.cache.set_query("exa_similar", seed_url, 5, [r.to_dict() for r in similar_results])
                all_search_results.extend(similar_results)
                records.extend(self._results_to_records(
                    similar_results, seen_domains, budget_manager, logs,
                    task_spec, scraper, provider_settings, local_llm,
                    execution_plan.max_candidates_to_process,
                    execution_plan.use_local_llm_classifier,
                ))

        # Stage 3 — Tavily
        if provider_settings.use_tavily and execution_plan.use_tavily and tavily.is_available():
            tavily_queries = all_planned_queries.get("tavily", [])[:execution_plan.max_queries_per_provider.get("tavily", 3)]
            _log(f"Stage 3 Tavily: {len(tavily_queries)} queries.")
            tavily_results = self._run_provider_queries(tavily, "tavily", tavily_queries, budget_manager, 10, logs)
            all_search_results.extend(tavily_results)
            records.extend(self._results_to_records(
                tavily_results, seen_domains, budget_manager, logs,
                task_spec, scraper, provider_settings, local_llm,
                execution_plan.max_candidates_to_process,
                execution_plan.use_local_llm_classifier,
            ))

        # Stage 4 — SerpApi
        if provider_settings.use_serpapi and execution_plan.use_serpapi and serpapi.is_available():
            serp_queries = all_planned_queries.get("serpapi", [])[:execution_plan.max_queries_per_provider.get("serpapi", 3)]
            _log(f"Stage 4 SerpApi: {len(serp_queries)} queries.")
            serp_results = self._run_provider_queries(serpapi, "serpapi", serp_queries, budget_manager, 10, logs)
            all_search_results.extend(serp_results)
            records.extend(self._results_to_records(
                serp_results, seen_domains, budget_manager, logs,
                task_spec, scraper, provider_settings, local_llm,
                execution_plan.max_candidates_to_process,
                execution_plan.use_local_llm_classifier,
            ))

        # Post-search
        _log(f"Raw candidates before dedup: {len(records)}")
        records = deduplicate_companies(records)
        _log(f"After dedup: {len(records)}")

        search_spec = self._task_to_search_spec(task_spec)
        records = score_records(records, search_spec)
        records.sort(key=lambda x: x.confidence_score, reverse=True)

        if (
            llm_client.is_available()
            and records
            and task_spec.mode.lower() != "fast"
            and task_spec.task_type != "people_search"
        ):
            _log(f"LLM re-ranking {min(len(records), 80)} candidates...")
            records = rerank_records(records=records, task_spec=task_spec, llm=llm_client, batch_size=40)
            records.sort(key=lambda x: x.confidence_score, reverse=True)

        final_records: List[CompanyRecord] = []
        rejected_records: List[CompanyRecord] = []

        for rec in records:
            reason = self._get_reject_reason(rec, task_spec, min_confidence_score)
            if reason:
                rec.notes = (rec.notes + f" | rejected:{reason}").strip(" |")
                rejected_records.append(rec)
                continue
            self.company_index.add_record(rec)
            final_records.append(rec)
            if len(final_records) >= task_spec.max_results:
                _log(f"Reached target {task_spec.max_results}.")
                break

        _log(f"Accepted: {len(final_records)} | Rejected: {len(rejected_records)}")
        if not final_records and task_spec.geography.strict_mode and task_spec.task_type not in {"people_search", "document_research"}:
            _log("No verified records satisfied strict geography evidence. This is better than returning false positives.")

        if task_spec.task_type in {"entity_discovery", "entity_enrichment", "similar_entity_expansion"}:
            final_records = self._enrich_accepted_records(final_records, scraper, budget_manager, logs, task_spec)

        export_path = export_records(
            records=final_records,
            output_format=task_spec.output.format,
            filename=task_spec.output.filename,
            task_type=task_spec.task_type,
        )

        return {
            "task_spec": task_spec.to_dict(),
            "execution_plan": execution_plan.to_dict(),
            "queries": {k: [q.to_dict() for q in v] for k, v in all_planned_queries.items()},
            "records": [r.to_dict() for r in final_records],
            "rejected_records": [r.to_dict() for r in rejected_records],
            "logs": logs,
            "budget": budget_manager.summary(),
            "company_index": self.company_index.summary(),
            "resolved_keys": resolved_keys.to_dict(),
            "llm_backends": llm_client.available_backends(),
            "export_path": str(export_path) if export_path else "",
            "total_found": len(final_records),
            "raw_search_results": len(all_search_results),
        }

    # ==================================================================
    # Provider query runner
    # ==================================================================

    def _run_provider_queries(
        self,
        provider,
        provider_name: str,
        queries: list,
        budget_manager: BudgetManager,
        max_results_per_query: int,
        logs: List[str],
    ) -> List[SearchResult]:
        all_results: List[SearchResult] = []

        for query in queries:
            if not budget_manager.can_use_provider(provider_name):
                logs.append(f"Budget cap reached for {provider_name}.")
                break
            q_text = query.text if hasattr(query, "text") else str(query)

            cached = self.cache.get_query(provider_name, q_text, max_results_per_query)
            if cached:
                results = [SearchResult(**item) for item in cached]
                logs.append(f"[{provider_name}] cache hit: '{q_text[:50]}' → {len(results)}")
            else:
                try:
                    results = provider.search(q_text, max_results=max_results_per_query)
                    budget_manager.register_search_call(provider_name)
                    self.cache.set_query(provider_name, q_text, max_results_per_query,
                                         [r.to_dict() for r in results])
                    logs.append(f"[{provider_name}] '{q_text[:50]}' → {len(results)} results")
                except Exception as e:
                    logs.append(f"[{provider_name}] error: {e}")
                    results = []

            all_results.extend(results)
        return all_results

    # ==================================================================
    # Convert search results → CompanyRecord
    # ==================================================================

    def _run_exa_linkedin_queries(
        self,
        exa,
        queries: list,
        budget_manager: BudgetManager,
        max_results_per_query: int,
        logs: List[str],
    ) -> List[SearchResult]:
        """
        Run EXA queries with include_domains=["linkedin.com/in"].
        Every result is guaranteed to be a linkedin.com/in/ personal profile.
        """
        all_results: List[SearchResult] = []
        for query in queries:
            if not budget_manager.can_use_provider("exa"):
                logs.append("EXA budget cap reached.")
                break
            q_text = query.text if hasattr(query, "text") else str(query)
            cached = self.cache.get_query("exa_linkedin", q_text, max_results_per_query)
            if cached:
                results = [SearchResult(**item) for item in cached]
                logs.append(f"[exa/linkedin] cache: '{q_text[:50]}' → {len(results)}")
            else:
                try:
                    results = exa.search_linkedin_profiles(q_text, max_results=max_results_per_query)
                    budget_manager.register_search_call("exa")
                    self.cache.set_query("exa_linkedin", q_text, max_results_per_query,
                                         [r.to_dict() for r in results])
                    logs.append(f"[exa/linkedin] '{q_text[:50]}' → {len(results)} profiles")
                except Exception as e:
                    logs.append(f"[exa/linkedin] error: {e}")
                    results = []
            all_results.extend(results)
        return all_results

    def _results_to_records(
        self,
        search_results: List[SearchResult],
        seen_domains: Set[str],
        budget_manager: BudgetManager,
        logs: List[str],
        task_spec: TaskSpec,
        scraper: WebsiteScraper,
        provider_settings: ProviderSettings,
        local_llm: LocalLLMProvider,
        max_candidates: int = 50,
        use_llm_classifier: bool = False,
    ) -> List[CompanyRecord]:
        records: List[CompanyRecord] = []
        processed: int = 0

        exclude_countries = [c.lower() for c in (task_spec.geography.exclude_countries or [])]

        for res in search_results:
            if processed >= max_candidates:
                break

            domain = extract_domain(res.url)
            if not domain:
                continue

            # ── People search: deduplicate by full URL, not domain ─────────
            if task_spec.task_type == "people_search":
                url_key = res.url.lower().rstrip("/")
                if url_key in seen_domains:
                    continue
                from core.people_search import extract_person_from_linkedin_result
                _name_preview = extract_person_from_linkedin_result(
                    res.title, res.url, res.snippet
                ).get("name", "")
                if self.company_index.contains_linkedin_profile(res.url, _name_preview):
                    logs.append(f"Skip (known profile): {_name_preview or res.url[:50]}")
                    continue
            else:
                # ── Company/paper search: deduplicate by domain ────────────
                if domain in seen_domains:
                    continue
                _skip_known = self.company_index.contains_company(res.title, domain)
                if _skip_known and (task_spec.max_results or 25) <= 50:
                    logs.append(f"Skip (known): {domain}")
                    continue
                elif _skip_known:
                    if domain in seen_domains:
                        continue

            # ── LinkedIn URL handling ─────────────────────────────────────────
            url_lower_check = res.url.lower()
            is_linkedin = "linkedin.com" in url_lower_check

            if is_linkedin:
                if task_spec.task_type == "people_search":
                    from core.people_search import is_linkedin_profile_url, is_linkedin_company_url
                    if is_linkedin_company_url(res.url):
                        logs.append(f"Skip (LinkedIn company page, want /in/): {res.url[:60]}")
                        continue
                    if not is_linkedin_profile_url(res.url):
                        logs.append(f"Skip (LinkedIn non-profile URL): {res.url[:60]}")
                        continue
                else:
                    logs.append(f"Skip (excluded domain): {domain}")
                    continue

            elif domain in EXCLUDED_DOMAINS or any(domain.endswith("." + d) for d in EXCLUDED_DOMAINS):
                logs.append(f"Skip (excluded domain): {domain}")
                continue

            # ── Title-based directory detection (before any scraping) ──────────
            if task_spec.task_type not in {"document_research", "people_search"} and _is_directory_title(res.title):
                logs.append(f"Skip (directory title): {res.title[:60]}")
                continue

            # ── Known-domain geo pre-filter ───────────────────────────────────
            if "usa" in exclude_countries and domain in KNOWN_USA_DOMAINS:
                logs.append(f"Skip (known USA domain): {domain}")
                continue
            if "egypt" in exclude_countries and domain in KNOWN_EGYPT_DOMAINS:
                logs.append(f"Skip (known Egypt domain): {domain}")
                continue

            # ── URL-path geo signal ───────────────────────────────────────────
            url_lower = res.url.lower()
            if "usa" in exclude_countries and not is_linkedin:
                if any(s in url_lower for s in ["/en-us/", "/us/en/", "/us-en/", "/us/"]):
                    if domain.endswith(".com") and not any(
                        s in url_lower for s in ["/global/", "/international/", "/world/"]
                    ):
                        logs.append(f"Skip (US URL path): {res.url[:60]}")
                        continue

            if task_spec.task_type == "people_search":
                seen_domains.add(res.url.lower().rstrip("/"))
            else:
                seen_domains.add(domain)

            company = CompanyRecord(
                company_name=res.title.strip(),
                website=res.url,
                domain=domain,
                description=res.snippet.strip(),
                source_url=res.url,
                source_provider=res.provider,
                raw_sources=[res.to_dict()],
            )

            # ── PEOPLE SEARCH: extract from snippet, never scrape LinkedIn ────
            if task_spec.task_type == "people_search":
                from core.people_search import extract_person_from_linkedin_result
                info = extract_person_from_linkedin_result(
                    title=res.title,
                    url=res.url,
                    snippet=res.snippet,
                )
                company.company_name = info.get("name", "") or res.title
                company.job_title = info.get("job_title", "")
                company.employer_name = info.get("employer", "")
                company.city = info.get("location", "")
                company.linkedin_url = res.url
                company.linkedin_profile = res.url
                company.page_type = "person"
                company.is_directory_or_media = False
                company.confidence_score = 85.0
                records.append(company)
                processed += 1
                continue

            # try scrape cache
            cached_scrape = self.cache.get_scrape(domain)
            if cached_scrape:
                logs.append(f"Scrape cache hit: {domain}")
                self._apply_scrape_cache(company, cached_scrape)
                records.append(company)
                processed += 1
                continue

            # live scrape
            if budget_manager.can_scrape_pages(1):
                follow_internal = task_spec.task_type != "document_research"
                crawl = scraper.scrape(res.url, follow_internal_links=follow_internal)
                budget_manager.register_scraped_pages(1)

                if crawl.success:
                    self._apply_crawl_to_record(company, crawl, res, task_spec)
                    if (
                        provider_settings.use_llm_parser
                        and use_llm_classifier
                        and local_llm.is_available()
                        and task_spec.task_type != "document_research"
                    ):
                        self._apply_llm_classification(company, crawl, task_spec, local_llm, logs)
                    self.cache.set_scrape(domain, company.to_dict())
                else:
                    logs.append(f"Scrape failed: {res.url} — {crawl.error}")
            else:
                logs.append("Page scrape budget exhausted.")

            records.append(company)
            processed += 1

        return records

    def _apply_scrape_cache(self, company: CompanyRecord, cached: dict):
        company.company_name = cached.get("company_name", company.company_name)
        company.description = cached.get("description", company.description)
        company.email = cached.get("email", "")
        company.phone = cached.get("phone", "")
        company.hq_country = cached.get("hq_country", "")
        company.country = cached.get("country", "")
        company.presence_countries = cached.get("presence_countries", [])
        company.contact_page = cached.get("contact_page", "")
        company.linkedin_url = cached.get("linkedin_url", "")
        company.has_usa_presence = cached.get("has_usa_presence", False)
        company.has_egypt_presence = cached.get("has_egypt_presence", False)
        company.is_directory_or_media = cached.get("is_directory_or_media", False)
        company.page_type = cached.get("page_type", "")
        company.notes = cached.get("notes", "")
        company.authors = cached.get("authors", "")
        company.doi = cached.get("doi", "")

    def _apply_crawl_to_record(
        self,
        company: CompanyRecord,
        crawl,
        res: SearchResult,
        task_spec: TaskSpec,
    ):
        company.company_name = crawl.detected_company_name or company.company_name
        company.description = crawl.meta_description or company.description

        if task_spec.task_type != "document_research":
            structured = self.struct_extractor.extract_from_html(
                html=getattr(crawl, "_raw_html", ""),
                base_url=company.website or "",
            )

            if not crawl.emails and not structured.get("emails"):
                probed = self.struct_extractor.probe_contact_pages(
                    base_url=company.website or "",
                    existing_emails=crawl.emails,
                    existing_phones=crawl.phones,
                    max_probes=2,
                )
                structured["emails"] = structured.get("emails", []) + probed.get("emails", [])
                structured["phones"] = structured.get("phones", []) + probed.get("phones", [])
                if not structured.get("linkedin_url") and probed.get("linkedin_url"):
                    structured["linkedin_url"] = probed["linkedin_url"]

            from core.utils import unique_list
            all_emails = unique_list(structured.get("emails", []) + crawl.emails)
            all_phones = unique_list(structured.get("phones", []) + crawl.phones)

            company.email = all_emails[0] if all_emails else ""
            company.phone = all_phones[0] if all_phones else ""
            company.contact_page = crawl.contact_links[0] if crawl.contact_links else ""
            company.linkedin_url = (structured.get("linkedin_url") or crawl.social_links.get("linkedin", ""))

            structured_hq = structured.get("hq_country", "")
            company.hq_country = structured_hq or crawl.detected_hq_country or ""
            company.country = company.hq_country
            company.presence_countries = crawl.detected_presence_countries or []
            company.has_usa_presence = "usa" in [c.lower() for c in company.presence_countries]
            company.has_egypt_presence = "egypt" in [c.lower() for c in company.presence_countries]

            if not company.company_name and structured.get("company_name"):
                company.company_name = structured["company_name"]

        # page-type classification
        haystack = " ".join([
            company.website or "", company.source_url or "",
            company.description or "", res.title or "", res.snippet or "",
            crawl.text[:500] if crawl.text else "",
        ]).lower()

        if task_spec.task_type == "document_research":
            if any(x in haystack for x in [
                "journal", "paper", "study", "publication", "report",
                "doi", "abstract", "preprint", "peer-reviewed", "pmid",
            ]):
                company.page_type = "document"
            else:
                company.page_type = "unknown"
            company.is_directory_or_media = False

            if not company.authors:
                company.authors = self._extract_authors_from_text(
                    company.description or ""
                )
            if not company.doi:
                doi_m = re.search(r"10\.\d{4,}/\S+", company.description or "")
                if doi_m:
                    company.doi = f"https://doi.org/{doi_m.group()}"
        else:
            if _is_directory_title(company.company_name):
                company.page_type = "directory"
                company.is_directory_or_media = True
            elif any(x in haystack for x in [
                "press release", "newsroom", "breaking news", "magazine",
            ]):
                company.page_type = "media"
                company.is_directory_or_media = True
            elif any(x in haystack for x in ["blog post", "how to guide", "tutorial"]):
                company.page_type = "blog"
                company.is_directory_or_media = True
            else:
                company.page_type = "company"
                company.is_directory_or_media = False

    def _apply_llm_classification(
        self, company: CompanyRecord, crawl, task_spec: TaskSpec,
        local_llm: LocalLLMProvider, logs: List[str],
    ):
        llm_page = local_llm.classify_company_page(
            title=company.company_name or "",
            url=company.website or "",
            text=crawl.text or company.description or "",
            sector=task_spec.industry or "oil and gas",
        )
        llm_page_type = (llm_page.get("page_type") or "").lower()
        llm_confidence = int(llm_page.get("confidence", 50))

        if llm_page_type and llm_page_type != "unknown":
            if company.page_type == "unknown" or llm_confidence >= 75:
                company.page_type = llm_page_type

        if company.page_type in {"directory", "media", "blog", "irrelevant"} and llm_confidence >= 80:
            company.is_directory_or_media = True
        else:
            company.is_directory_or_media = False

        llm_presence = local_llm.classify_presence(
            company_name=company.company_name,
            text=crawl.text or company.description or "",
        )
        if llm_presence.get("hq_country") and not company.hq_country:
            company.hq_country = llm_presence["hq_country"]
            company.country = llm_presence["hq_country"]

    def _extract_authors_from_text(self, text: str) -> str:
        """
        Multi-strategy author extraction from academic paper descriptions.
        """
        if not text:
            return ""
        chunk = text[:6000]

        ABSTRACT_WORDS = {
            "abstract", "using", "results", "methods", "study", "approach",
            "analysis", "based", "show", "data", "research", "presented",
            "model", "method", "technique", "system", "maximizing", "improving",
            "considering", "alongside", "consistent", "poor", "deploying",
            "background", "objective", "conclusion", "findings", "introduction",
        }

        def _valid(s):
            if not s or len(s) < 5:
                return False
            return not (set(s.lower().split()[:3]) & ABSTRACT_WORDS)

        candidates = []

        m = re.search(
            r"[*]\s*([A-Z][a-zA-Zéàüöçõã-]+"
            r",\s+[A-Z][a-zA-Zéàüöçõã-]+"
            r"(?:;\s*[*]?\s*[A-Z][a-zA-Zéàüöçõã-]+"
            r",\s+[A-Z][a-zA-Zéàüöçõã-]+)+)",
            chunk)
        if m:
            candidates.append(re.sub(r"[*]\s*", "", m.group(1)).strip())

        m = re.search(
            r"([A-Z][a-z]+ [A-Z][.] [A-Za-z]+\d*[,*]*"
            r"(?:,? [A-Z][a-z]+ [A-Z][.] [A-Za-z]+\d*[,*]*)*"
            r"(?: and [A-Z][a-z]+ [A-Z][.] [A-Za-z]+)?)",
            chunk)
        if m:
            raw = re.sub(r"[\d*]+", "", m.group(1)).strip().rstrip(",")
            candidates.append(raw)

        m = re.search(
            r"([A-Z][a-zéáúóí]+"
            r" [A-Z][a-zéáúóí]+"
            r"(?:\s*\d+)?,"
            r"\s*[A-Z][a-zéáúóí]+"
            r" [A-Z][a-zéáúóí]+)",
            chunk)
        if m:
            candidates.append(re.sub(r"\s*\d+", "", m.group(1)).strip())

        m = re.search(
            r"written\s+by\s+([A-Z][a-z]+ [A-Z][a-z]+"
            r"(?:,\s+[A-Z][a-z]+ [A-Z][a-z]+)*)",
            chunk, re.I)
        if m:
            candidates.append(m.group(1))

        m = re.search(
            r"([A-Z][a-z]+,\s+[A-Z][a-z]+(?: [A-Z][.])?(?:;\s+"
            r"[A-Z][a-z]+,\s+[A-Z][a-z]+(?: [A-Z][.])?)+)",
            chunk)
        if m:
            candidates.append(m.group(1))

        m = re.search(
            r"[Bb]y ([A-Z][a-z]+ [A-Z][a-z]+(?: and [A-Z][a-z]+ [A-Z][a-z]+)*)",
            chunk)
        if m:
            candidates.append(m.group(1))

        for c in sorted(candidates, key=len, reverse=True):
            if _valid(c):
                return c[:300]
        return ""

    def _enrich_accepted_records(
        self,
        records: List[CompanyRecord],
        scraper: WebsiteScraper,
        budget_manager: BudgetManager,
        logs: List[str],
        task_spec: TaskSpec,
    ) -> List[CompanyRecord]:
        for rec in records:
            if rec.email and rec.phone:
                continue
            if not budget_manager.can_scrape_pages(1):
                break
            try:
                crawl = scraper.scrape(rec.website or rec.source_url or "", follow_internal_links=True)
                budget_manager.register_scraped_pages(1)
                if crawl.success:
                    if not rec.email and crawl.emails:
                        rec.email = crawl.emails[0]
                    if not rec.phone and crawl.phones:
                        rec.phone = crawl.phones[0]
                    if not rec.linkedin_url:
                        rec.linkedin_url = crawl.social_links.get("linkedin", "")
                    logs.append(f"Enrichment updated: {rec.company_name}")
            except Exception as e:
                logs.append(f"Enrichment failed: {rec.company_name} — {e}")
        return records

    # ==================================================================
    # Rejection logic
    # ==================================================================

    def _get_reject_reason(
        self,
        rec: CompanyRecord,
        task_spec: TaskSpec,
        min_confidence_score: int,
    ) -> Optional[str]:

        if task_spec.task_type == "people_search":
            from core.people_search import is_linkedin_profile_url, is_linkedin_company_url
            url = rec.linkedin_url or rec.linkedin_profile or rec.website or ""
            if is_linkedin_company_url(url):
                return "linkedin_company_page"
            if not is_linkedin_profile_url(url):
                return "not_linkedin_profile"
            if not rec.company_name and not rec.job_title:
                return "no_person_info"
            return None

        llm_rank = 0
        rank_m = re.search(r"llm_rank:(\d+)/10", rec.notes or "")
        if rank_m:
            llm_rank = int(rank_m.group(1))

        if rec.confidence_score < min_confidence_score:
            if llm_rank >= 8:
                pass
            else:
                return f"low_confidence({rec.confidence_score:.0f}<{min_confidence_score})"

        if rec.is_directory_or_media and llm_rank < 7:
            return "directory_or_media"

        if task_spec.task_type != "document_research" and _is_directory_title(rec.company_name):
            if llm_rank >= 8:
                rec.is_directory_or_media = False
            else:
                return "directory_or_media"

        if self._violates_geography(rec, task_spec):
            return "geography_violation"

        if self.company_index.contains_company(rec.company_name, rec.domain):
            return "duplicate"

        # strict geography searches should not keep globally ambiguous company pages
        if task_spec.geography.strict_mode and task_spec.task_type not in {"people_search", "document_research"}:
            geo_known = bool((rec.hq_country or "").strip() or (rec.country or "").strip() or list(rec.presence_countries or []))
            text_blob = " ".join([rec.company_name or "", rec.description or "", rec.notes or "", rec.website or "", rec.source_url or ""]).lower()
            has_include_hint = any(contains_country_or_city(text_blob, c) for c in (task_spec.geography.include_countries or []))
            if not geo_known and not has_include_hint:
                return "missing_required_geography_evidence"

        return None

    def _violates_geography(self, rec: CompanyRecord, task_spec: TaskSpec) -> bool:
        exclude_countries = [c.lower() for c in (task_spec.geography.exclude_countries or [])]
        exclude_presence = [c.lower() for c in (task_spec.geography.exclude_presence_countries or [])]
        include_countries = [c.lower() for c in (task_spec.geography.include_countries or [])]

        rec_hq = (rec.hq_country or "").lower().strip()
        rec_country = (rec.country or "").lower().strip()
        presence = [c.lower().strip() for c in (rec.presence_countries or [])]
        domain = (rec.domain or "").lower()
        website = (rec.website or rec.source_url or "").lower()
        evidence_text = " ".join([
            rec.company_name or "", rec.description or "", rec.notes or "",
            rec.website or "", rec.source_url or "", rec.contact_page or "", rec.email or "",
        ]).lower()

        def _known_match(country: str) -> bool:
            if not country:
                return False
            if rec_hq == country or rec_country == country:
                return True
            if country in presence:
                return True
            if contains_country_or_city(evidence_text, country):
                return True
            if country == "egypt" and domain in KNOWN_EGYPT_DOMAINS:
                return True
            if country == "usa" and domain in KNOWN_USA_DOMAINS:
                return True
            if country == "egypt" and any(s in website for s in ["/egypt", "cairo", ".eg/"]):
                return True
            if country == "usa" and any(s in website for s in ["/us/", "/en-us/", "houston", "texas"]):
                return True
            return False

        if exclude_countries:
            for c in exclude_countries:
                if _known_match(c):
                    return True

        if exclude_presence:
            for c in exclude_presence:
                if c in presence:
                    return True
                if c == "usa" and rec.has_usa_presence:
                    return True
                if c == "egypt" and rec.has_egypt_presence:
                    return True
                # explicit presence-style text should reject for exclude_presence
                if c and contains_country_or_city(evidence_text, c) and any(x in evidence_text for x in ["office", "branch", "presence", "operations", "مكتب", "فرع", "تواجد"]):
                    return True

        # strict include mode: for company/entity searches, require positive evidence instead of allowing global unknowns
        if include_countries and task_spec.geography.strict_mode and task_spec.task_type != "people_search":
            matched = any(_known_match(c) for c in include_countries)
            if not matched:
                return True

        return False

    # ==================================================================
    # Helpers
    # ==================================================================

    def _task_to_search_spec(self, task_spec: TaskSpec) -> SearchSpec:
        required = list(task_spec.target_attributes or [])
        optional = list(task_spec.target_attributes or [])

        if task_spec.task_type == "document_research":
            required = ["website"]
            optional = ["website", "summary"]

        include_terms = list(getattr(task_spec, "solution_keywords", []) or []) + list(getattr(task_spec, "domain_keywords", []) or [])

        return SearchSpec(
            raw_prompt=task_spec.raw_prompt,
            entity_type=(task_spec.target_entity_types[0] if task_spec.target_entity_types else "company"),
            intent_type=task_spec.task_type,
            sector=task_spec.industry or "",
            target_category=getattr(task_spec, "target_category", "general") or "general",
            solution_keywords=list(getattr(task_spec, "solution_keywords", []) or []),
            domain_keywords=list(getattr(task_spec, "domain_keywords", []) or []),
            commercial_intent=getattr(task_spec, "commercial_intent", "general") or "general",
            include_terms=include_terms,
            exclude_terms=[],
            include_countries=list(task_spec.geography.include_countries or []),
            exclude_countries=list(task_spec.geography.exclude_countries or []),
            exclude_presence_countries=list(task_spec.geography.exclude_presence_countries or []),
            required_fields=required,
            optional_fields=optional,
            max_results=task_spec.max_results,
            mode=task_spec.mode,
            language=("ar" if re.search(r"[\u0600-\u06FF]", task_spec.raw_prompt or "") else "en"),
        )

    def _build_budget(
        self,
        mode: str,
        provider_settings: ProviderSettings,
        overrides: dict,
        execution_plan: ExecutionPlan,
        max_results: int = 25,
    ) -> SearchBudget:
        """
        Budget scales with mode AND max_results.
        Larger max_results → more search calls and more pages to scrape.
        The plan_builder already set the query counts; budget caps match them.
        """
        mode = (task_mode := (mode or "Balanced").lower())

        if execution_plan.max_candidates_to_process >= 150:
            task_mode = "deep"

        def _q(p: str) -> int:
            return execution_plan.max_queries_per_provider.get(p, 0)

        max_cand = execution_plan.max_candidates_to_process
        base_pages = max(30, int(max_cand * 1.5))

        if task_mode == "fast":
            budget = SearchBudget(
                max_total_search_calls=10,
                max_ddg_calls=_q("ddg") if provider_settings.use_ddg else 0,
                max_exa_calls=_q("exa") if provider_settings.use_exa else 0,
                max_tavily_calls=0,
                max_serpapi_calls=0,
                max_pages_to_scrape=min(base_pages, 40),
            )
        elif task_mode == "deep":
            budget = SearchBudget(
                max_total_search_calls=60,
                max_ddg_calls=_q("ddg") if provider_settings.use_ddg else 0,
                max_exa_calls=_q("exa") if provider_settings.use_exa else 0,
                max_tavily_calls=_q("tavily") if provider_settings.use_tavily else 0,
                max_serpapi_calls=_q("serpapi") if provider_settings.use_serpapi else 0,
                max_pages_to_scrape=base_pages,
            )
        else:
            budget = SearchBudget(
                max_total_search_calls=30,
                max_ddg_calls=_q("ddg") if provider_settings.use_ddg else 0,
                max_exa_calls=_q("exa") if provider_settings.use_exa else 0,
                max_tavily_calls=_q("tavily") if provider_settings.use_tavily else 0,
                max_serpapi_calls=_q("serpapi") if provider_settings.use_serpapi else 0,
                max_pages_to_scrape=min(base_pages, 90),
            )

        for key, value in (overrides or {}).items():
            if hasattr(budget, key):
                try:
                    setattr(budget, key, int(value))
                except Exception:
                    pass

        return budget
