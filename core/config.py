from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DB   = DATA_DIR / "cache.db"

for _d in [DATA_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API keys from .env ─────────────────────────────────────────────────────
EXA_API_KEY:        str = os.getenv("EXA_API_KEY",        "")
TAVILY_API_KEY:     str = os.getenv("TAVILY_API_KEY",     "")
SERPAPI_KEY:        str = os.getenv("SERPAPI_KEY",        "")
FIRECRAWL_API_KEY:  str = os.getenv("FIRECRAWL_API_KEY",  "")
OPENAI_API_KEY:     str = os.getenv("OPENAI_API_KEY",     "")
ANTHROPIC_API_KEY:  str = os.getenv("ANTHROPIC_API_KEY",  "")
GROQ_API_KEY:       str = os.getenv("GROQ_API_KEY",       "")
GEMINI_API_KEY:     str = os.getenv("GEMINI_API_KEY",     "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_BASE_URL:    str = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_MODEL:       str = os.getenv("OLLAMA_MODEL",       "llama3")

# ── Scraping settings ──────────────────────────────────────────────────────
DEFAULT_TIMEOUT:           int = 20   # seconds per HTTP request
DEFAULT_MAX_INTERNAL_PAGES: int = 3   # max internal pages to follow per site

# ── Domains to skip entirely ───────────────────────────────────────────────
EXCLUDED_DOMAINS: set[str] = {
    # Social media (linkedin.com is ALLOWED for people_search — handled per task type)
    "facebook.com", "twitter.com", "x.com",
    "youtube.com", "instagram.com", "pinterest.com", "reddit.com",
    "tiktok.com", "vimeo.com", "dailymotion.com",
    # Knowledge / encyclopedias
    "wikipedia.org", "wikimedia.org", "wikidata.org", "britannica.com",
    # E-commerce
    "amazon.com", "ebay.com", "alibaba.com", "aliexpress.com",
    # Job boards
    "indeed.com", "glassdoor.com", "monster.com", "ziprecruiter.com",
    # Paid company databases / aggregators
    "crunchbase.com", "pitchbook.com", "zoominfo.com", "dnb.com", "manta.com",
    "owler.com", "bloomberg.com", "reuters.com",
    # General news / media
    "forbes.com", "businessinsider.com", "techcrunch.com", "wired.com",
    "cnbc.com", "wsj.com", "ft.com", "economist.com",
    # Blogging / Q&A
    "medium.com", "substack.com", "quora.com", "stackoverflow.com",
    # Code repos
    "github.com", "gitlab.com", "bitbucket.org",
    # Search engines / AI tools
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    "chatgpt.com", "openai.com", "perplexity.ai", "bard.google.com",
    # Tech giants (not domain-specific vendors)
    "apple.com", "microsoft.com", "about.youtube",
    # Reviews / ratings
    "yelp.com", "tripadvisor.com", "trustpilot.com",
    "clutch.co", "g2.com", "capterra.com",
    # Startup/company directories (list ABOUT companies, not company sites)
    "f6s.com", "tracxn.com", "goodfirms.co", "ensun.io",
    "sortlist.com", "expertise.com", "designrush.com",
    "topdevelopers.co", "techreviewer.co", "themanifest.com",
    "gartner.com", "getlatka.com",
    # Oil & gas media/news (lists, not company sites)
    "oilandgasmiddleeast.com", "rigzone.com", "offshore-technology.com",
    "energymonitor.ai", "oilandgaspeople.com",
    # Blogs and comparison sites
    "lastingdynamics.com", "fingerlakes1.com", "kejoragasbumi.com",
    "thedigitalprojectmanager.com", "crunch-is.com",
    # Market research
    "researchandmarkets.com", "mordorintelligence.com", "skyquestt.com",
    "fortunebusinessinsights.com", "polarismarketresearch.com",
    "coherentmarketinsights.com", "intelmarketresearch.com",
    # People/company data aggregators (not company sites)
    "lusha.com", "apollo.io", "hunter.io", "rocketreach.com",
    "contactout.com", "clearbit.com", "seamless.ai",
    # Irrelevant
    "airbnb.com", "airbnb.co.uk", "lacentrale.fr",
    "soulac-sur-mer", "francethisway.com", "beachatlas.com",
}

# Academic/research domains — explicitly ALLOWED
ACADEMIC_DOMAINS: set[str] = {
    "researchgate.net", "academia.edu", "semanticscholar.org",
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "scholar.google.com", "sciencedirect.com", "ieee.org",
    "onepetro.org", "mdpi.com", "springer.com", "springerlink.com",
    "wiley.com", "elsevier.com", "frontiersin.org",
    "tandfonline.com", "sciopen.com", "e3s-conferences.org",
    "ui.adsabs.harvard.edu", "ajol.info", "eric.ed.gov",
    "repository.uobaghdad.edu.iq", "ijcpe.uobaghdad.edu.iq",
    "hub.hku.hk", "cambridge.org", "osti.gov", "scirp.org",
    "hesp.umd.edu", "icaiit.org",
}

# Well-known US-headquartered company domains
# These are used when geo-exclusion for USA is active and HQ can't be scraped
KNOWN_USA_DOMAINS: set[str] = {
    # Oil & Gas / Energy tech
    "enverus.com", "quorumsoftware.com", "wenergysoftware.com",
    "aucerna.com", "ogsys.com", "wellware.com", "progscheduler.com",
    "bsee.gov", "eia.gov", "energy.gov",
    "eaginc.com", "pakenergy.com", "enertia-software.com",
    "gptsoft.com", "pcs-inc.com", "ogssi.com",
    "dataparc.com", "osisoft.com", "aspentech.com",
    "intelex.com", "cenovus.com",
    # Large IT / Consulting (USA HQ)
    "cognizant.com", "ibm.com", "oracle.com", "sap.com",
    "accenture.com", "deloitte.com", "slb.com", "halliburton.com",
    "bakerhughes.com", "weatherford.com",
    # Industrial automation (USA HQ)
    "emerson.com", "honeywell.com", "ge.com", "abb.com",
    "championx.com", "corteva.com",
    # Misc
    "callmc.com", "softserveinc.com",
}

# Well-known Egypt-headquartered company domains
KNOWN_EGYPT_DOMAINS: set[str] = {
    "petrojet.com.eg", "petrogas.com.eg", "gupco.com.eg",
    "petromaint.com.eg", "suco.com.eg",
}

# Title patterns that signal a blog/list/directory — not a company home page
DIRECTORY_TITLE_PATTERNS = [
    r"^top\s+\d+",                         # "Top 10 Companies..."
    r"^best\s+\d+",                         # "Best 7 Software..."
    r"^\d+\s+best",                         # "7 Best Companies..."
    r"^\d+\s+leading",                      # "7 Leading Providers..."
    r"^\d+\s+top",                          # "10 Top Vendors..."
    r"best .+ companies",                    # "Best Oil and Gas Companies"
    r"best .+ software",                     # "Best Oil Gas Software..."
    r"best .+ development",                  # "Best Software Development Services"
    r"top .+ companies",                     # "Top Software Companies"
    r"top .+ software",                      # "Top Software Solutions"
    r"top .+ solutions",                     # "Top Software Solutions of 2026"
    r"\d+ best",                             # "10 Best..."
    r"list of .+",                           # "List of Companies..."
    r"guide to .+",                          # "Guide to Software..."
    r"comparison of .+",                     # "Comparison of..."
    r"review of .+",                         # "Review of..."
    r"shaping the market",                   # market analysis articles
    r"companies to watch",                   # industry articles
    r"vendors? to know",                     # vendor lists
    r"providers? to know",                   # provider lists
    r"briefing news",                        # "China Briefing News"
    r"the \d+ best",                         # "The 10 Best..."
    r"software shortlist",                   # shortlists
    r"market report",                        # market reports
    r"market size",                          # market size articles
]

