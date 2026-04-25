from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = DATA_DIR / "exports"
OUTPUT_DIR = EXPORT_DIR  # backward-compatible alias
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
ONTOLOGY_DIR = DATA_DIR / "ontology"
CACHE_DB = DATA_DIR / "cache.db"

for _d in [DATA_DIR, EXPORT_DIR, UPLOAD_DIR, CACHE_DIR, ONTOLOGY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# API keys
# -----------------------------------------------------------------------------
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# -----------------------------------------------------------------------------
# Scraping / retrieval settings
# -----------------------------------------------------------------------------
DEFAULT_TIMEOUT: int = 20
DEFAULT_MAX_INTERNAL_PAGES: int = 3
DEFAULT_TAVILY_SEARCH_DEPTH: str = os.getenv("DEFAULT_TAVILY_SEARCH_DEPTH", "basic")
DEFAULT_RESULT_LANGUAGE: str = os.getenv("DEFAULT_RESULT_LANGUAGE", "auto")

# -----------------------------------------------------------------------------
# Domain rules
# -----------------------------------------------------------------------------
EXCLUDED_DOMAINS: set[str] = {
    "facebook.com", "twitter.com", "x.com", "youtube.com", "instagram.com",
    "pinterest.com", "reddit.com", "tiktok.com", "vimeo.com", "dailymotion.com",
    "wikipedia.org", "wikimedia.org", "wikidata.org", "britannica.com",
    "amazon.com", "ebay.com", "alibaba.com", "aliexpress.com",
    "indeed.com", "glassdoor.com", "monster.com", "ziprecruiter.com",
    "crunchbase.com", "pitchbook.com", "zoominfo.com", "dnb.com", "manta.com",
    "owler.com", "bloomberg.com", "reuters.com",
    "forbes.com", "businessinsider.com", "techcrunch.com", "wired.com",
    "cnbc.com", "wsj.com", "ft.com", "economist.com",
    "medium.com", "substack.com", "quora.com", "stackoverflow.com",
    "github.com", "gitlab.com", "bitbucket.org",
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    "chatgpt.com", "openai.com", "perplexity.ai", "bard.google.com",
    "apple.com", "microsoft.com", "about.youtube",
    "yelp.com", "tripadvisor.com", "trustpilot.com", "clutch.co", "g2.com", "capterra.com",
    "f6s.com", "tracxn.com", "goodfirms.co", "ensun.io", "sortlist.com", "expertise.com",
    "designrush.com", "topdevelopers.co", "techreviewer.co", "themanifest.com", "gartner.com",
    "getlatka.com", "oilandgasmiddleeast.com", "rigzone.com", "offshore-technology.com",
    "energymonitor.ai", "oilandgaspeople.com", "lastingdynamics.com", "fingerlakes1.com",
    "kejoragasbumi.com", "thedigitalprojectmanager.com", "crunch-is.com",
    "researchandmarkets.com", "mordorintelligence.com", "skyquestt.com",
    "fortunebusinessinsights.com", "polarismarketresearch.com", "coherentmarketinsights.com",
    "intelmarketresearch.com", "lusha.com", "apollo.io", "hunter.io", "rocketreach.com",
    "contactout.com", "clearbit.com", "seamless.ai", "airbnb.com", "airbnb.co.uk",
    "lacentrale.fr", "soulac-sur-mer", "francethisway.com", "beachatlas.com",
}

ACADEMIC_DOMAINS: set[str] = {
    "researchgate.net", "academia.edu", "semanticscholar.org", "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "scholar.google.com",
    "sciencedirect.com", "ieee.org", "onepetro.org", "mdpi.com", "springer.com",
    "springerlink.com", "wiley.com", "elsevier.com", "frontiersin.org",
    "tandfonline.com", "sciopen.com", "e3s-conferences.org", "ui.adsabs.harvard.edu",
    "ajol.info", "eric.ed.gov", "repository.uobaghdad.edu.iq", "ijcpe.uobaghdad.edu.iq",
    "hub.hku.hk", "cambridge.org", "osti.gov", "scirp.org", "hesp.umd.edu", "icaiit.org",
}

KNOWN_USA_DOMAINS: set[str] = {
    "enverus.com", "quorumsoftware.com", "wenergysoftware.com", "aucerna.com",
    "ogsys.com", "wellware.com", "progscheduler.com", "bsee.gov", "eia.gov",
    "energy.gov", "eaginc.com", "pakenergy.com", "enertia-software.com", "gptsoft.com",
    "pcs-inc.com", "ogssi.com", "dataparc.com", "osisoft.com", "aspentech.com",
    "intelex.com", "cenovus.com", "cognizant.com", "ibm.com", "oracle.com",
    "sap.com", "accenture.com", "deloitte.com", "slb.com", "halliburton.com",
    "bakerhughes.com", "weatherford.com", "emerson.com", "honeywell.com", "ge.com",
    "abb.com", "championx.com", "corteva.com", "callmc.com", "softserveinc.com",
}

KNOWN_EGYPT_DOMAINS: set[str] = {
    "petrojet.com.eg", "petrogas.com.eg", "gupco.com.eg", "petromaint.com.eg", "suco.com.eg",
}

DIRECTORY_TITLE_PATTERNS = [
    r"^top\s+\d+", r"^best\s+\d+", r"^\d+\s+best", r"^\d+\s+leading", r"^\d+\s+top",
    r"best .+ companies", r"best .+ software", r"best .+ development", r"top .+ companies",
    r"top .+ software", r"top .+ solutions", r"\d+ best", r"list of .+", r"guide to .+",
    r"comparison of .+", r"review of .+", r"directory of .+", r"companies in .+",
    r"vendors in .+", r"providers in .+", r"services in .+", r"exhibitors", r"sponsors",
    r"ranked", r"ranking", r"market map", r"market landscape", r"who are the leading",
]
