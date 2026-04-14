# 🔍 AI Research Agent

Find companies, research papers, and LinkedIn profiles using plain English — powered by free APIs.

## What it does

- **Companies**: Find vendors, contractors, or any business entity globally with email, phone, LinkedIn
- **Research papers**: Find academic papers with authors, DOIs, abstracts — export to PDF
- **LinkedIn people**: Find engineers, managers, and HR professionals by job title and location

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env
# Edit .env with your keys (Groq is free and enough to start)

# 3. Run the developer interface
streamlit run app.py

# 4. Or run the user-friendly interface
streamlit run app_user.py
```

## Two interfaces

| File | For whom | Features |
|---|---|---|
| `app.py` | Developers | All tabs, budget controls, raw logs, provider tuning |
| `app_user.py` | End users | Clean 3-tab UI, step-by-step key setup, download buttons |

## Free API keys (no credit card)

| Key | Where to get | Free quota |
|---|---|---|
| Groq | console.groq.com | 14,400 req/day |
| Gemini | aistudio.google.com | 1,500 req/day |
| Exa | exa.ai | 1,000 searches/month |
| Tavily | tavily.com | 1,000 searches/month |
| SerpApi | serpapi.com | 100 searches/month |

You only need **one key** (Groq) to start. More keys = more results.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file: `app_user.py`
4. In Settings → Secrets, paste your keys from `.env.example`

## Example prompts

```
Find oilfield service companies in Egypt and Saudi Arabia with email and phone
Find digital oil and gas software companies outside USA and Egypt with email
Find research papers about ESP electrical submersible pump with authors export as PDF
Find LinkedIn profiles of petroleum engineers in oil gas companies in Egypt
Find renewable energy companies in Germany and Norway with contact details
```

## Project structure

```
ai_research_agent/
├── app.py              ← Developer interface
├── app_user.py         ← User-facing deployable interface
├── requirements.txt
├── .env.example
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── core/               ← Business logic
│   ├── config.py       ← API keys, excluded domains
│   ├── models.py       ← Data models
│   ├── task_parser.py  ← Prompt → structured task
│   ├── plan_builder.py ← Search strategy
│   ├── geography.py    ← 200+ countries, US states, cities
│   ├── people_search.py ← LinkedIn X-ray search
│   ├── keyword_expander.py ← Search expansion
│   ├── feynman_bridge.py   ← Deep research integration
│   └── ...
├── providers/          ← Search APIs
│   ├── ddg_provider.py
│   ├── exa_provider.py
│   ├── tavily_provider.py
│   ├── serpapi_provider.py
│   └── website_scraper.py
└── pipeline/
    └── orchestrator.py ← Main search pipeline
```

## License

MIT
