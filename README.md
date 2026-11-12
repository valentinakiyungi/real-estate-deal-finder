# DealScout: Real-Estate Price & Deal Finder

Find homes listed below what they're worth. An end-to-end project combining classic ML, GenAI, an agent
and a multi-page web app:

| Layer | What it does | Where |
|---|---|---|
| **ML** | Gradient-boosted house-price model on log price with feature engineering; conformalised quantile models give a calibrated 80% fair-value band | `dealfinder/model.py` |
| **GenAI** | Turns messy listing text ("3bd / 2 full + 1 half bath… 1760 sqft lot… approx 2,110 SF… cash only") into validated structured features + red flags | `dealfinder/parser.py` |
| **Agent** | Searches listings, parses and values each one, applies a risk haircut, separates clean deals from speculative ones and value traps, writes a report | `dealfinder/agent.py` |
| **Web app** | DealScout site: landing page, deal cards + map + table, price check, watchlist with compare | `app.py`, `ui/` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py                        # opens the site in your browser
```

The first launch builds the data and trains the model (~10 s).

### The site
| Page | What it does |
|---|---|
| **Home** | Hero with a live example, how it works, the biggest gaps right now, and a "cheap for a reason" example |
| **Deals** | Filter by neighborhood, price, bedrooms and verdict; view as **cards**, a **map** or a **table**; upload your own CSV; run the AI analyst; download the report or results |
| **Check a price** | Paste any listing and asking price; get our estimate, likely range, value after risks and a verdict |
| **Watchlist** | Save homes from anywhere, compare up to 4 side by side, download as CSV |
| **How it works** | Accuracy, what each label means, how risk is priced in, what drives price, limits |

Every property gets a line-drawn front elevation generated from its data (width from bedrooms, a second
storey for larger homes, garage, pool, a chimney on older homes), and a **price gap bar**: our likely range
as a band, the asking price as a dot. Design: Archivo type, survey-drawing palette, defined in
`ui/theme.py` and `.streamlit/config.toml`. To rename the product, change `BRAND` in `ui/theme.py`.

### Turning on AI (free Gemini key)
1. Get a free key at <https://aistudio.google.com/apikey>.
2. Copy `.env.example` to `.env` and paste the key: `GEMINI_API_KEY=...`
3. Restart the app. "Run the AI analyst" appears under "Change listings or run the AI analyst" on the
   Deals page. The quick scan stays the default, because it's instant.

`.env` is in `.gitignore`, so the key never reaches GitHub. With AI on, Gemini (1) parses descriptions,
15 per request to stay inside free-tier rate limits, through a forced function call whose schema is the
Pydantic `ListingFeatures` model, cross-checked against the rule extractor (disagreements lower confidence),
and (2) drives the agent loop: choosing searches, inspecting candidates and writing the narrative. Rate-limit
errors are retried with back-off; if the AI still fails, the app falls back to the offline planner and says so.
`ANTHROPIC_API_KEY` works too; `DEALFINDER_PROVIDER=gemini|anthropic` forces one. The API client is
standard-library only (`dealfinder/llm.py`), no SDK required.

### Put it online (free, Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app**, then
   **Deploy a public app from GitHub**.
3. Repository `your-name/real-estate-deal-finder`, branch `main`, main file `app.py`. Pick a URL such as
   `dealscout.streamlit.app`.
4. Open **Advanced settings**: choose Python 3.12, and in **Secrets** paste
   `GEMINI_API_KEY = "your-key"` (with the quotes). Leave Secrets empty for a fully offline site.
5. Click **Deploy**. The first start takes a few minutes while packages install and the model trains.

Visitors share your free Gemini quota, so each visitor can run the AI analyst up to 3 times per visit
(`AI_RUNS_PER_VISIT` in `ui/state.py`). Watchlists live in each visitor's browser session.
`.streamlit/secrets.toml` is in `.gitignore` in case you test secrets locally.

### Command line
```bash
python main.py train          # regenerate data + retrain
python main.py find           # agent -> reports/deal_report.{md,html}
python main.py find --neighborhood "Cedar Hills" --max-price 400000
python main.py find --csv my_listings.csv
python main.py parse "3 bed / 1 full + 1 half bath, approx 1,480 SF, circa 1962, needs TLC, cash only"
python main.py evaluate       # accuracy against hidden ground truth
pytest -q                     # 18 tests, including simulated Gemini round-trips
```

## How a listing is judged
1. **Parse** description → beds, baths, living sqft (not the lot!), lot, year, condition, garage, pool, view, red flags.
2. **Value** with the GBM: point estimate + 80% band.
3. **Risk haircut** per red flag (foundation 12%, cash-only 8%, flood 7%, tenant 5%, as-is 4%, unpermitted 4%, roof 3%; capped at 35%).
4. **Verdict**
   - *strong deal* — no red flags and ≥15% below fair value (or below the band and ≥10%)
   - *deal* — no red flags and ≥10% below fair value
   - *speculative* — has red flags but still ≥10% below the risk-adjusted value
   - *value trap* — looks cheap, but the risks explain the discount
   - *fair* / *overpriced*

   Confidence drops when the AI and rule parsers disagree, when the 80% band is wide, or when a listing's
   neighbourhood isn't in the training data (its location is then valued at the market average).

## Results (synthetic market, seed-fixed)
- Price model, 1,200 held-out sales: **MAPE 6.2%** (neighbourhood $/sqft baseline: 14.9%), R² (log) 0.96,
  80% band coverage 82%. The data has ~5.6% irreducible noise, so the model is close to the floor.
- Parser vs ground truth on 60 listings: beds/baths/year/condition/garage/pool/view 100%; sqft and lot exact
  except where the text itself rounds ("3.2k sq ft", "0.19 acres").
- Fair value from *parsed text*: 5.4% MAPE.
- Deal flags: 80% recall, 57% precision on clean deals; two of three false positives are near-misses
  (true discounts of 7% and 9.6% vs a 10% cut-off). No red-flag listing is ever ranked as a clean deal.

## Using real data
`dealfinder/data.py` generates a realistic synthetic market because listing APIs need keys and have
restrictive terms. To use real data, provide:
- training sales with the columns in `model.RAW_COLS` + `sale_price`
- active listings with `listing_id, address, neighborhood, asking_price, days_on_market, dist_center_km,
  school_rating, description`

Estimates only — not financial advice.
