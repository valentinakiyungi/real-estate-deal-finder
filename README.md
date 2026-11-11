# Real-Estate Price & Deal Finder

An end-to-end project combining classic ML, GenAI and an agent:

| Layer | What it does | Where |
|---|---|---|
| **ML** | Gradient-boosted house-price model on log price with feature engineering; conformalised quantile models give a calibrated 80% fair-value band | `dealfinder/model.py` |
| **GenAI** | Turns messy listing text ("3bd / 2 full + 1 half bath… 1760 sqft lot… approx 2,110 SF… cash only") into validated structured features + red flags | `dealfinder/parser.py` |
| **Agent** | Searches listings, parses and values each one, applies a risk haircut, separates clean deals from speculative ones and value traps, writes a report | `dealfinder/agent.py` |

## Quick start

```bash
pip install scikit-learn pandas numpy pydantic joblib pytest
python main.py train          # generate data + train model (~10 s)
python main.py find           # run the agent -> reports/deal_report.{md,html}
python main.py find --neighborhood "Cedar Hills" --max-price 400000
python main.py parse "3 bed / 1 full + 1 half bath, approx 1,480 SF on a 0.21 acre lot, circa 1962, needs TLC, cash only"
python main.py evaluate       # accuracy against hidden ground truth
pytest -q
```

### Turning on Claude
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DEALFINDER_MODEL=claude-sonnet-5-5   # optional
python main.py find
```
With a key, Claude (1) parses each description through a forced tool call whose schema is the Pydantic
`ListingFeatures` model, cross-checked against the rule extractor (disagreements lower confidence), and
(2) drives the agent loop: choosing searches, inspecting candidates and writing the narrative. Without a key
the same tools run under a deterministic planner, so everything works offline. `--offline` forces that mode.
The API client is stdlib-only (`dealfinder/llm.py`), no SDK required.

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
