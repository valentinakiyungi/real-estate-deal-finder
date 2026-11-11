"""Deal-finder agent.

The agent works through a small set of tools:

    search_listings    filter the active-listing feed
    analyze_listings   GenAI-parse descriptions -> ML fair value band -> risk adjustment
    get_listing        read one raw listing (to sanity-check a candidate)
    submit_report      final answer: ranked deals + market notes -> Markdown/HTML report

With ANTHROPIC_API_KEY set, Claude drives the loop (decides what to search,
which candidates to inspect, and writes the narrative). Without a key, a
deterministic planner calls the same tools in a fixed order, so the project
runs fully offline.
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import llm
from .model import PriceModel
from .parser import parse

# Rough cost of each risk as a share of value (what a buyer should haircut).
RISK_HAIRCUT = {
    "foundation": 0.12, "cash only": 0.08, "unfinanceable": 0.08, "flood": 0.07, "roof": 0.03,
    "tenant": 0.05, "as-is": 0.04, "estate": 0.02, "unpermitted": 0.04,
}


def _haircut(flags: list[str]) -> float:
    hit = {k for f in flags for k in RISK_HAIRCUT if k in f.lower()}
    hit -= {"unfinanceable"} if "cash only" in hit else set()
    return min(0.35, sum(RISK_HAIRCUT[k] for k in hit))


class DealTools:
    def __init__(self, listings: pd.DataFrame, model: PriceModel, use_llm_parser: bool | None = None):
        self.listings = listings.set_index("listing_id", drop=False)
        self.model = model
        self.use_llm_parser = use_llm_parser
        self.cache: dict[str, dict] = {}
        self.log: list[str] = []
        self.report: dict | None = None

    # ---- tool implementations -------------------------------------------
    def search_listings(self, neighborhood: str | None = None, max_price: int | None = None,
                        min_price: int | None = None) -> list[dict]:
        d = self.listings
        if neighborhood:
            d = d[d["neighborhood"].str.lower() == neighborhood.lower()]
        if max_price:
            d = d[d["asking_price"] <= max_price]
        if min_price:
            d = d[d["asking_price"] >= min_price]
        self.log.append(f"search_listings(neighborhood={neighborhood}, max_price={max_price}) -> {len(d)} hits")
        return [{"id": r.listing_id, "address": r.address, "neighborhood": r.neighborhood,
                 "asking_price": int(r.asking_price), "days_on_market": int(r.days_on_market)} for r in d.itertuples()]

    def get_listing(self, listing_id: str) -> dict:
        r = self.listings.loc[listing_id]
        self.log.append(f"get_listing({listing_id})")
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if not str(k).startswith("_truth")}

    def analyze_listings(self, ids: list[str]) -> list[dict]:
        out = []
        for i in ids:
            if i not in self.cache:
                self.cache[i] = self._analyze_one(i)
            out.append(self.cache[i])
        self.log.append(f"analyze_listings({len(ids)} ids)")
        return out

    def _analyze_one(self, i: str) -> dict:
        r = self.listings.loc[i]
        feats, meta = parse(r["description"], self.use_llm_parser)
        f = feats.model_dump()
        missing = [k for k in ("beds", "baths", "sqft", "year_built") if f[k] is None]
        res = {"id": i, "address": r["address"], "neighborhood": r["neighborhood"],
               "asking_price": int(r["asking_price"]), "days_on_market": int(r["days_on_market"]),
               "features": f, "parser": meta["source"], "parser_disagreements": meta["disagreements"]}
        if missing:
            res.update(verdict="insufficient data", missing=missing)
            return res
        row = {**f, "neighborhood": r["neighborhood"], "dist_center_km": r["dist_center_km"],
               "school_rating": r["school_rating"], "lot_sqft": f["lot_sqft"] or 5000}
        est = self.model.predict(pd.DataFrame([row])).iloc[0]
        hc = _haircut(f["red_flags"])
        risk_adj = est.fair_value * (1 - hc)
        ask = r["asking_price"]
        discount = 1 - ask / est.fair_value
        adj_discount = 1 - ask / risk_adj
        if hc == 0 and (discount >= 0.15 or (ask < est.fair_low and discount >= 0.10)):
            verdict = "strong deal"
        elif hc == 0 and discount >= 0.10:
            verdict = "deal"
        elif hc > 0 and adj_discount >= 0.10:
            verdict = "speculative"  # cheap even after pricing in the risks, but needs due diligence
        elif hc > 0 and discount >= 0.08:
            verdict = "value trap"  # looks cheap, but the risks explain the discount
        elif discount <= -0.10:
            verdict = "overpriced"
        else:
            verdict = "fair"
        confidence = "high" if not meta["disagreements"] and est.fair_high / est.fair_low < 1.35 else "medium"
        if len(meta["disagreements"]) >= 2:
            confidence = "low"
        res.update(fair_value=int(est.fair_value), fair_low=int(est.fair_low), fair_high=int(est.fair_high),
                   risk_haircut=round(hc, 3), risk_adjusted_value=int(round(risk_adj, -3)),
                   discount_vs_fair=round(float(discount), 3), discount_vs_risk_adjusted=round(float(adj_discount), 3),
                   price_per_sqft=round(ask / f["sqft"]), verdict=verdict, confidence=confidence)
        return res

    def submit_report(self, deals: list[dict], market_summary: str) -> str:
        self.report = {"deals": deals, "market_summary": market_summary}
        self.log.append(f"submit_report({len(deals)} deals)")
        return "report accepted"


# ---------------------------------------------------------------- schemas ---
TOOL_SPECS = [
    {"name": "search_listings", "description": "Search active listings. All filters optional.",
     "input_schema": {"type": "object", "properties": {
         "neighborhood": {"type": "string"}, "max_price": {"type": "integer"}, "min_price": {"type": "integer"}}}},
    {"name": "analyze_listings",
     "description": ("Parse each listing's description with the GenAI parser, estimate fair value with the ML model "
                     "(point + calibrated 80% band), apply a risk haircut for red flags, and return a verdict. "
                     "Accepts up to 80 ids per call."),
     "input_schema": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                      "required": ["ids"]}},
    {"name": "get_listing", "description": "Read one listing's raw record including full description.",
     "input_schema": {"type": "object", "properties": {"listing_id": {"type": "string"}}, "required": ["listing_id"]}},
    {"name": "submit_report", "description": "Finish: submit the ranked deals and a short market summary.",
     "input_schema": {"type": "object", "required": ["deals", "market_summary"], "properties": {
         "market_summary": {"type": "string", "description": "3-5 sentences on what the search found."},
         "deals": {"type": "array", "items": {"type": "object", "required": ["id", "headline", "rationale"],
                   "properties": {"id": {"type": "string"}, "headline": {"type": "string"},
                                  "rationale": {"type": "string", "description": "2-4 sentences, cite numbers"},
                                  "next_steps": {"type": "string"}}}}}}},
]

SYSTEM = """You are a careful real-estate deal analyst. Goal: find genuinely under-priced active listings.
Method: search the feed (respect the buyer's criteria), analyze every candidate, then look closely at the best ones
with get_listing before deciding. Rank only listings whose verdict is
"strong deal" or "deal" (clean, >=10% below fair value). "speculative" listings (red flags, but cheap even after a
risk haircut) and "value trap" listings (the risks explain the discount) are listed separately in the report
automatically; mention the most notable ones in the market summary but don't rank them. Low parser confidence
means verify. Rank deals best-first, then call submit_report. Be concrete and quantitative; no hype."""


# ------------------------------------------------------------------ runner ---
def run_llm_agent(tools: DealTools, goal: str, max_turns: int = 15) -> None:
    msgs = [{"role": "user", "content": goal}]
    for _ in range(max_turns):
        resp = llm.messages(msgs, system=SYSTEM, tools=TOOL_SPECS, max_tokens=4000)
        msgs.append({"role": "assistant", "content": resp["content"]})
        calls = [b for b in resp["content"] if b["type"] == "tool_use"]
        if not calls:
            break
        results = []
        for c in calls:
            try:
                out = getattr(tools, c["name"])(**c["input"])
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": json.dumps(out, default=str)})
        msgs.append({"role": "user", "content": results})
        if tools.report:
            return
    if not tools.report:  # model never submitted -> fall back so the user still gets a report
        run_rule_agent(tools, goal_filters={})


def run_rule_agent(tools: DealTools, goal_filters: dict) -> None:
    hits = tools.search_listings(**goal_filters)
    analyzed = tools.analyze_listings([h["id"] for h in hits])
    good = [a for a in analyzed if a["verdict"] in ("strong deal", "deal")]
    good.sort(key=lambda a: (a["confidence"] == "low", a["verdict"] != "strong deal", -a["discount_vs_fair"]))
    deals = []
    for a in good:
        tools.get_listing(a["id"])
        f = a["features"]
        risk = (f"Red flags ({', '.join(f['red_flags'])}) are priced in with a {a['risk_haircut']:.0%} haircut. "
                if f["red_flags"] else "No red flags in the description. ")
        deals.append({
            "id": a["id"],
            "headline": f"{f['beds']}bd/{f['baths']:g}ba, {f['sqft']:,} sqft in {a['neighborhood']} "
                        f"— {a['discount_vs_fair']:.0%} below fair value",
            "rationale": (f"Asking ${a['asking_price']:,} vs model fair value ${a['fair_value']:,} "
                          f"(80% band ${a['fair_low']:,}–${a['fair_high']:,}). "
                          + ("Asking is below the bottom of the band. " if a["asking_price"] < a["fair_low"] else "")
                          + risk + f"On market {a['days_on_market']} days."),
            "next_steps": "Order an inspection and pull 3 recent comps within 1 km before offering.",
        })
    traps = [a for a in analyzed if a["verdict"] == "value trap"]
    spec = [a for a in analyzed if a["verdict"] == "speculative"]
    n_over = sum(a["verdict"] == "overpriced" for a in analyzed)
    common = pd.Series([fl for a in spec + traps for fl in a["features"]["red_flags"]]).value_counts().index[:3]
    summary = (f"Scanned {len(analyzed)} listings. {len(deals)} clean listings are priced at least 10% below the "
               f"model's fair value with no disclosed red flags. {len(spec)} more are cheap even after a risk haircut "
               f"but carry problems such as {', '.join(common) or 'n/a'} — treat these as speculative. "
               f"{len(traps)} are value traps whose discount is explained by their risks, and "
               f"{n_over} are priced 10%+ above fair value.")
    tools.submit_report(deals, summary)


# ------------------------------------------------------------------ report ---
def render_report(tools: DealTools, out_dir: Path, title: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = tools.report
    all_a = list(tools.cache.values())
    by_id = {a["id"]: a for a in all_a}
    traps = sorted([a for a in all_a if a.get("verdict") == "value trap"], key=lambda a: -a["discount_vs_fair"])
    spec = sorted([a for a in all_a if a.get("verdict") == "speculative"], key=lambda a: -a["discount_vs_risk_adjusted"])
    m = tools.model.metrics
    spec_line = lambda s: (f"{s['id']} {s['address']}, {s['neighborhood']} — asking ${s['asking_price']:,}, "
                           f"fair ${s['fair_value']:,}, {s['discount_vs_risk_adjusted']:.0%} below after a "
                           f"{s['risk_haircut']:.0%} haircut for: {', '.join(s['features']['red_flags'])}")

    md = [f"# {title}", f"_Generated {date.today():%d %b %Y}_", "", "## Summary", rep["market_summary"], "",
          "## Ranked deals", "",
          "| # | Listing | Asking | Fair value (80% band) | Discount | Confidence |", "|---|---|---|---|---|---|"]
    for k, d in enumerate(rep["deals"], 1):
        a = by_id[d["id"]]
        md.append(f"| {k} | {d['id']} · {a['address']}, {a['neighborhood']} | ${a['asking_price']:,} | "
                  f"${a['fair_value']:,} (${a['fair_low']:,}–${a['fair_high']:,}) | "
                  f"{a['discount_vs_risk_adjusted']:.0%} | {a['confidence']} |")
    for k, d in enumerate(rep["deals"], 1):
        md += ["", f"### {k}. {d['headline']}", d["rationale"], "", f"**Next steps:** {d.get('next_steps', '')}"]
    md += ["", "## Speculative (cheap after risk haircut, needs due diligence)", ""]
    md += [f"- {spec_line(s)}" for s in spec] or ["- none"]
    md += ["", "## Value traps (cheap for a reason)", ""]
    md += [f"- **{t['id']}** {t['address']} — asking ${t['asking_price']:,}, {t['discount_vs_fair']:.0%} under "
           f"fair value, but: {', '.join(t['features']['red_flags'])}" for t in traps] or ["- none"]
    md += ["", "## Model card",
           f"Gradient-boosted trees on log price, {len(m) and m['test_rows']} held-out sales: MAPE {m['MAPE']:.1%} "
           f"(baseline $/sqft {m['baseline_ppsf_MAPE']:.1%}), {m['within_10pct']:.0%} within 10%, "
           f"80% band coverage {m['interval_80_coverage']:.0%}.", "",
           "_Estimates only — not financial advice. Verify with an inspection and local comps._"]
    md_path = out_dir / "deal_report.md"
    md_path.write_text("\n".join(md))

    # HTML version
    e = html.escape
    rows = "".join(
        f"<tr><td>{k}</td><td><b>{e(d['id'])}</b><br><span class=m>{e(by_id[d['id']]['address'])}, "
        f"{e(by_id[d['id']]['neighborhood'])}</span></td><td>${by_id[d['id']]['asking_price']:,}</td>"
        f"<td>{_bar(by_id[d['id']])}</td><td class=g>{by_id[d['id']]['discount_vs_risk_adjusted']:.0%}</td>"
        f"<td>{by_id[d['id']]['confidence']}</td></tr>" for k, d in enumerate(rep["deals"], 1))
    cards = "".join(f"<section class=card><h3>{k}. {e(d['headline'])}</h3><p>{e(d['rationale'])}</p>"
                    f"<p class=m><b>Next steps:</b> {e(d.get('next_steps', ''))}</p></section>"
                    for k, d in enumerate(rep["deals"], 1))
    trap_li = "".join(f"<li><b>{e(t['id'])}</b> {e(t['address'])} — ${t['asking_price']:,}, "
                      f"{t['discount_vs_fair']:.0%} under fair value, but <i>{e(', '.join(t['features']['red_flags']))}</i></li>"
                      for t in traps) or "<li>none</li>"
    spec_li = "".join(f"<li>{e(spec_line(s))}</li>" for s in spec) or "<li>none</li>"
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Deal Report</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1f22;--m:#6b6f76;--line:#e4e1da;--acc:#1f6f5c;--band:#d9e7e2;--ask:#c2410c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181b;--fg:#e8e6e1;--m:#9aa0a6;--line:#2c2f34;--acc:#5fc4a8;--band:#24433b;--ask:#fb923c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:28px;margin:0}} h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.m{{color:var(--m);font-size:13px}} table{{width:100%;border-collapse:collapse}} td,th{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--m)}} .g{{color:var(--acc);font-weight:700}}
.card{{border:1px solid var(--line);border-radius:10px;padding:4px 16px;margin:12px 0}} .wrap{{overflow-x:auto}}
.kpi{{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}} .kpi div{{font-size:13px;color:var(--m)}} .kpi b{{display:block;font-size:22px;color:var(--fg)}}
</style></head><body>
<h1>{e(title)}</h1><p class=m>Generated {date.today():%d %b %Y} · {len(all_a)} listings analysed · agent log: {len(tools.log)} tool calls</p>
<div class=kpi><div><b>{len(rep['deals'])}</b>clean deals</div><div><b>{len(spec)}</b>speculative</div><div><b>{len(traps)}</b>value traps</div>
<div><b>{m['MAPE']:.1%}</b>model MAPE</div><div><b>{m['interval_80_coverage']:.0%}</b>80% band coverage</div></div>
<p>{e(rep['market_summary'])}</p>
<h2>Ranked deals</h2><div class=wrap><table><tr><th>#</th><th>Listing</th><th>Asking</th><th>Asking vs fair-value band</th><th>Risk-adj. discount</th><th>Conf.</th></tr>{rows}</table></div>
{cards}<h2>Speculative — cheap after risk haircut, needs due diligence</h2><ul>{spec_li}</ul>
<h2>Value traps — cheap for a reason</h2><ul>{trap_li}</ul>
<h2>Model card</h2><p class=m>Gradient-boosted trees on log price with engineered features (effective age, condition×age,
out-of-fold neighbourhood $/sqft encoding, size ratios). Conformalised quantile models give the 80% band.
Held-out MAPE {m['MAPE']:.1%} vs {m['baseline_ppsf_MAPE']:.1%} for a neighbourhood $/sqft baseline.
Estimates only — not financial advice.</p></body></html>"""
    html_path = out_dir / "deal_report.html"
    html_path.write_text(page)
    (out_dir / "agent_log.json").write_text(json.dumps({"log": tools.log, "analyses": all_a}, indent=1, default=str))
    return md_path, html_path


def _bar(a: dict) -> str:
    lo, hi = a["fair_low"] * 0.85, a["fair_high"] * 1.05
    x = lambda v: 4 + 152 * (min(max(v, lo), hi) - lo) / (hi - lo)
    return (f'<svg width=160 height=22 role=img aria-label="asking vs band"><rect x={x(a["fair_low"]):.0f} y=7 '
            f'width={x(a["fair_high"]) - x(a["fair_low"]):.0f} height=8 rx=4 fill="var(--band)"/>'
            f'<line x1={x(a["fair_value"]):.0f} x2={x(a["fair_value"]):.0f} y1=4 y2=18 stroke="var(--acc)" stroke-width=2/>'
            f'<circle cx={x(a["asking_price"]):.0f} cy=11 r=4.5 fill="var(--ask)"/></svg>')
