"""Real-Estate Price & Deal Finder.

    python main.py train                       # (re)generate data + train model
    python main.py find                        # run the agent over all listings
    python main.py find --max-price 400000 --neighborhood Riverside
    python main.py parse "3bd/2ba, 1,650 sq ft, built 1988, needs TLC, 2 car garage"
    python main.py evaluate                    # parser + deal-flag accuracy vs ground truth

    streamlit run app.py                       # the web app

Put GEMINI_API_KEY=... (free) or ANTHROPIC_API_KEY=... in a .env file to let the
AI parse listings and drive the agent; without a key everything runs offline.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from dealfinder import llm, services
from dealfinder.agent import DealTools
from dealfinder.parser import parse


def cmd_train(_):
    m = services.ensure_data_and_model(force=True)
    print(json.dumps(m.metrics, indent=2))


def _load():
    model = services.ensure_data_and_model()
    return services.default_listings(), model


def cmd_find(a):
    listings, model = _load()
    if a.csv:
        listings = services.prepare_listings(pd.read_csv(a.csv))
    print("AI:", "off (--offline)" if a.offline else llm.describe())
    res = services.run_search(listings, model, {"neighborhood": a.neighborhood, "max_price": a.max_price},
                              use_ai=not a.offline, on_event=lambda m: print("  ·", m), title=a.title)
    print("mode:", res["mode"])
    print(res["tools"].report["market_summary"])
    for d in res["tools"].report["deals"]:
        print(" -", d["id"], d["headline"])
    print("report:", services.REPORT_DIR / "deal_report.html")


def cmd_parse(a):
    f, meta = parse(a.text)
    print(json.dumps({"features": f.model_dump(), **meta}, indent=2))


def cmd_evaluate(_):
    listings, model = _load()
    tools = DealTools(listings, model, use_llm_parser=False)
    res = pd.DataFrame(tools.analyze_listings(list(listings["listing_id"])))
    df = listings.merge(res[["id", "verdict", "fair_value"]], left_on="listing_id", right_on="id")
    # ground-truth deal: asking >= 10% below true value AND no hidden red flags
    df["true_deal"] = (df["asking_price"] <= 0.90 * df["_truth_value"]) & (df["_truth_red_flags"].str.len() == 0)
    df["flagged"] = df["verdict"].isin(["strong deal", "deal"])
    tp = int((df.true_deal & df.flagged).sum())
    print(f"listings: {len(df)}  true clean deals: {int(df.true_deal.sum())}  flagged: {int(df.flagged.sum())}")
    print(f"precision: {tp / max(1, df.flagged.sum()):.0%}   recall: {tp / max(1, df.true_deal.sum()):.0%}")
    ape = (df.fair_value - df._truth_value).abs() / df._truth_value
    print(f"fair-value MAPE on listings (parsed from text): {ape.mean():.1%}")
    traps = df[df.verdict == "value trap"]
    print(f"value traps caught: {len(traps)} (all have true red flags: {bool((traps._truth_red_flags.str.len() > 0).all())})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    f = sub.add_parser("find")
    f.add_argument("--neighborhood")
    f.add_argument("--max-price", type=int)
    f.add_argument("--offline", action="store_true", help="never call the LLM")
    f.add_argument("--csv", help="your own listings CSV (needs description, asking_price)")
    f.add_argument("--title", default="Under-priced Homes — Deal Report")
    f.set_defaults(fn=cmd_find)
    pp = sub.add_parser("parse")
    pp.add_argument("text")
    pp.set_defaults(fn=cmd_parse)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    a = p.parse_args()
    a.fn(a)
