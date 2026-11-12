"""Glue used by both the CLI and the web app (keeps app.py thin and testable)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data, llm
from .agent import DealTools, render_report, run_llm_agent, run_rule_agent
from .model import PriceModel, train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "price_model.joblib"
SALES_PATH = ROOT / "data" / "historical_sales.csv"
LISTINGS_PATH = ROOT / "data" / "active_listings.json"
REPORT_DIR = ROOT / "reports"

REQUIRED_COLS = ["description", "asking_price"]
OPTIONAL_COLS = ["listing_id", "address", "neighborhood", "days_on_market", "dist_center_km", "school_rating"]


def ensure_data_and_model(force: bool = False) -> PriceModel:
    """Generate data and train the model if missing (first run takes ~10 s)."""
    if force or not SALES_PATH.exists() or not LISTINGS_PATH.exists():
        SALES_PATH.parent.mkdir(exist_ok=True)
        data.historical_sales().to_csv(SALES_PATH, index=False)
        data.active_listings().to_json(LISTINGS_PATH, orient="records", indent=1)
    if force or not MODEL_PATH.exists():
        return train_and_evaluate(pd.read_csv(SALES_PATH), MODEL_PATH.parent)
    return PriceModel.load(MODEL_PATH)


def neighborhoods() -> list[str]:
    return list(data.NEIGHBORHOODS)


def neighborhood_defaults() -> dict[str, dict]:
    """Typical distance / school rating per neighbourhood, from past sales."""
    if SALES_PATH.exists():
        s = pd.read_csv(SALES_PATH)
        g = s.groupby("neighborhood")[["dist_center_km", "school_rating"]].median().round(1)
        return g.to_dict(orient="index")
    return {n: {"dist_center_km": v[1], "school_rating": v[2]} for n, v in data.NEIGHBORHOODS.items()}


def default_listings() -> pd.DataFrame:
    return pd.read_json(LISTINGS_PATH)


def prepare_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an uploaded CSV and fill optional columns with sensible defaults."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"id": "listing_id", "listing": "listing_id", "price": "asking_price", "list_price": "asking_price",
               "asking": "asking_price", "desc": "description", "remarks": "description", "details": "description",
               "area": "neighborhood", "neighbourhood": "neighborhood", "district": "neighborhood",
               "dom": "days_on_market"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases and aliases[c] not in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}. "
                         f"Required: {', '.join(REQUIRED_COLS)}; optional: {', '.join(OPTIONAL_COLS)}.")
    df = df.dropna(subset=["description", "asking_price"])
    df["asking_price"] = pd.to_numeric(df["asking_price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                                       errors="coerce")
    df = df.dropna(subset=["asking_price"])
    if df.empty:
        raise ValueError("No usable rows: every row needs a description and a numeric asking_price.")
    df["asking_price"] = df["asking_price"].astype(int)
    if "listing_id" not in df:
        df["listing_id"] = [f"U{i + 1:03d}" for i in range(len(df))]
    df["listing_id"] = df["listing_id"].astype(str)
    if df["listing_id"].duplicated().any():
        df["listing_id"] = df["listing_id"] + "-" + df.groupby("listing_id").cumcount().astype(str)
    if "address" not in df:
        df["address"] = "(no address)"
    if "neighborhood" not in df:
        df["neighborhood"] = "Unknown"
    df["neighborhood"] = df["neighborhood"].fillna("Unknown").astype(str)
    if "days_on_market" not in df:
        df["days_on_market"] = 0
    df["days_on_market"] = pd.to_numeric(df["days_on_market"], errors="coerce").fillna(0).astype(int)
    defaults = neighborhood_defaults()
    glob = {k: float(pd.Series([v[k] for v in defaults.values()]).median()) for k in ("dist_center_km", "school_rating")}
    for col in ("dist_center_km", "school_rating"):
        fill = df["neighborhood"].map(lambda n: defaults.get(n, glob)[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill) if col in df else fill
    return df.reset_index(drop=True)


def run_search(listings: pd.DataFrame, model: PriceModel, filters: dict, use_ai: bool,
               on_event=None, title: str = "Under-priced Homes — Deal Report") -> dict:
    """Run the agent and render the report. Returns tools, mode note and report text."""
    filters = {k: v for k, v in filters.items() if v}
    use_ai = use_ai and llm.available()
    tools = DealTools(listings, model, use_llm_parser=use_ai, on_event=on_event)
    if use_ai:
        note = run_llm_agent(tools, filters)
        mode = f"AI agent ({llm.describe()})" if note == "llm" else note
    else:
        run_rule_agent(tools, filters)
        mode = "Offline planner + rule-based parser"
    md_path, html_path = render_report(tools, REPORT_DIR, title)
    return {"tools": tools, "mode": mode, "markdown": md_path.read_text(), "html": html_path.read_text()}


def value_one(description: str, asking_price: int, neighborhood: str, model: PriceModel, use_ai: bool,
              dist_center_km: float | None = None, school_rating: float | None = None) -> dict:
    d = neighborhood_defaults().get(neighborhood, {"dist_center_km": 6.0, "school_rating": 6.5})
    row = pd.DataFrame([{
        "listing_id": "YOURS", "address": "(your listing)", "neighborhood": neighborhood,
        "asking_price": int(asking_price), "days_on_market": 0, "description": description,
        "dist_center_km": dist_center_km if dist_center_km is not None else d["dist_center_km"],
        "school_rating": school_rating if school_rating is not None else d["school_rating"],
    }])
    tools = DealTools(row, model, use_llm_parser=use_ai and llm.available())
    return tools.analyze_listings(["YOURS"])[0]
