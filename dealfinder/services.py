"""Glue used by both the CLI and the web app (keeps app.py thin and testable)."""
from __future__ import annotations

import hashlib
import math
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
OPTIONAL_COLS = ["listing_id", "address", "neighborhood", "days_on_market", "dist_center_km", "school_rating",
                 "lat", "lon"]


def ensure_data_and_model(force: bool = False) -> PriceModel:
    """Generate data and train the model if missing (first run takes ~10 s)."""
    if force or not SALES_PATH.exists() or not LISTINGS_PATH.exists():
        SALES_PATH.parent.mkdir(exist_ok=True)
        data.historical_sales().to_csv(SALES_PATH, index=False)
        data.active_listings().to_json(LISTINGS_PATH, orient="records", indent=1)
    if not force and MODEL_PATH.exists():
        try:
            return PriceModel.load(MODEL_PATH)
        except Exception:
            pass  # saved with different numpy/scikit-learn versions -> retrain below
    return train_and_evaluate(pd.read_csv(SALES_PATH), MODEL_PATH.parent)


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
               "dom": "days_on_market", "latitude": "lat", "longitude": "lon", "lng": "lon", "long": "lon"}
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
               on_event=None, title: str = "Under-priced Homes — Deal Report", out_dir: Path | None = None) -> dict:
    """Run the agent and render the report. Returns tools, mode note and report text.

    out_dir: where report files go (the web app passes a private temp folder per run so
    simultaneous visitors never overwrite each other's reports)."""
    filters = {k: v for k, v in filters.items() if v}
    use_ai = use_ai and llm.available()
    tools = DealTools(listings, model, use_llm_parser=use_ai, on_event=on_event)
    if use_ai:
        note = run_llm_agent(tools, filters)
        mode = f"AI analyst ({llm.describe()})" if note == "llm" else note
    else:
        run_rule_agent(tools, filters)
        mode = "Quick scan (built-in parser and planner)"
    md_path, html_path = render_report(tools, Path(out_dir) if out_dir else REPORT_DIR, title)
    return {"tools": tools, "mode": mode, "ai": use_ai and mode.startswith("AI analyst"),
            "markdown": md_path.read_text(), "html": html_path.read_text()}


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
    out = dict(tools.analyze_listings(["YOURS"])[0])
    out["description"] = description
    return out


# ------------------------------------------------------------ presentation ---
# User-facing verdict names (the engine's internal names stay short and stable).
VERDICTS = {
    "strong deal": ("Well below value", "#1E7F5C"),
    "deal": ("Below value", "#3B9474"),
    "speculative": ("Below value, has risks", "#B9741A"),
    "value trap": ("Cheap for a reason", "#A8373A"),
    "fair": ("Fairly priced", "#6B7B84"),
    "overpriced": ("Above value", "#45545D"),
    "insufficient data": ("Not enough info", "#9AA6AC"),
}
DEAL_VERDICTS = ("strong deal", "deal")

# Plain-English names for the model's features (How it works page, static site)
FEATURE_LABELS = {"hood_x_sqft": "Location and size together", "effective_age": "Age (since last renovation)",
    "condition": "Condition", "garage": "Garage spaces", "hood_ppsf": "Neighborhood price level",
    "school_rating": "School rating", "pool": "Pool", "view": "View", "log_sqft": "Living area",
    "cond_x_age": "Condition for its age", "dist_center_km": "Distance to centre", "log_lot": "Lot size",
    "baths": "Bathrooms", "beds": "Bedrooms", "age": "Age", "renovated": "Renovated",
    "sqft_per_bed": "Space per bedroom", "bath_bed_ratio": "Bathrooms per bedroom",
    "lot_to_sqft": "Lot relative to house"}

# The sample market is simulated; pins are placed around this point for illustration.
MAP_CENTER = (39.9612, -82.9988)
_BEARING = {"Riverside": 200, "Old Town": 25, "Maple Heights": 295, "Lakeview": 75, "Eastgate": 110,
            "Industrial Pk": 160, "Cedar Hills": 245, "Northbrook": 345}


def verdict_label(v: str | None) -> str:
    return VERDICTS.get(v or "", (str(v or "Unknown").capitalize(), ""))[0]


def verdict_color(v: str | None) -> str:
    return VERDICTS.get(v or "", ("", "#9AA6AC"))[1]


def money_short(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    v = float(v)
    return f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v / 1e3:.0f}k"


def _unit(key: str) -> tuple[float, float]:
    h = hashlib.md5(key.encode()).digest()
    return h[0] / 255, h[1] / 255


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Give each listing a map position. Uses lat/lon columns when present; otherwise places it
    by neighbourhood bearing and its distance to the centre (deterministic, illustrative)."""
    df = df.copy()
    lat = pd.to_numeric(df["lat"], errors="coerce") if "lat" in df else pd.Series(float("nan"), index=df.index)
    lon = pd.to_numeric(df["lon"], errors="coerce") if "lon" in df else pd.Series(float("nan"), index=df.index)
    for i, r in df.iterrows():
        if pd.notna(lat[i]) and pd.notna(lon[i]):
            continue
        bearing = _BEARING.get(str(r.get("neighborhood")))
        if bearing is None:
            continue
        u1, u2 = _unit(str(r.get("listing_id", i)))
        d = max(0.4, float(r.get("dist_center_km", 5) or 5) + (u1 - 0.5) * 1.2)
        b = math.radians(bearing + (u2 - 0.5) * 34)
        lat[i] = MAP_CENTER[0] + d * math.cos(b) / 111.0
        lon[i] = MAP_CENTER[1] + d * math.sin(b) / (111.0 * math.cos(math.radians(MAP_CENTER[0])))
    df["lat"], df["lon"] = lat, lon
    return df


def explain(a: dict) -> str:
    """Plain-language explanation of one analysis."""
    if a.get("verdict") == "insufficient data":
        return f"We couldn't find {', '.join(a.get('missing', []))} in the description, so we can't value it."
    gap = a["discount_vs_fair"]
    rel = (f"{gap:.0%} below" if gap > 0.005 else f"{-gap:.0%} above" if gap < -0.005 else "right at")
    s = (f"Asking {money_short(a['asking_price'])} is {rel} our estimate of {money_short(a['fair_value'])} "
         f"(likely range {money_short(a['fair_low'])} to {money_short(a['fair_high'])}).")
    flags = a.get("features", {}).get("red_flags", [])
    if flags:
        s += (f" The listing discloses {', '.join(flags)}; allowing {a['risk_haircut']:.0%} for that puts the "
              f"value at {money_short(a['risk_adjusted_value'])}.")
    for n in a.get("notes", []):
        s += f" Note: {n}."
    return s


def analyses_frame(analyses: list[dict], listings: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per analysed home, with everything the cards, map, table and compare views need."""
    extra = {}
    if listings is not None and len(listings):
        lst = add_coordinates(listings)
        extra = lst.set_index("listing_id")[["description", "lat", "lon"]].to_dict(orient="index")
    rows = []
    for a in analyses:
        f = a.get("features", {})
        x = extra.get(a["id"], {})
        rows.append({
            "id": a["id"], "address": a["address"], "neighborhood": a["neighborhood"],
            "verdict": a.get("verdict"), "verdict_label": verdict_label(a.get("verdict")),
            "asking_price": a["asking_price"], "fair_value": a.get("fair_value"),
            "fair_low": a.get("fair_low"), "fair_high": a.get("fair_high"),
            "discount": a.get("discount_vs_fair"), "risk_adj_discount": a.get("discount_vs_risk_adjusted"),
            "risk_adjusted_value": a.get("risk_adjusted_value"), "risk_haircut": a.get("risk_haircut", 0),
            "beds": f.get("beds"), "baths": f.get("baths"), "sqft": f.get("sqft"), "lot_sqft": f.get("lot_sqft"),
            "year_built": f.get("year_built"), "condition": f.get("condition"), "renovated": f.get("renovated"),
            "garage_spaces": f.get("garage_spaces"), "pool": f.get("pool"), "view": f.get("view"),
            "red_flags": ", ".join(f.get("red_flags", [])), "price_per_sqft": a.get("price_per_sqft"),
            "days_on_market": a.get("days_on_market", 0), "confidence": a.get("confidence"),
            "parsed_by": a.get("parser"), "explanation": explain(a),
            "description": x.get("description", a.get("description", "")),
            "lat": x.get("lat", float("nan")), "lon": x.get("lon", float("nan")),
        })
    return pd.DataFrame(rows)


COMPARE_FIELDS = [
    ("Verdict", lambda r: r["verdict_label"]),
    ("Asking price", lambda r: f"${r['asking_price']:,.0f}"),
    ("Our estimate", lambda r: f"${r['fair_value']:,.0f}" if pd.notna(r["fair_value"]) else "–"),
    ("Likely range", lambda r: f"{money_short(r['fair_low'])} to {money_short(r['fair_high'])}"),
    ("Asking vs estimate", lambda r: "–" if pd.isna(r["discount"]) else
     f"{r['discount']:.0%} below" if r["discount"] >= 0 else f"{-r['discount']:.0%} above"),
    ("Value after risks", lambda r: money_short(r["risk_adjusted_value"])),
    ("Neighborhood", lambda r: r["neighborhood"]),
    ("Bedrooms", lambda r: r["beds"]),
    ("Bathrooms", lambda r: f"{r['baths']:g}" if pd.notna(r["baths"]) else "–"),
    ("Living area", lambda r: f"{r['sqft']:,.0f} sq ft" if pd.notna(r["sqft"]) else "–"),
    ("Price per sq ft", lambda r: f"${r['price_per_sqft']:,.0f}" if pd.notna(r["price_per_sqft"]) else "–"),
    ("Year built", lambda r: r["year_built"]),
    ("Condition", lambda r: str(r["condition"]).capitalize()),
    ("Garage", lambda r: f"{r['garage_spaces']} car" if r["garage_spaces"] else "None"),
    ("Pool", lambda r: "Yes" if r["pool"] else "No"),
    ("Disclosed problems", lambda r: r["red_flags"] or "None"),
    ("Days listed", lambda r: r["days_on_market"]),
    ("Confidence", lambda r: str(r["confidence"]).capitalize()),
]


def compare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side table: one column per home, one row per attribute."""
    cols = {}
    for _, r in frame.iterrows():
        name = f"{r['address']} ({r['id']})"
        vals = []
        for _, fn in COMPARE_FIELDS:
            try:
                v = fn(r)
            except Exception:
                v = "–"
            vals.append("–" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
        cols[name] = vals
    return pd.DataFrame(cols, index=[k for k, _ in COMPARE_FIELDS])
