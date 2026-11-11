"""Synthetic housing market.

Real listing feeds (Zillow, Redfin, MLS) need API keys and have restrictive terms,
so this module generates a realistic stand-in:

* ``historical_sales``  - clean, structured past sales used to TRAIN the price model.
* ``active_listings``   - current listings where most facts live only in a messy
                          free-text description, the way real listings look. The
                          GenAI parser has to recover the structure.

Swap either function for a real data source with the same columns and the rest
of the pipeline works unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEIGHBORHOODS = {
    #  name           base $/sqft  dist_km  school
    "Riverside":     (310, 2.5, 8.5),
    "Old Town":      (280, 1.0, 7.0),
    "Maple Heights": (240, 6.0, 8.0),
    "Lakeview":      (295, 8.0, 7.5),
    "Eastgate":      (175, 9.0, 5.5),
    "Industrial Pk": (150, 4.5, 4.5),
    "Cedar Hills":   (220, 12.0, 7.0),
    "Northbrook":    (200, 14.0, 6.0),
}
CONDITIONS = ["poor", "fair", "good", "excellent"]
CONDITION_MULT = {"poor": 0.78, "fair": 0.90, "good": 1.0, "excellent": 1.10}
CURRENT_YEAR = 2026


def _true_value(df: pd.DataFrame, rng: np.random.Generator | None = None) -> np.ndarray:
    """The hidden 'ground-truth' pricing function (nonlinear, with interactions)."""
    base = df["neighborhood"].map(lambda n: NEIGHBORHOODS[n][0]).to_numpy(float)
    sqft = df["sqft"].to_numpy(float)
    age = CURRENT_YEAR - df["year_built"].to_numpy(float)
    # diminishing returns on size
    size_val = base * sqft * (sqft / 1800) ** -0.12
    age_mult = np.where(df["renovated"].to_numpy(bool), 1.0 - 0.0008 * age, 1.0 - 0.0035 * np.minimum(age, 80))
    cond_mult = df["condition"].map(CONDITION_MULT).to_numpy(float)
    school_mult = 1 + 0.035 * (df["school_rating"].to_numpy(float) - 6.5)
    dist_mult = 1 - 0.008 * df["dist_center_km"].to_numpy(float)
    extras = (
        14_000 * df["garage_spaces"].to_numpy(float)
        + 28_000 * df["pool"].to_numpy(float)
        + 0.06 * size_val * df["view"].to_numpy(float)
        + 9 * np.sqrt(df["lot_sqft"].to_numpy(float)) * 10
        + 6_000 * (df["baths"].to_numpy(float) - df["beds"].to_numpy(float) / 2)
    )
    v = size_val * age_mult * cond_mult * school_mult * dist_mult + extras
    if rng is not None:
        v *= rng.lognormal(0, 0.07, len(v))  # idiosyncratic noise
    return v


def _base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    hoods = rng.choice(list(NEIGHBORHOODS), n, p=[.14, .12, .15, .10, .14, .08, .14, .13])
    beds = rng.choice([1, 2, 3, 4, 5, 6], n, p=[.05, .2, .38, .25, .09, .03])
    sqft = np.clip(rng.normal(450 + 430 * beds, 220), 450, 6000).round(-1)
    baths = np.clip(np.round((beds * 0.6 + rng.normal(0.4, 0.5, n)) * 2) / 2, 1, 5)
    year = rng.integers(1925, 2026, n)
    renovated = (year < 1995) & (rng.random(n) < 0.35)
    cond_p = np.where(year[:, None] > 2010, [.02, .1, .5, .38], [.12, .3, .43, .15])
    condition = [rng.choice(CONDITIONS, p=p) for p in cond_p]
    df = pd.DataFrame({
        "neighborhood": hoods,
        "beds": beds,
        "baths": baths,
        "sqft": sqft.astype(int),
        "lot_sqft": np.clip(rng.lognormal(8.6, 0.5, n), 1200, 40000).round(-1).astype(int),
        "year_built": year,
        "renovated": renovated,
        "condition": condition,
        "garage_spaces": rng.choice([0, 1, 2, 3], n, p=[.2, .3, .42, .08]),
        "pool": rng.random(n) < 0.12,
        "view": rng.random(n) < 0.10,
    })
    df["dist_center_km"] = [round(max(0.3, NEIGHBORHOODS[h][1] + rng.normal(0, 1.2)), 1) for h in hoods]
    df["school_rating"] = [round(float(np.clip(NEIGHBORHOODS[h][2] + rng.normal(0, 0.6), 1, 10)), 1) for h in hoods]
    return df


def historical_sales(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_frame(n, rng)
    df["sale_price"] = _true_value(df, rng).round(-3).astype(int)
    df["sale_date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    return df


# ---------------------------------------------------------------- listings ---

_OPENERS = [
    "Welcome home!", "PRICE IMPROVEMENT!!", "Don't miss this one.", "Investor alert -", "Charming",
    "Just listed:", "Motivated seller!", "Rare find in", "Bring your vision!", "Move-in ready",
]
_COND_TEXT = {
    "poor": ["needs TLC", "sold as-is, needs major work", "handyman special", "fixer upper - bring your tools"],
    "fair": ["some updating needed", "original kitchen, solid bones", "a little cosmetic work and it shines"],
    "good": ["well maintained", "nicely kept", "clean and bright"],
    "excellent": ["immaculate", "pristine condition, turnkey", "stunning designer finishes"],
}
_RED_FLAGS = [
    "foundation issues disclosed", "cash only - will not qualify for financing",
    "in a flood zone", "roof at end of life", "tenant occupied, lease through 2028",
    "estate sale, sold as-is", "unpermitted addition",
]
_WORD_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _beds_text(b: int, rng) -> str:
    return rng.choice([f"{b} bed", f"{b}BR", f"{_WORD_NUM[b]} bedrooms", f"{b} bdrm", f"{b}bd"])


def _baths_text(b: float, rng) -> str:
    if b != int(b):
        full = int(b)
        return rng.choice([f"{b:g} baths", f"{full} full + 1 half bath", f"{b:g}BA"])
    return rng.choice([f"{int(b)} bath", f"{int(b)}BA", f"{_WORD_NUM.get(int(b), int(b))} bathrooms"])


def _sqft_text(s: int, rng) -> str:
    return rng.choice([f"{s:,} sq ft", f"{s} sqft", f"approx {s:,} SF", f"{s/1000:.1f}k sq. ft. of living space"])


def _describe(row: pd.Series, flags: list[str], rng) -> str:
    bits = [rng.choice(_OPENERS)]
    bits.append(f"{_beds_text(int(row.beds), rng)} / {_baths_text(float(row.baths), rng)}")
    bits.append(_sqft_text(int(row.sqft), rng))
    yb = int(row.year_built)
    bits.append(rng.choice([f"built {yb}", f"circa {yb}", f"yr blt {yb}", f"constructed in {yb}"]))
    bits.append(rng.choice(_COND_TEXT[row.condition]))
    if row.renovated:
        bits.append(rng.choice(["fully renovated 2021", "gut reno w/ new kitchen + baths", "remodeled top to bottom"]))
    g = int(row.garage_spaces)
    if g:
        bits.append(rng.choice([f"{g}-car garage", f"{g} car gar.", f"attached garage ({g})"]))
    else:
        bits.append(rng.choice(["street parking", "no garage", ""]))
    if row.pool:
        bits.append(rng.choice(["sparkling pool", "inground pool!", "backyard oasis w/ pool"]))
    if row.view:
        bits.append(rng.choice(["breathtaking lake views", "city views", "panoramic mountain view"]))
    bits.append(rng.choice([f"{int(row.lot_sqft):,} sf lot", f"lot {row.lot_sqft/43560:.2f} acres", f"{int(row.lot_sqft)} sqft lot"]))
    bits.extend(flags)
    bits.append(rng.choice(["Call today!", "Won't last!!", "Showings start Sat.", "", "Offers due Monday."]))
    order = bits[:1] + list(rng.permutation(bits[1:-1])) + bits[-1:]
    sep = rng.choice([", ", ". ", " | ", " ~ ", "... "])
    return sep.join(b for b in order if b).strip()


def active_listings(n: int = 60, seed: int = 2026) -> pd.DataFrame:
    """Current listings. Only id, address, neighborhood, asking price and a messy
    description are 'visible'; the ``_truth_*`` columns are kept for evaluation only."""
    rng = np.random.default_rng(seed)
    df = _base_frame(n, rng)
    true_val = _true_value(df, rng)

    flags = [list(rng.choice(_RED_FLAGS, rng.integers(1, 3), replace=False)) if rng.random() < 0.18 else []
             for _ in range(n)]
    # Most sellers list near market; a few are genuinely under-priced; red-flag
    # listings are cheap *for a reason* (the trap the agent must catch).
    ratio = rng.normal(1.03, 0.06, n)
    bargains = rng.random(n) < 0.15
    ratio[bargains] = rng.uniform(0.72, 0.86, bargains.sum())
    has_flag = np.array([bool(f) for f in flags])
    ratio[has_flag] = rng.uniform(0.62, 0.80, has_flag.sum())
    asking = (true_val * ratio).round(-3).astype(int)

    streets = ["Oak", "Pine", "Elm", "Birch", "Willow", "Harbor", "Summit", "Mill", "Park", "Spring", "Ridge", "Bay"]
    kinds = ["St", "Ave", "Ln", "Dr", "Ct", "Way"]
    out = pd.DataFrame({
        "listing_id": [f"L{1000 + i}" for i in range(n)],
        "address": [f"{rng.integers(10, 9999)} {rng.choice(streets)} {rng.choice(kinds)}" for _ in range(n)],
        "neighborhood": df["neighborhood"],
        "asking_price": asking,
        "days_on_market": rng.integers(1, 120, n),
        "dist_center_km": df["dist_center_km"],
        "school_rating": df["school_rating"],
        "description": [_describe(r, f, rng) for (_, r), f in zip(df.iterrows(), flags)],
    })
    for c in ["beds", "baths", "sqft", "lot_sqft", "year_built", "renovated", "condition", "garage_spaces", "pool", "view"]:
        out[f"_truth_{c}"] = df[c]
    out["_truth_value"] = true_val.round(-3)
    out["_truth_red_flags"] = flags
    return out


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(exist_ok=True)
    historical_sales().to_csv(root / "historical_sales.csv", index=False)
    active_listings().to_json(root / "active_listings.json", orient="records", indent=1)
    print("wrote", root)
