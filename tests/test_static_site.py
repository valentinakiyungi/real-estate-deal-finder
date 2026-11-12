"""The static GitHub Pages site must give exactly the same answers as the Python app."""
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dealfinder import data, services
from dealfinder.agent import DealTools
from dealfinder.export import build_site

ROOT = Path(__file__).resolve().parents[1]
EDGE = [
    ("3 bed / 1 full + 1 half bath, approx 1,480 SF on a 0.21 acre lot, circa 1962, needs TLC, cash only", "Riverside"),
    ("three bedrooms / two bathrooms, 2.4k sq. ft. of living space, lot 0.30 acres, yr blt 2011, pristine, pool",
     "Lakeview"),
    ("4BR | 3BA | 1,200 sf lot | 2,050 sqft | built 1999 | 1,200 sf lot | tenant occupied | flood zone", "Eastgate"),
    ("2 bed 1 bath 900 sqft built 1940, foundation issues disclosed, estate sale, roof at end of life", "Old Town"),
    ("5 bdrm, 4.5 baths, 3,900 sq ft, constructed in 2019, 3 car garage, city views, unpermitted addition",
     "Atlantis"),
    ("lovely home, call now", "Riverside"),
    ("3 bed 2 bath 1500 sqft, lot 60 acres, built 1990, 9 car garage", "Northbrook"),
    ("1 bed 1 bath 450 sqft built 2026 immaculate", "Old Town"),
    ("6 bed / 5 bath / 6000 sqft / built 1925 / handyman special / 2 car gar.", "Cedar Hills"),
]


def _cases(pm):
    listings = pd.concat([services.default_listings(), data.active_listings(n=300, seed=77).assign(
        listing_id=lambda d: "X" + d["listing_id"])], ignore_index=True)
    defaults = services.neighborhood_defaults()
    extra = pd.DataFrame([{
        "listing_id": f"E{i}", "address": "edge", "neighborhood": h, "asking_price": 260_000 + 37_000 * i,
        "days_on_market": 0, "description": d, "dist_center_km": defaults.get(h, {"dist_center_km": 6.0})["dist_center_km"],
        "school_rating": defaults.get(h, {"school_rating": 6.5})["school_rating"]} for i, (d, h) in enumerate(EDGE)])
    listings = pd.concat([listings, extra], ignore_index=True)
    tools = DealTools(listings, pm, use_llm_parser=False)
    out = []
    for a, (_, r) in zip(tools.analyze_listings(list(listings["listing_id"])), listings.iterrows()):
        a = dict(a)
        expect = {k: a.get(k) for k in ("verdict", "fair_value", "fair_low", "fair_high", "risk_adjusted_value",
                                         "risk_haircut", "price_per_sqft", "confidence") if k in a}
        if "discount_vs_fair" in a:
            expect["discount"] = a["discount_vs_fair"]
            a["discount"] = a["discount_vs_fair"]
        expect["explanation"] = services.explain(a)
        out.append({"input": {"id": r["listing_id"], "description": r["description"], "asking_price": int(r["asking_price"]),
                              "neighborhood": r["neighborhood"], "dist_center_km": float(r["dist_center_km"]),
                              "school_rating": float(r["school_rating"])},
                    "features": a["features"], "expect": expect})
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_engine_matches_python(tmp_path):
    pm = services.ensure_data_and_model()
    info = build_site(out_dir=tmp_path, pm=pm)
    assert info["max_rel_diff"] == 0.0
    cases = _cases(pm)
    (tmp_path / "cases.json").write_text(json.dumps(cases, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    r = subprocess.run(["node", str(ROOT / "tests" / "parity.mjs"), str(ROOT / "docs" / "js" / "engine.js"),
                        str(tmp_path / "data" / "model.json"), str(tmp_path / "cases.json")],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def test_market_json_is_complete(tmp_path):
    build_site(out_dir=tmp_path)
    m = json.loads((tmp_path / "data" / "market.json").read_text())
    assert len(m["homes"]) == 60 and m["metrics"]["MAPE"] > 0
    assert all(h["lat"] is not None and h["explanation"] for h in m["homes"])
    assert set(m["verdicts"]) >= {"strong deal", "value trap"}
