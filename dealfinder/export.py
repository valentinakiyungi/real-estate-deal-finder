"""Export the trained model and the analysed sample market for the static (GitHub Pages) site.

    python main.py build-site          # writes docs/data/model.json and docs/data/market.json

The static site re-implements the rule parser, feature engineering, tree evaluation and deal rules in
JavaScript (docs/js/engine.js), so price checks run in the visitor's browser with no server.
Before writing, this module re-scores the exported trees in pure Python and refuses to export if the
result differs from scikit-learn, so the website always matches the Python model.
"""
from __future__ import annotations

import json
import math
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import agent, data, model as model_mod, services
from .model import RAW_COLS, PriceModel

DOCS = services.ROOT / "docs"


def _num(x: float) -> float:
    # full precision: some inputs sit within 1e-9 of a split point, so rounding would change predictions
    return float(x)


def _trees(est) -> dict:
    """Flatten a HistGradientBoostingRegressor into arrays: per tree [feature, threshold_or_value, left, right]."""
    trees = []
    for (pred,) in est._predictors:
        n = pred.nodes
        if bool(n["is_categorical"].any()):  # our features are all numeric and never missing
            raise ValueError("categorical splits are not supported by the static site")
        feat = np.where(n["is_leaf"], -1, n["feature_idx"]).astype(int).tolist()
        thr = [_num(v if leaf else t) for v, t, leaf in zip(n["value"], n["num_threshold"], n["is_leaf"])]
        trees.append([feat, thr, n["left"].astype(int).tolist(), n["right"].astype(int).tolist()])
    return {"baseline": _num(float(np.ravel(est._baseline_prediction)[0])), "trees": trees}


def _score(m: dict, x: list[float]) -> float:
    total = m["baseline"]
    for feat, thr, left, right in m["trees"]:
        i = 0
        while feat[i] != -1:
            i = left[i] if x[feat[i]] <= thr[i] else right[i]
        total += thr[i]
    return total


def model_payload(pm: PriceModel) -> dict:
    fe = pm.fe
    return {
        "version": 1,
        "current_year": model_mod.CURRENT_YEAR,
        "features": list(fe.transform(data.historical_sales(n=5, seed=1)[RAW_COLS]).columns),
        "cond_ord": model_mod.COND_ORD,
        "hood_te": {k: _num(v) for k, v in fe.hood_te_.items()},
        "hood_global": _num(fe.global_),
        "conformal_q": _num(getattr(pm, "conformal_q_", 0.0)),
        "point": _trees(pm.point), "lo": _trees(pm.lo), "hi": _trees(pm.hi),
        "risk_haircut": agent.RISK_HAIRCUT,
    }


def verify(pm: PriceModel, payload: dict, frame: pd.DataFrame, tol: float = 1e-6) -> float:
    """Max relative difference between exported-tree scores and scikit-learn. Raises if too large."""
    X = pm.fe.transform(frame[RAW_COLS])
    worst = 0.0
    for key in ("point", "lo", "hi"):
        ref = getattr(pm, key).predict(X)
        for row, r in zip(X.to_numpy(float), ref):
            worst = max(worst, abs(_score(payload[key], row.tolist()) - r) / abs(r))
    if worst > tol:
        raise RuntimeError(f"exported trees disagree with scikit-learn (max rel diff {worst:.2e})")
    return worst


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def market_payload(pm: PriceModel) -> dict:
    listings = services.default_listings()
    res = services.run_search(listings, pm, {}, use_ai=False, out_dir=Path(tempfile.mkdtemp(prefix="ds_site_")))
    frame = services.analyses_frame(list(res["tools"].cache.values()), listings)
    keep = ["id", "address", "neighborhood", "verdict", "asking_price", "fair_value", "fair_low", "fair_high",
            "discount", "risk_adjusted_value", "risk_haircut", "beds", "baths", "sqft", "lot_sqft", "year_built",
            "condition", "renovated", "garage_spaces", "pool", "view", "red_flags", "price_per_sqft",
            "days_on_market", "confidence", "explanation", "description", "lat", "lon"]
    homes = [{k: _clean(v) for k, v in row.items()} for row in frame[keep].to_dict(orient="records")]
    for h in homes:
        for k in ("lat", "lon"):
            if h[k] is not None:
                h[k] = round(h[k], 5)
    return {
        "generated": date.today().isoformat(),
        "source": "Sample market (simulated)",
        "summary": res["tools"].report["market_summary"],
        "homes": homes,
        "neighborhoods": {k: {kk: _clean(vv) for kk, vv in v.items()}
                          for k, v in services.neighborhood_defaults().items()},
        "verdicts": {k: {"label": lab, "color": col} for k, (lab, col) in services.VERDICTS.items()},
        "metrics": pm.metrics,
        "importances": [{"factor": services.FEATURE_LABELS.get(k, k), "value": v}
                        for k, v in pm.importances.items() if v > 0][:10],
        "training_sales": int(len(pd.read_csv(services.SALES_PATH))),
        "map_center": list(services.MAP_CENTER),
    }


def build_site(out_dir: Path = DOCS, pm: PriceModel | None = None) -> dict:
    pm = pm or services.ensure_data_and_model()
    payload = model_payload(pm)
    check = pd.concat([data.historical_sales(n=300, seed=11), data.historical_sales(n=5, seed=1)])
    worst = verify(pm, payload, check)
    market = market_payload(pm)
    d = Path(out_dir) / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps(payload, separators=(",", ":")))
    (d / "market.json").write_text(json.dumps(market, separators=(",", ":")))
    return {"out": str(d), "max_rel_diff": worst, "homes": len(market["homes"]),
            "model_kb": round((d / "model.json").stat().st_size / 1024),
            "market_kb": round((d / "market.json").stat().st_size / 1024)}
