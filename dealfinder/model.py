"""House-price regression: gradient boosting + feature engineering.

* Point estimate: HistGradientBoostingRegressor on log(price).
* Fair-value band: two quantile GBMs (10th / 90th percentile) so the agent can
  say "under-priced" only when the asking price sits below the plausible range,
  not merely below a single point estimate.
* Neighborhood target encoding is done out-of-fold to avoid leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split

CURRENT_YEAR = 2026
COND_ORD = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
RAW_COLS = ["neighborhood", "beds", "baths", "sqft", "lot_sqft", "year_built", "renovated", "condition",
            "garage_spaces", "pool", "view", "dist_center_km", "school_rating"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw listing attributes -> model features."""

    def fit(self, X: pd.DataFrame, y=None):
        X = X.reset_index(drop=True)
        logy = np.log(np.asarray(y, float)) - np.log(X["sqft"].to_numpy(float))  # log $/sqft
        self.global_ = float(logy.mean())
        stats = pd.DataFrame({"n": X["neighborhood"], "t": logy}).groupby("n")["t"].agg(["mean", "count"])
        k = 20  # smoothing strength
        self.hood_te_ = ((stats["mean"] * stats["count"] + self.global_ * k) / (stats["count"] + k)).to_dict()
        # out-of-fold encoding for the training rows themselves
        oof = np.full(len(X), self.global_)
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            s = pd.DataFrame({"n": X["neighborhood"].iloc[tr], "t": logy[tr]}).groupby("n")["t"].agg(["mean", "count"])
            te = (s["mean"] * s["count"] + self.global_ * k) / (s["count"] + k)
            oof[va] = X["neighborhood"].iloc[va].map(te).fillna(self.global_).to_numpy()
        self._oof = oof
        return self

    def transform(self, X: pd.DataFrame, _use_oof: bool = False) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        f = pd.DataFrame(index=X.index)
        age = CURRENT_YEAR - X["year_built"].astype(float)
        sqft = X["sqft"].astype(float)
        f["log_sqft"] = np.log(sqft)
        f["log_lot"] = np.log(X["lot_sqft"].astype(float).clip(lower=500))
        f["beds"] = X["beds"].astype(float)
        f["baths"] = X["baths"].astype(float)
        f["sqft_per_bed"] = sqft / X["beds"].clip(lower=1)
        f["bath_bed_ratio"] = X["baths"] / X["beds"].clip(lower=1)
        f["age"] = age
        f["renovated"] = X["renovated"].astype(int)
        f["effective_age"] = np.where(X["renovated"].astype(bool), np.minimum(age, 8), age)
        f["condition"] = X["condition"].map(COND_ORD).astype(float)
        f["cond_x_age"] = f["condition"] * np.log1p(age)
        f["garage"] = X["garage_spaces"].astype(float)
        f["pool"] = X["pool"].astype(int)
        f["view"] = X["view"].astype(int)
        f["dist_center_km"] = X["dist_center_km"].astype(float)
        f["school_rating"] = X["school_rating"].astype(float)
        f["lot_to_sqft"] = X["lot_sqft"] / sqft
        f["hood_ppsf"] = (self._oof if _use_oof else
                          X["neighborhood"].map(self.hood_te_).fillna(self.global_).to_numpy())
        f["hood_x_sqft"] = f["hood_ppsf"] + f["log_sqft"]  # expected log price from location * size
        return f


@dataclass
class PriceModel:
    fe: FeatureEngineer = field(default_factory=FeatureEngineer)
    point: HistGradientBoostingRegressor | None = None
    lo: HistGradientBoostingRegressor | None = None
    hi: HistGradientBoostingRegressor | None = None
    metrics: dict = field(default_factory=dict)
    importances: dict = field(default_factory=dict)

    @staticmethod
    def _gbm(**kw):
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=20, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=30, random_state=0, **kw)

    def fit(self, df: pd.DataFrame) -> "PriceModel":
        y = np.log(df["sale_price"].to_numpy(float))
        X = self.fe.fit(df[RAW_COLS], df["sale_price"]).transform(df[RAW_COLS], _use_oof=True)
        self.point = self._gbm().fit(X, y)
        # Conformalized quantile regression: fit quantiles on 80%, calibrate width on 20%.
        idx_fit, idx_cal = train_test_split(np.arange(len(y)), test_size=0.2, random_state=1)
        self.lo = self._gbm(loss="quantile", quantile=0.10).fit(X.iloc[idx_fit], y[idx_fit])
        self.hi = self._gbm(loss="quantile", quantile=0.90).fit(X.iloc[idx_fit], y[idx_fit])
        Xc, yc = X.iloc[idx_cal], y[idx_cal]
        scores = np.maximum(self.lo.predict(Xc) - yc, yc - self.hi.predict(Xc))
        n = len(yc)
        self.conformal_q_ = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.8) / n)))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.fe.transform(df[RAW_COLS])
        q = getattr(self, "conformal_q_", 0.0)
        p = np.exp(self.point.predict(X))
        lo = np.exp(self.lo.predict(X) - q)
        hi = np.exp(self.hi.predict(X) + q)
        lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        return pd.DataFrame({"fair_value": p.round(-3), "fair_low": lo.round(-3), "fair_high": hi.round(-3)})

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PriceModel":
        return joblib.load(path)


def train_and_evaluate(sales: pd.DataFrame, out_dir: Path) -> PriceModel:
    tr, te = train_test_split(sales, test_size=0.2, random_state=42)
    m = PriceModel().fit(tr)
    pred = m.predict(te)
    y = te["sale_price"].to_numpy(float)
    p = pred["fair_value"].to_numpy(float)
    ape = np.abs(p - y) / y
    # baseline: neighborhood median $/sqft * sqft
    ppsf = (tr["sale_price"] / tr["sqft"]).groupby(tr["neighborhood"]).median()
    base = te["neighborhood"].map(ppsf).to_numpy() * te["sqft"].to_numpy()
    metrics = {
        "test_rows": int(len(te)),
        "MAE": float(np.mean(np.abs(p - y))),
        "MAPE": float(ape.mean()),
        "median_APE": float(np.median(ape)),
        "within_10pct": float((ape <= 0.10).mean()),
        "R2_log": float(1 - np.sum((np.log(p) - np.log(y)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)),
        "interval_80_coverage": float(((y >= pred["fair_low"]) & (y <= pred["fair_high"])).mean()),
        "baseline_ppsf_MAPE": float(np.mean(np.abs(base - y) / y)),
    }
    Xte = m.fe.transform(te[RAW_COLS])
    imp = permutation_importance(m.point, Xte, np.log(y), n_repeats=5, random_state=0)
    importances = dict(sorted(zip(Xte.columns, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    final = PriceModel().fit(sales)  # refit on everything for deployment
    final.metrics, final.importances = metrics, importances
    out_dir.mkdir(parents=True, exist_ok=True)
    final.save(out_dir / "price_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "importances": importances}, indent=2))
    return final


if __name__ == "__main__":
    # import via the package so pickled classes resolve as dealfinder.model.*
    from dealfinder.model import train_and_evaluate
    root = Path(__file__).resolve().parents[1]
    m = train_and_evaluate(pd.read_csv(root / "data" / "historical_sales.csv"), root / "models")
    print(json.dumps(m.metrics, indent=2))
    print("top features:", list(m.importances)[:8])
