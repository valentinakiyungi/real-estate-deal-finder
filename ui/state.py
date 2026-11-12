"""Per-visitor state: the current market scan and the watchlist."""
from __future__ import annotations

import tempfile

import pandas as pd
import streamlit as st

from dealfinder import llm, services

PAGES: dict = {}          # filled by app.py so pages can link to each other
AI_RUNS_PER_VISIT = 3     # protects a shared free API quota when the site is public
SAMPLE_SOURCE = "Sample market (simulated)"


@st.cache_resource(show_spinner="Preparing the price model. The first visit takes about 10 seconds.")
def get_model():
    return services.ensure_data_and_model()


@st.cache_resource(show_spinner="Scanning today's listings")
def _sample_scan():
    listings = services.default_listings()
    res = services.run_search(listings, get_model(), {}, use_ai=False, out_dir=tempfile.mkdtemp(prefix="ds_"))
    return listings, res


def market() -> dict:
    if "market" not in st.session_state:
        listings, res = _sample_scan()
        st.session_state.market = {"listings": listings, "analyses": dict(res["tools"].cache), "result": res,
                                   "source": SAMPLE_SOURCE, "mode": res["mode"]}
    return st.session_state.market


def set_market(listings: pd.DataFrame, result: dict, source: str, keep_existing: bool = False) -> None:
    analyses = dict(market()["analyses"]) if keep_existing else {}
    analyses.update(result["tools"].cache)  # newer (e.g. AI) analyses win
    st.session_state.market = {"listings": listings, "analyses": analyses, "result": result,
                               "source": source, "mode": result["mode"]}
    st.session_state.pop("deals_shown", None)


def reset_to_sample() -> None:
    st.session_state.pop("market", None)
    st.session_state.pop("deals_shown", None)


def frame() -> pd.DataFrame:
    m = market()
    return services.analyses_frame(list(m["analyses"].values()), m["listings"])


def report_notes() -> dict[str, dict]:
    """Deal write-ups from the last scan's report, keyed by listing id."""
    rep = market()["result"]["tools"].report or {}
    return {d["id"]: d for d in rep.get("deals", [])}


def scan(listings: pd.DataFrame, source: str, use_ai: bool, on_event=None) -> dict:
    res = services.run_search(listings, get_model(), {}, use_ai=use_ai, on_event=on_event,
                              out_dir=tempfile.mkdtemp(prefix="ds_"))
    set_market(listings, res, source, keep_existing=use_ai and market()["source"] == source)
    if use_ai:
        st.session_state.ai_runs = ai_runs() + 1
    return res


def ai_available() -> bool:
    return llm.available()


def ai_runs() -> int:
    return st.session_state.get("ai_runs", 0)


# ---------------------------------------------------------------- watchlist ---
def watch() -> dict[str, dict]:
    return st.session_state.setdefault("watch", {})


def is_saved(listing_id: str) -> bool:
    return listing_id in watch()


def toggle_save(row: dict) -> None:
    w = watch()
    if row["id"] in w:
        del w[row["id"]]
    else:
        w[row["id"]] = row


def watch_frame() -> pd.DataFrame:
    return pd.DataFrame(list(watch().values()))
