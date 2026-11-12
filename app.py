"""DealScout — find homes listed below what they're worth.

Run locally:   python -m streamlit run app.py
Deploy:        see "Put it online" in the README.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ICON = str(ROOT / "assets" / "icon.png")

st.set_page_config(page_title="DealScout: homes below value", page_icon=ICON, layout="wide",
                   initial_sidebar_state="collapsed")


def _load_secrets() -> None:
    """On Streamlit Community Cloud the API key lives in the app's Secrets, not in a .env file."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str) and k.isupper() and not os.environ.get(k):
                os.environ[k] = v
    except Exception:
        pass  # no secrets configured: fine locally (the .env file is used instead)


_load_secrets()

from ui import state, theme, views  # noqa: E402  (import after secrets so the AI key is visible)

st.markdown(theme.CSS, unsafe_allow_html=True)
st.logo(ICON, size="large")

n_saved = len(state.watch())
pages = {
    "home": st.Page(views.home, title="Home", icon=":material/home:", default=True),
    "deals": st.Page(views.deals, title="Deals", icon=":material/sell:", url_path="deals"),
    "check": st.Page(views.price_check, title="Check a price", icon=":material/calculate:", url_path="check"),
    "watch": st.Page(views.watchlist, title=f"Watchlist ({n_saved})" if n_saved else "Watchlist",
                     icon=":material/bookmark:", url_path="watchlist"),
    "method": st.Page(views.method, title="How it works", icon=":material/info:", url_path="how-it-works"),
}
state.PAGES.update(pages)
st.navigation(list(pages.values()), position="top").run()
