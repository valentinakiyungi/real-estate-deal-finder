"""Brand, colours and global styles.

Rename the product by changing BRAND. Colours are mirrored in .streamlit/config.toml.
"""
from __future__ import annotations

import math

BRAND = "DealScout"
TAGLINE = "Find homes listed below what they're worth."

PAPER = "#F5F7F6"
INK = "#1B2A33"
SURVEY = "#2F5D8A"
SURVEY_LIGHT = "#D6E2EE"
GREEN = "#1E7F5C"
AMBER = "#B9741A"
RED = "#A8373A"
LINE = "#D3DCDA"
MUTED = "#5E6E77"


def _contours() -> str:
    """Topographic contour lines for the hero background (inline SVG data URI)."""
    paths = []
    for k in range(9):
        r = 60 + k * 34
        pts = []
        for i in range(73):
            a = 2 * math.pi * i / 72
            wob = 1 + 0.10 * math.sin(3 * a + k * 0.7) + 0.06 * math.sin(5 * a - k)
            pts.append(f"{780 + r * 1.35 * wob * math.cos(a):.1f},{170 + r * 0.8 * wob * math.sin(a):.1f}")
        paths.append(f"<path d='M{' L'.join(pts)} Z'/>")
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1100 420' preserveAspectRatio='xMidYMid slice'>"
           "<g fill='none' stroke='%232F5D8A' stroke-opacity='0.16' stroke-width='1.2'>" + "".join(paths) +
           "</g></svg>")
    return "data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E")


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th, .stApp [data-testid="stMetricValue"] {{
  font-family: 'Archivo', system-ui, sans-serif;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 4.2rem; padding-bottom: 4rem; }}
.stApp h1, .stApp h2, .stApp h3 {{ font-stretch: 78%; font-weight: 760; letter-spacing: -0.005em; color: {INK}; }}
.stApp h1 {{ font-size: 2.6rem; line-height: 1.05; }}
.stApp h2 {{ font-size: 1.75rem; line-height: 1.15; }}
.stApp h3 {{ font-size: 1.2rem; }}
.ds-num, .ds-price, .ds-gap-lab, [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
.ds-muted {{ color: {MUTED}; }}
.ds-lede {{ font-size: 1.15rem; line-height: 1.55; color: {MUTED}; max-width: 34em; }}

/* hero */
.ds-hero {{ position: relative; margin: 0 -1rem 1.2rem; padding: 3.2rem 1rem 2.4rem;
  background: url("{_contours()}") right center / cover no-repeat; border-bottom: 1px solid {LINE}; }}
.ds-hero h1 {{ font-size: clamp(2.3rem, 5.2vw, 4.1rem); font-stretch: 72%; font-weight: 800; line-height: 0.98;
  max-width: 11em; margin: 0 0 1rem; }}
.ds-brand {{ font-stretch: 90%; font-weight: 700; color: {SURVEY}; font-size: 1.05rem; margin-bottom: 1.4rem; }}

/* property card content */
.ds-card {{ display: flex; flex-direction: column; gap: 0.35rem; }}
.ds-elev {{ width: 100%; height: auto; display: block; border: 1px solid {LINE}; border-radius: 3px;
  background: #EEF2F1; }}
.ds-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem;
  margin-top: 0.45rem; font-size: 0.86rem; }}
.ds-badge {{ display: inline-flex; align-items: center; gap: 0.4rem; font-weight: 650; }}
.ds-badge i {{ width: 0.62rem; height: 0.62rem; border-radius: 2px; display: inline-block; }}
.ds-hood {{ color: {MUTED}; }}
.ds-price {{ font-size: 1.65rem; font-weight: 760; font-stretch: 80%; line-height: 1.1; }}
.ds-addr {{ color: {INK}; font-weight: 500; }}
.ds-specs {{ display: flex; flex-wrap: wrap; gap: 0.2rem 0.9rem; color: {MUTED}; font-size: 0.9rem; }}
.ds-flags {{ color: {RED}; font-size: 0.86rem; }}

/* the price gap bar: fair-value range as a band, asking price as a dot */
.ds-gap {{ position: relative; height: 58px; margin: 0.4rem 0 0.2rem; }}
.ds-gap-track {{ position: absolute; left: 0; right: 0; top: 28px; height: 2px; background: {LINE}; }}
.ds-gap-band {{ position: absolute; top: 22px; height: 14px; background: {SURVEY_LIGHT};
  border-left: 2px solid {SURVEY}; border-right: 2px solid {SURVEY}; }}
.ds-gap-tick {{ position: absolute; top: 18px; width: 2px; height: 22px; background: {SURVEY}; }}
.ds-gap-dot {{ position: absolute; top: 21px; width: 16px; height: 16px; border-radius: 50%;
  border: 2px solid #fff; box-shadow: 0 0 0 1px {INK}; transform: translateX(-50%); }}
.ds-gap-lab {{ position: absolute; font-size: 0.78rem; white-space: nowrap; }}
.ds-gap-lab.ask {{ top: 0; font-weight: 650; }}
.ds-gap-lab.est {{ top: 41px; color: {SURVEY}; }}

/* how it works */
.ds-step {{ border-top: 2px solid {INK}; padding-top: 0.7rem; }}
.ds-step b {{ display: block; font-stretch: 80%; font-size: 1.05rem; margin-bottom: 0.25rem; }}
.ds-step span {{ color: {MUTED}; font-size: 0.95rem; line-height: 1.5; }}
.ds-step em {{ font-style: normal; font-weight: 800; color: {SURVEY}; margin-right: 0.35rem; }}

.ds-foot {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {LINE}; color: {MUTED}; font-size: 0.85rem; }}
.ds-empty {{ border: 1px dashed {LINE}; border-radius: 6px; padding: 2rem 1.5rem; }}

/* streamlit chrome */
.stButton button, .stDownloadButton button {{ border-radius: 4px; font-weight: 600; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 6px; }}
footer {{ visibility: hidden; }}
/* phones */
@media (max-width: 640px) {{
  .block-container {{ padding: 3.6rem 1rem 3rem; }}
  .ds-hero {{ margin: 0 -1rem 1rem; padding: 1.6rem 1rem 1.6rem; background-size: 220% auto; }}
  .ds-hero h1 {{ font-size: 2.25rem; }}
  .ds-lede {{ font-size: 1.02rem; }}
  .stApp h1 {{ font-size: 1.85rem; }}
  .stApp h2 {{ font-size: 1.4rem; }}
  .ds-price {{ font-size: 1.45rem; }}
  .ds-step {{ margin-bottom: 0.4rem; }}
  /* inside property cards keep the Details and Save buttons on one row */
  [class*="st-key-card-"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: 0.5rem; }}
  [class*="st-key-card-"] [data-testid="stColumn"] {{ min-width: 0 !important; width: auto !important;
    flex: 1 1 0 !important; }}
  /* bigger touch targets */
  .stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{ min-height: 2.75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
}}
</style>
"""
