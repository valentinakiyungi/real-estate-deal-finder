"""Reusable visual pieces: property elevations, the price gap bar, cards and the map."""
from __future__ import annotations

import base64
import html
import math

import pandas as pd

from dealfinder import services
from ui.theme import INK, LINE, SURVEY

e = html.escape


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------- house elevation ---
def house_svg(beds=3, sqft=1500, year_built=1990, garage_spaces=0, pool=False, view=False,
              lot_sqft=5000, accent="#1E7F5C") -> str:
    """Line drawing of the home's front elevation, generated from its features."""
    beds = int(beds) if _ok(beds) else 3
    sqft = float(sqft) if _ok(sqft) else 1500
    year = int(year_built) if _ok(year_built) else 1990
    g = int(garage_spaces) if _ok(garage_spaces) else 0
    lot = float(lot_sqft) if _ok(lot_sqft) else 5000
    ground, story_h = 118, 34
    stories = 2 if sqft >= 2000 else 1
    w = 80 + min(beds, 6) * 14
    gw = 22 + 16 * min(g, 3) if g else 0
    total = w + gw + (42 if pool else 0)
    x0 = round((320 - total) / 2 + 12)  # centre the house (trees sit to its left)
    top = ground - stories * story_h
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 6 320 124'>",
         "<rect width='320' height='140' fill='#EEF2F1'/>",
         "<g stroke='#DDE5E3' stroke-width='0.6'>"
         + "".join(f"<line x1='{x}' y1='0' x2='{x}' y2='140'/>" for x in range(0, 321, 16))
         + "".join(f"<line x1='0' y1='{y}' x2='320' y2='{y}'/>" for y in range(0, 141, 16)) + "</g>"]
    if view:
        s.append(f"<path d='M0 92 Q60 58 120 84 T240 70 T320 80' fill='none' stroke='{SURVEY}' "
                 "stroke-opacity='0.45' stroke-width='1.2'/>")
    st = f"stroke='{INK}' stroke-width='1.4' stroke-linejoin='round'"
    # trees: more lot, more trees
    for k in range(1 if lot < 6000 else 2):
        tx, tr = (x0 - 26, 11) if k == 0 else (x0 - 12, 8)
        s.append(f"<line x1='{tx}' y1='{ground}' x2='{tx}' y2='{ground - 14}' {st}/>"
                 f"<circle cx='{tx}' cy='{ground - 14 - tr}' r='{tr}' fill='#DCE8E1' {st}/>")
    # body
    s.append(f"<rect x='{x0}' y='{top}' width='{w}' height='{stories * story_h}' fill='#F9FBFA' {st}/>")
    if stories == 2:
        s.append(f"<line x1='{x0}' y1='{top + story_h}' x2='{x0 + w}' y2='{top + story_h}' {st} stroke-opacity='0.5'/>")
    # roof: flat parapet for newer homes, gable otherwise; chimney on older homes
    if year >= 2005:
        s.append(f"<rect x='{x0 - 4}' y='{top - 8}' width='{w + 8}' height='8' fill='#DCE4E2' {st}/>")
    else:
        peak = top - 30
        if year < 1960:
            cx = x0 + w * 0.72
            s.append(f"<rect x='{cx:.0f}' y='{peak + 6}' width='10' height='22' fill='#DCE4E2' {st}/>")
        s.append(f"<path d='M{x0 - 8} {top} L{x0 + w / 2:.0f} {peak} L{x0 + w + 8} {top} Z' fill='#DCE4E2' {st}/>")
    # door (in the verdict colour) and windows
    door_x = x0 + w / 2 - 7
    s.append(f"<rect x='{door_x:.0f}' y='{ground - 24}' width='14' height='24' fill='{accent}' {st}/>")
    per_row = max(2, min(beds + 1, 5))
    for row in range(stories):
        wy = ground - story_h * (row + 1) + 9
        for i in range(per_row):
            wx = x0 + (i + 0.5) * w / per_row - 7
            if row == 0 and abs(wx + 7 - (x0 + w / 2)) < 16:
                continue  # leave room for the door
            s.append(f"<rect x='{wx:.0f}' y='{wy}' width='14' height='14' fill='#CFE0EA' {st}/>"
                     f"<line x1='{wx + 7:.0f}' y1='{wy}' x2='{wx + 7:.0f}' y2='{wy + 14}' stroke='{INK}' "
                     "stroke-width='0.8'/>")
    gx = x0 + w
    if g:
        s.append(f"<rect x='{gx}' y='{ground - 30}' width='{gw}' height='30' fill='#F9FBFA' {st}/>"
                 f"<rect x='{gx + 6}' y='{ground - 22}' width='{gw - 12}' height='22' fill='#EEF2F1' {st}/>"
                 + "".join(f"<line x1='{gx + 6}' y1='{ground - 22 + k * 5.5:.1f}' x2='{gx + gw - 6}' "
                           f"y2='{ground - 22 + k * 5.5:.1f}' stroke='{INK}' stroke-width='0.7'/>" for k in range(1, 4)))
        gx += gw
    if pool:
        s.append(f"<rect x='{gx + 8}' y='{ground - 5}' width='34' height='5' rx='1' fill='#8FB8D8' {st}/>")
    s.append(f"<line x1='6' y1='{ground}' x2='314' y2='{ground}' stroke='{INK}' stroke-width='1.6'/>")
    s.append("</svg>")
    return "data:image/svg+xml;base64," + base64.b64encode("".join(s).encode()).decode()


def elevation_for(r) -> str:
    return house_svg(r.get("beds"), r.get("sqft"), r.get("year_built"), r.get("garage_spaces"), r.get("pool"),
                     r.get("view"), r.get("lot_sqft"), services.verdict_color(r.get("verdict")))


# ------------------------------------------------------------------ gap bar ---
def gap_bar(asking, low, high, fair, color) -> str:
    if not all(_ok(v) for v in (asking, low, high, fair)):
        return ""
    lo = min(asking, low) * 0.92
    hi = max(asking, high) * 1.04
    pos = lambda v: 100 * (v - lo) / (hi - lo)
    a, l, h, f = pos(asking), pos(low), pos(high), pos(fair)

    def anchored(p: float, left_edge: float, right_edge: float) -> str:
        """Grow the label towards the side with room, so it never spills past the bar."""
        if p <= 50:
            return f"left:{left_edge:.1f}%;transform:translateX(-6px)"
        return f"right:{100 - right_edge:.1f}%;left:auto;transform:translateX(6px)"
    return (f"<div class='ds-gap' role='img' aria-label='Asking {services.money_short(asking)}, our range "
            f"{services.money_short(low)} to {services.money_short(high)}'>"
            "<div class='ds-gap-track'></div>"
            f"<div class='ds-gap-band' style='left:{l:.1f}%;width:{h - l:.1f}%'></div>"
            f"<div class='ds-gap-tick' style='left:{f:.1f}%'></div>"
            f"<div class='ds-gap-dot' style='left:{a:.1f}%;background:{color}'></div>"
            f"<div class='ds-gap-lab ask' style='{anchored(a, a, a)}'>Asking {services.money_short(asking)}</div>"
            f"<div class='ds-gap-lab est' style='{anchored((l + h) / 2, l, h)}'>"
            f"Our range {services.money_short(low)} to {services.money_short(high)}</div></div>")


# --------------------------------------------------------------------- card ---
def specs_html(r) -> str:
    bits = []
    if _ok(r.get("beds")):
        bits.append(f"{int(r['beds'])} bd")
    if _ok(r.get("baths")):
        bits.append(f"{r['baths']:g} ba")
    if _ok(r.get("sqft")):
        bits.append(f"{r['sqft']:,.0f} sq ft")
    if _ok(r.get("year_built")):
        bits.append(f"Built {int(r['year_built'])}")
    return "<div class='ds-specs'>" + "".join(f"<span>{e(b)}</span>" for b in bits) + "</div>"


def card_html(r, show_gap: bool = True) -> str:
    color = services.verdict_color(r.get("verdict"))
    flags = r.get("red_flags") or ""
    return ("<div class='ds-card'>"
            f"<img class='ds-elev' src='{elevation_for(r)}' alt='Drawing of {e(str(r.get('address')))}'/>"
            f"<div class='ds-top'><span class='ds-badge'><i style='background:{color}'></i>"
            f"{e(services.verdict_label(r.get('verdict')))}</span>"
            f"<span class='ds-hood'>{e(str(r.get('neighborhood')))}</span></div>"
            f"<div class='ds-price'>${r['asking_price']:,.0f}</div>"
            f"<div class='ds-addr'>{e(str(r.get('address')))}</div>"
            + specs_html(r)
            + (gap_bar(r.get("asking_price"), r.get("fair_low"), r.get("fair_high"), r.get("fair_value"), color)
               if show_gap else "")
            + (f"<div class='ds-flags'>Disclosed: {e(flags)}</div>" if flags else "")
            + "</div>")


# ---------------------------------------------------------------------- map ---
def _rgb(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [220]


def map_deck(frame: pd.DataFrame):
    import pydeck as pdk
    d = frame.dropna(subset=["lat", "lon"]).copy()
    if d.empty:
        return None
    d["color"] = d["verdict"].map(lambda v: _rgb(services.verdict_color(v)))
    d["price"] = d["asking_price"].map(lambda v: f"${v:,.0f}")
    d["estimate"] = d["fair_value"].map(lambda v: services.money_short(v) if _ok(v) else "n/a")
    d["radius"] = d["verdict"].map(lambda v: 9 if v in services.DEAL_VERDICTS else 6)
    d = d[["lon", "lat", "color", "radius", "address", "neighborhood", "price", "estimate", "verdict_label", "id"]]
    layer = pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                      get_radius="radius", radius_units="pixels", pickable=True, stroked=True,
                      get_line_color=[255, 255, 255], line_width_min_pixels=1.5)
    view = pdk.ViewState(latitude=float(d["lat"].mean()), longitude=float(d["lon"].mean()), zoom=10.6)
    tooltip = {"html": "<b>{address}</b><br/>{neighborhood}<br/>Asking {price}, our estimate {estimate}"
                       "<br/>{verdict_label}",
               "style": {"backgroundColor": INK, "color": "white", "fontFamily": "Archivo, sans-serif",
                         "fontSize": "13px", "padding": "8px 10px", "borderRadius": "4px"}}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style=None)
