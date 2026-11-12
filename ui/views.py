"""The five pages of the site. Each function draws one page."""
from __future__ import annotations

import io
import math

import altair as alt
import pandas as pd
import streamlit as st

from dealfinder import services
from ui import components as c
from ui import state
from ui.theme import BRAND, INK, LINE, SURVEY, TAGLINE

FILTER_VERDICTS = ["strong deal", "deal", "speculative", "value trap", "fair", "overpriced", "insufficient data"]
DEFAULT_SHOWN = ["strong deal", "deal", "speculative"]
PAGE_SIZE = 12


def _html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def _go(page: str) -> None:
    st.switch_page(state.PAGES[page])


def _footer() -> None:
    _html(f"<div class='ds-foot'>{BRAND} gives estimates, not appraisals or financial advice. Check any home "
          "with an inspection and recent nearby sales before you make an offer. The sample market is simulated "
          "for demonstration.</div>")


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ------------------------------------------------------------- shared bits ---
def details(r: dict, notes: dict | None = None) -> None:
    st.markdown(f"**{r['address']}**, {r['neighborhood']}")
    st.write(r["explanation"])
    note = (notes or {}).get(r["id"])
    if note and str(r.get("parsed_by", "")).startswith("llm"):
        st.markdown("**Analyst note**")
        st.write(note.get("rationale", ""))
    if note and note.get("next_steps"):
        st.markdown(f"**Before you offer:** {note['next_steps']}")
    if r.get("description"):
        st.markdown("**From the listing**")
        st.caption(r["description"])
    facts = []
    if _ok(r.get("lot_sqft")):
        facts.append(f"Lot {r['lot_sqft']:,.0f} sq ft")
    if r.get("condition"):
        facts.append(f"Condition: {str(r['condition'])}")
    if r.get("renovated"):
        facts.append("Renovated")
    if r.get("garage_spaces"):
        facts.append(f"{int(r['garage_spaces'])}-car garage")
    if r.get("pool"):
        facts.append("Pool")
    if r.get("view"):
        facts.append("View")
    facts.append(f"Listed {int(r.get('days_on_market') or 0)} days")
    st.markdown("\n".join(f"- {f}" for f in facts))
    reader = "AI" if str(r.get("parsed_by", "")).startswith("llm") else "built-in parser"
    st.caption(f"Confidence: {r.get('confidence') or 'n/a'}. Description read by the {reader}.")


def save_button(r: dict, key: str) -> None:
    saved = state.is_saved(r["id"])
    if st.button("Saved" if saved else "Save", key=key, width="stretch",
                 icon=":material/bookmark:" if saved else ":material/bookmark_border:",
                 help="Remove from your watchlist" if saved else "Add to your watchlist"):
        state.toggle_save(r)
        st.rerun()


def card_grid(frame: pd.DataFrame, key: str, cols: int = 3, notes: dict | None = None) -> None:
    rows = [r for _, r in frame.iterrows()]
    for start in range(0, len(rows), cols):
        for col, r in zip(st.columns(cols, gap="medium"), rows[start:start + cols]):
            d = r.to_dict()
            with col, st.container(border=True):
                _html(c.card_html(d))
                b1, b2 = st.columns(2)
                with b1.popover("Details"):
                    details(d, notes)
                with b2:
                    save_button(d, f"{key}-save-{d['id']}")


def legend() -> None:
    _html("<div class='ds-specs' style='margin:.2rem 0 .6rem'>" + "".join(
        f"<span class='ds-badge' style='font-weight:500'><i style='background:{services.verdict_color(v)}'></i>"
        f"{services.verdict_label(v)}</span>" for v in FILTER_VERDICTS[:6]) + "</div>")


# -------------------------------------------------------------------- home ---
def home() -> None:
    f = state.frame()
    deals = f[f["verdict"].isin(services.DEAL_VERDICTS)].sort_values("discount", ascending=False)
    traps = f[f["verdict"] == "value trap"].sort_values("discount", ascending=False)
    ex = deals.iloc[0].to_dict() if len(deals) else (f.iloc[0].to_dict() if len(f) else None)

    example = ""
    if ex:
        color = services.verdict_color(ex["verdict"])
        example = (f"<div style='background:#fff;border:1px solid {LINE};border-radius:6px;padding:14px 16px'>"
                   f"<img class='ds-elev' src='{c.elevation_for(ex)}' alt=''/>"
                   f"<div class='ds-top'><span class='ds-badge'><i style='background:{color}'></i>"
                   f"{services.verdict_label(ex['verdict'])}</span><span class='ds-hood'>{ex['neighborhood']}</span></div>"
                   f"<div class='ds-price'>${ex['asking_price']:,.0f}</div>"
                   f"<div class='ds-addr'>{ex['address']}</div>"
                   + c.gap_bar(ex["asking_price"], ex["fair_low"], ex["fair_high"], ex["fair_value"], color)
                   + "</div>")
    _html(f"""<div class='ds-hero'><div style='display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);
          gap:2.5rem;align-items:center' class='ds-hero-grid'>
          <div><div class='ds-brand'>{BRAND}</div><h1>{TAGLINE}</h1>
          <p class='ds-lede'>{BRAND} reads every listing, estimates what the home is worth from thousands of past
          sales, and shows you the ones priced below that, with the risks spelled out.</p></div>
          <div>{example}</div></div></div>
          <style>@media (max-width:760px){{.ds-hero-grid{{grid-template-columns:1fr !important}}}}</style>""")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Browse deals", type="primary", width="stretch"):
        _go("deals")
    if b2.button("Check a price", width="stretch"):
        _go("check")

    n_spec = int((f["verdict"] == "speculative").sum())
    st.markdown(f"#### Today we checked {len(f)} homes. {len(deals)} are listed well below our estimate "
                f"with no disclosed problems, and {n_spec + len(traps)} look cheap but come with problems "
                "you should know about.")

    st.markdown("## How it works")
    steps = [
        ("Read the listing", "Messy descriptions become clean facts: bedrooms, living area, condition, and any "
                             "problems the seller disclosed."),
        ("Estimate the value", f"A model trained on {len(pd.read_csv(services.SALES_PATH)):,} past sales "
                               "prices the home and gives a likely range, not just one number."),
        ("Account for risk", "Foundation trouble, flood zones, tenants or unpermitted work lower the value "
                             "before we compare it with the asking price."),
        ("Rank the gaps", "Homes listed below the likely range, with nothing alarming disclosed, rise to "
                          "the top."),
    ]
    for col, (i, (t, d)) in zip(st.columns(4, gap="large"), enumerate(steps, 1)):
        col.markdown(f"<div class='ds-step'><b><em>{i}</em>{t}</b><span>{d}</span></div>",
                     unsafe_allow_html=True)

    st.markdown("## Biggest gaps right now")
    if len(deals):
        # the hero already shows the best one; start from the next when there are enough
        card_grid(deals.iloc[1:4] if len(deals) > 3 else deals.head(3), "home", notes=state.report_notes())
        if st.button(f"See all {len(deals)} homes below value"):
            _go("deals")
    else:
        st.write("No homes are listed below value today.")

    if len(traps):
        st.markdown("## Cheap for a reason")
        left, right = st.columns([1, 1.4], gap="large")
        t = traps.iloc[0].to_dict()
        with left, st.container(border=True):
            _html(c.card_html(t))
        right.write("Some listings look like bargains until you read the small print. When a seller "
                    "discloses something like foundation trouble or a cash-only sale, we lower the value "
                    "to account for it. If that explains the low price, we say so instead of calling it a deal.")
        right.write(t["explanation"])
    _footer()


# ------------------------------------------------------------------- deals ---
def _data_source_panel(m: dict) -> None:
    with st.expander("Change listings or run the AI analyst", icon=":material/tune:"):
        src = st.radio("Listings to scan", ["Sample market", "My own CSV"], horizontal=True,
                       index=0 if m["source"] == state.SAMPLE_SOURCE else 1)
        if src == "My own CSV":
            template = pd.DataFrame([{
                "listing_id": "A1", "address": "12 Oak St", "neighborhood": "Riverside", "asking_price": 325000,
                "days_on_market": 14, "lat": "", "lon": "",
                "description": "3BR/2BA, 1,650 sq ft, built 1988, 2-car garage, needs TLC",
            }])
            st.download_button("Download the CSV template", template.to_csv(index=False), "listings_template.csv",
                               "text/csv", icon=":material/download:")
            st.caption("Each row needs a description and an asking price. Optional: listing_id, address, "
                       f"neighborhood ({', '.join(services.neighborhoods())}), days_on_market, lat, lon.")
            up = st.file_uploader("Your listings (CSV)", type="csv")
            if up is not None and st.button("Scan my listings", type="primary"):
                try:
                    listings = services.prepare_listings(pd.read_csv(io.BytesIO(up.getvalue())))
                except Exception as e:
                    st.error(f"That file can't be scanned: {e}")
                    return
                with st.spinner(f"Scanning {len(listings)} listings"):
                    state.scan(listings, f"Your file: {up.name}", use_ai=False)
                st.rerun()
        elif m["source"] != state.SAMPLE_SOURCE:  # picking "Sample market" switches straight back
            state.reset_to_sample()
            st.rerun()

        st.divider()
        st.markdown("**AI analyst**")
        if not state.ai_available():
            st.caption("Not set up on this site. Add a GEMINI_API_KEY to switch it on (see the README).")
        elif state.ai_runs() >= state.AI_RUNS_PER_VISIT:
            st.caption("You've used the AI analyst the maximum number of times for this visit. "
                       "Reload the page later to run it again.")
        else:
            st.caption("The AI re-reads every description and decides which homes deserve a closer look. "
                       "It takes about a minute on the free plan. If it's busy, the quick scan is used instead.")
            if st.button("Run the AI analyst", icon=":material/auto_awesome:"):
                with st.status("AI analyst working", expanded=True) as status:
                    res = state.scan(m["listings"], m["source"], use_ai=True, on_event=st.write)
                    status.update(label=res["mode"], state="complete" if res["ai"] else "error", expanded=False)
                st.rerun()


def deals() -> None:
    m = state.market()
    f = state.frame()
    st.title("Homes priced below what they're worth")
    st.caption(f"{m['source']}. {m['mode']}.")
    _data_source_panel(m)

    if f.empty:
        _html("<div class='ds-empty'>No listings to show yet. Upload a CSV or switch back to the sample market.</div>")
        return

    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1, 1.4])
    hoods = c1.multiselect("Neighborhood", sorted(f["neighborhood"].unique()), placeholder="All neighborhoods")
    lo = int(f["asking_price"].min() // 10_000 * 10_000)
    hi = int(-(-f["asking_price"].max() // 10_000) * 10_000)
    price = c2.slider("Asking price", lo, hi, (lo, hi), step=10_000, format="$%d") if hi > lo else (lo, hi)
    beds = c3.selectbox("Bedrooms", ["Any", "1+", "2+", "3+", "4+", "5+"])
    sort = c4.selectbox("Sort by", ["Biggest gap", "Lowest price", "Highest price", "Newest listings"])
    present = [v for v in FILTER_VERDICTS if v in set(f["verdict"])]
    shown = st.pills("Show", present, default=[v for v in DEFAULT_SHOWN if v in present],
                     selection_mode="multi", format_func=services.verdict_label)

    view = f.copy()
    if hoods:
        view = view[view["neighborhood"].isin(hoods)]
    view = view[view["asking_price"].between(*price)]
    if beds != "Any":
        view = view[view["beds"].fillna(0) >= int(beds[0])]
    view = view[view["verdict"].isin(shown or [])]
    order = {"Biggest gap": ("discount", False), "Lowest price": ("asking_price", True),
             "Highest price": ("asking_price", False), "Newest listings": ("days_on_market", True)}[sort]
    view = view.sort_values(order[0], ascending=order[1], na_position="last")

    top = st.columns([3, 1.3])
    top[0].markdown(f"**{len(view)} of {len(f)} homes** match.")
    mode = top[1].segmented_control("View", ["Cards", "Map", "Table"], default="Cards",
                                    label_visibility="collapsed", key="deals_view") or "Cards"

    if view.empty:
        _html("<div class='ds-empty'>No homes match these filters. Widen the price range, pick more "
              "neighborhoods, or choose more categories under Show.</div>")
    elif mode == "Cards":
        n = st.session_state.get("deals_shown", PAGE_SIZE)
        card_grid(view.head(n), "deals", notes=state.report_notes())
        if len(view) > n and st.button(f"Show more ({len(view) - n} left)"):
            st.session_state.deals_shown = n + PAGE_SIZE
            st.rerun()
    elif mode == "Map":
        deck = c.map_deck(view)
        if deck is None:
            st.info("None of these homes has a location. Add lat and lon columns to your CSV to map them.")
        else:
            legend()
            st.pydeck_chart(deck, height=560)
            missing = int(view[["lat", "lon"]].isna().any(axis=1).sum())
            st.caption("Hover a pin for details. Pin positions in the sample market are illustrative."
                       + (f" {missing} homes without a location aren't shown." if missing else ""))
    else:
        cols = {"address": "Address", "neighborhood": "Neighborhood", "verdict_label": "Verdict",
                "asking_price": "Asking", "fair_value": "Our estimate", "fair_low": "Low", "fair_high": "High",
                "discount": "Gap", "beds": "Beds", "baths": "Baths", "sqft": "Sq ft", "year_built": "Built",
                "red_flags": "Disclosed problems", "days_on_market": "Days listed", "confidence": "Confidence"}
        t = view[list(cols)].rename(columns=cols)
        money = {k: st.column_config.NumberColumn(k, format="$%d") for k in ("Asking", "Our estimate", "Low", "High")}
        st.dataframe(t, hide_index=True, width="stretch", height=520,
                     column_config={**money, "Gap": st.column_config.NumberColumn("Gap", format="percent",
                                                                                  help="How far the asking price is "
                                                                                       "below our estimate"),
                                    "Built": st.column_config.NumberColumn("Built", format="%d")})

    st.divider()
    res = m["result"]
    d1, d2, d3 = st.columns(3)
    d1.download_button("Deal report (web page)", res["html"], "deal_report.html", "text/html", width="stretch",
                       icon=":material/description:")
    d2.download_button("Deal report (text)", res["markdown"], "deal_report.md", "text/markdown", width="stretch",
                       icon=":material/article:")
    d3.download_button("These results (CSV)", view.drop(columns=["explanation"]).to_csv(index=False),
                       "deal_results.csv", "text/csv", width="stretch", icon=":material/table:")
    _footer()


# ------------------------------------------------------------- price check ---
def price_check() -> None:
    st.title("Check a price")
    st.write("Paste a listing and its asking price. We'll pull out the facts, estimate what the home is worth, "
             "and tell you how the price compares.")
    left, right = st.columns([1, 1.05], gap="large")
    defaults = services.neighborhood_defaults()
    with left, st.form("check"):
        desc = st.text_area("Listing description", height=170, value=(
            "Charming 3 bed / 1 full + 1 half bath, approx 1,480 SF on a 0.21 acre lot, circa 1962. "
            "Original kitchen, solid bones. 1-car garage. Offers due Monday."))
        a, b = st.columns(2)
        hood = a.selectbox("Neighborhood", services.neighborhoods())
        ask = b.number_input("Asking price ($)", min_value=10_000, value=265_000, step=5_000)
        with st.expander("Location details (optional)"):
            dflt = defaults.get(hood, {"dist_center_km": 6.0, "school_rating": 6.5})
            x, y = st.columns(2)
            dist = x.number_input("Distance to centre (km)", 0.0, 50.0, float(dflt["dist_center_km"]), 0.5,
                                  help="Leave as is to use the neighborhood's typical value")
            school = y.number_input("School rating (1 to 10)", 1.0, 10.0, float(dflt["school_rating"]), 0.5)
        use_ai = st.toggle("Let the AI read the description", value=False, disabled=not state.ai_available(),
                           help="Slower. The built-in parser handles most listings well.")
        go = st.form_submit_button("Check this price", type="primary", width="stretch")
    if go:
        if not desc.strip():
            left.error("Paste a listing description first.")
        else:
            with st.spinner("Reading the listing and estimating its value"):
                a = services.value_one(desc, int(ask), hood, state.get_model(), use_ai, dist, school)
            n = st.session_state.get("check_n", 0) + 1
            st.session_state.check_n = n
            row = services.analyses_frame([{**a, "id": f"CHECK-{n}", "address": f"Price check {n}"}]).iloc[0]
            st.session_state.check_row = row.to_dict()
    with right:
        r = st.session_state.get("check_row")
        if not r:
            _html("<div class='ds-empty'>Your result appears here: the asking price against our likely range, "
                  "what we read from the listing, and anything that lowers the value.</div>")
            return
        if r["verdict"] == "insufficient data":
            st.warning(r["explanation"] + " Add those details to the description and check again.")
            return
        with st.container(border=True):
            _html(c.card_html(r))
            st.write(r["explanation"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Our estimate", services.money_short(r["fair_value"]))
            k2.metric("Likely range", f"{services.money_short(r['fair_low'])} to {services.money_short(r['fair_high'])}")
            k3.metric("Value after risks", services.money_short(r["risk_adjusted_value"]))
            save_button(r, f"check-save-{r['id']}")
        with st.expander("What we read from the listing"):
            details(r)


# --------------------------------------------------------------- watchlist ---
def watchlist() -> None:
    st.title("Your watchlist")
    w = state.watch_frame()
    if w.empty:
        _html("<div class='ds-empty'><b>You haven't saved any homes yet.</b><br/>Save homes from the deals page "
              "or a price check, then compare them side by side here.</div>")
        st.write("")
        if st.button("Browse deals", type="primary"):
            _go("deals")
        return
    st.caption("Saved homes stay while this tab is open. Download the list to keep a copy.")
    card_grid(w, "watch", notes=state.report_notes())

    st.markdown("## Compare side by side")
    labels = {r["id"]: f"{r['address']} ({r['id']})" for _, r in w.iterrows()}
    pick = st.multiselect("Homes to compare (up to 4)", list(labels), default=list(labels)[:3],
                          format_func=labels.get, max_selections=4)
    if len(pick) < 2:
        st.info("Pick at least two homes to compare.")
    else:
        sub = w.set_index("id").loc[pick].reset_index()
        for col, (_, r) in zip(st.columns(len(sub), gap="medium"), sub.iterrows()):
            d = r.to_dict()
            color = services.verdict_color(d["verdict"])
            col.markdown(f"<img class='ds-elev' src='{c.elevation_for(d)}' alt=''/>"
                         f"<div class='ds-price' style='font-size:1.25rem;margin-top:.4rem'>${d['asking_price']:,.0f}</div>"
                         f"<div class='ds-addr'>{d['address']}</div>"
                         + c.gap_bar(d["asking_price"], d["fair_low"], d["fair_high"], d["fair_value"], color),
                         unsafe_allow_html=True)
        st.dataframe(services.compare_frame(sub), width="stretch", height=36 * len(services.COMPARE_FIELDS) + 40)
        best = sub.loc[sub["discount"].fillna(-9).idxmax()]
        st.write(f"Biggest gap: **{best['address']}**, listed {best['discount']:.0%} below our estimate"
                 + (f", but it discloses {best['red_flags']}." if best["red_flags"] else
                    " with no disclosed problems."))
    st.download_button("Download my watchlist (CSV)", w.drop(columns=["explanation"], errors="ignore").to_csv(index=False),
                       "watchlist.csv", "text/csv", icon=":material/download:")
    _footer()


# ------------------------------------------------------------ how it works ---
def method() -> None:
    model = state.get_model()
    mt = model.metrics
    st.title("How the estimates work")
    st.write(f"{BRAND} combines three pieces: a parser that reads listing text, a price model trained on past "
             "sales, and an analyst that compares the two and ranks the results.")
    k = st.columns(4)
    k[0].metric("Typical error", f"{mt['MAPE']:.1%}", help="Average gap between our estimate and the actual "
                                                            "sale price, on sales the model never saw")
    k[1].metric("Simple price-per-foot error", f"{mt['baseline_ppsf_MAPE']:.1%}",
                help="The same test for a neighborhood price-per-square-foot rule of thumb")
    k[2].metric("Within 10% of the sale price", f"{mt['within_10pct']:.0%}")
    k[3].metric("Sales inside our range", f"{mt['interval_80_coverage']:.0%}",
                help="Our likely range is built to contain the real price 80% of the time")

    st.markdown("## What the labels mean")
    meaning = {
        "strong deal": "Asking price is at least 15% below our estimate, or below the whole likely range, and the "
                       "listing discloses nothing alarming.",
        "deal": "Asking price is at least 10% below our estimate, with nothing alarming disclosed.",
        "speculative": "The listing discloses problems, but it's still at least 10% below the value after we "
                       "allow for them. Worth a look with careful checks.",
        "value trap": "Looks cheap, but the disclosed problems explain the price.",
        "fair": "Within 10% of our estimate either way.",
        "overpriced": "Asking price is more than 10% above our estimate.",
    }
    for v, text in meaning.items():
        _html(f"<p><span class='ds-badge'><i style='background:{services.verdict_color(v)}'></i>"
              f"{services.verdict_label(v)}</span><br/><span class='ds-muted'>{text}</span></p>")

    st.markdown("## How we allow for risk")
    st.write("When a listing discloses a problem, we lower its value before comparing: foundation issues 12%, "
             "cash-only or unfinanceable 8%, flood zone 7%, tenant in place 5%, sold as-is 4%, unpermitted work "
             "4%, roof at end of life 3%, up to 35% in total.")

    st.markdown("## What drives price in the model")
    nice = {"hood_x_sqft": "Location and size together", "effective_age": "Age (since last renovation)",
            "condition": "Condition", "garage": "Garage spaces", "hood_ppsf": "Neighborhood price level",
            "school_rating": "School rating", "pool": "Pool", "view": "View", "log_sqft": "Living area",
            "cond_x_age": "Condition for its age", "dist_center_km": "Distance to centre", "log_lot": "Lot size",
            "baths": "Bathrooms", "beds": "Bedrooms", "age": "Age", "renovated": "Renovated",
            "sqft_per_bed": "Space per bedroom", "bath_bed_ratio": "Bathrooms per bedroom",
            "lot_to_sqft": "Lot relative to house"}
    imp = pd.DataFrame({"Factor": [nice.get(k, k) for k in model.importances],
                        "Importance": list(model.importances.values())})
    imp = imp[imp["Importance"] > 0].head(10)
    st.altair_chart(alt.Chart(imp).mark_bar(color=SURVEY, cornerRadiusEnd=2).encode(
        x=alt.X("Importance:Q", title="How much the estimate gets worse without it", axis=alt.Axis(format=".2f")),
        y=alt.Y("Factor:N", sort="-x", title=None),
        tooltip=["Factor", alt.Tooltip("Importance:Q", format=".3f")],
    ).properties(height=320).configure_axis(labelFont="Archivo", titleFont="Archivo", labelColor=INK,
                                            gridColor=LINE).configure_view(stroke=None), width="stretch")

    st.markdown("## Limits")
    st.write(f"Estimates come from {mt['test_rows'] * 5:,} sales in a simulated market. To use {BRAND} for a real "
             "city, train it on that city's recent sales (the README explains how). Listings in neighborhoods the "
             "model hasn't seen get a low confidence rating. Nothing here replaces an inspection, an appraisal or "
             "local advice.")
    _footer()
