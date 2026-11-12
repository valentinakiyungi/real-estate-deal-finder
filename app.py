"""Real-Estate Price & Deal Finder — web app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dealfinder import llm, services

st.set_page_config(page_title="Deal Finder", page_icon="🏠", layout="wide")

VERDICT_COLORS = {
    "strong deal": "#1f7a5c", "deal": "#4fa383", "speculative": "#d08c2a", "value trap": "#c0392b",
    "fair": "#8a8f98", "overpriced": "#5b6170", "insufficient data": "#c4c7cc",
}
EXAMPLE = ("PRICE IMPROVEMENT!! Charming 3 bed / 1 full + 1 half bath, approx 1,480 SF on a 0.21 acre lot, "
           "circa 1962. Original kitchen, solid bones. 1-car garage. Offers due Monday.")


# ------------------------------------------------------------------ loaders --
@st.cache_resource(show_spinner="First run: building data and training the price model (~10 s)…")
def get_model():
    return services.ensure_data_and_model()


def money(v) -> str:
    return f"${v:,.0f}"


def results_frame(analyses: list[dict]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        f = a.get("features", {})
        rows.append({
            "ID": a["id"], "Address": a["address"], "Neighborhood": a["neighborhood"],
            "Verdict": a.get("verdict"), "Asking": a["asking_price"], "Fair value": a.get("fair_value"),
            "Low": a.get("fair_low"), "High": a.get("fair_high"),
            "Discount": a.get("discount_vs_fair"), "Risk-adj. discount": a.get("discount_vs_risk_adjusted"),
            "Beds": f.get("beds"), "Baths": f.get("baths"), "Sqft": f.get("sqft"), "Year": f.get("year_built"),
            "Condition": f.get("condition"), "Red flags": ", ".join(f.get("red_flags", [])),
            "Confidence": a.get("confidence"), "Parsed by": a.get("parser"), "Days listed": a["days_on_market"],
        })
    return pd.DataFrame(rows)


MONEY_COLS = {c: st.column_config.NumberColumn(c, format="$%d") for c in ("Asking", "Fair value", "Low", "High")}
PCT_COLS = {c: st.column_config.NumberColumn(c, format="percent") for c in ("Discount", "Risk-adj. discount")}


# ------------------------------------------------------------------- sidebar --
model = get_model()
with st.sidebar:
    st.title("🏠 Deal Finder")
    st.caption("ML price model · GenAI listing parser · deal-hunting agent")
    if llm.available():
        st.success(f"AI connected: {llm.describe()}")
    else:
        st.info("AI is off — running offline.\n\nTo turn it on, create a file named `.env` in the project "
                "folder containing:\n\n`GEMINI_API_KEY=your-key`\n\nthen restart the app. Free keys: "
                "aistudio.google.com/apikey")
    use_ai = st.toggle("Use AI for parsing and the agent", value=llm.available(), disabled=not llm.available(),
                       help="Off = fast rule-based parser and a fixed planner. On = Gemini/Claude reads each "
                            "listing and decides how to search. The free tier is slower (rate limits).")
    st.divider()
    m = model.metrics
    st.metric("Model error (MAPE)", f"{m['MAPE']:.1%}", f"{m['MAPE'] - m['baseline_ppsf_MAPE']:.1%} vs baseline",
              delta_color="inverse")
    if st.button("Retrain model", help="Regenerates the training data and retrains (~10 s)"):
        get_model.clear()
        services.ensure_data_and_model(force=True)
        st.session_state.pop("result", None)
        st.rerun()

tab_find, tab_one, tab_model = st.tabs(["🔎 Find deals", "🏷️ Value a listing", "📊 Model"])

# ------------------------------------------------------------- find deals ----
with tab_find:
    st.header("Find under-priced homes")
    src = st.radio("Listings", ["Sample market (60 listings)", "Upload my own CSV"], horizontal=True)
    listings = None
    if src.startswith("Sample"):
        listings = services.default_listings()
    else:
        template = pd.DataFrame([{
            "listing_id": "A1", "address": "12 Oak St", "neighborhood": "Riverside", "asking_price": 325000,
            "days_on_market": 14, "description": "3BR/2BA, 1,650 sq ft, built 1988, 2-car garage, needs TLC",
        }])
        st.download_button("Download CSV template", template.to_csv(index=False), "listings_template.csv",
                           "text/csv")
        st.caption("Required columns: `description`, `asking_price`. Optional: listing_id, address, "
                   f"neighborhood ({', '.join(services.neighborhoods())}), days_on_market, dist_center_km, "
                   "school_rating.")
        up = st.file_uploader("Listings CSV", type="csv")
        if up is not None:
            try:
                listings = services.prepare_listings(pd.read_csv(io.BytesIO(up.getvalue())))
                st.success(f"Loaded {len(listings)} listings.")
            except Exception as e:
                st.error(str(e))

    c1, c2, c3 = st.columns([2, 2, 1])
    hoods = ["All"] + (sorted(listings["neighborhood"].unique()) if listings is not None else [])
    hood = c1.selectbox("Neighborhood", hoods)
    max_price = c2.number_input("Max asking price ($, 0 = no limit)", min_value=0, value=0, step=25_000)
    c3.write("")
    run = c3.button("Run deal finder", type="primary", disabled=listings is None, width="stretch")

    if run and listings is not None:
        filters = {"neighborhood": None if hood == "All" else hood, "max_price": int(max_price) or None}
        with st.status("Agent working…", expanded=True) as status:
            res = services.run_search(listings, model, filters, use_ai, on_event=st.write)
            status.update(label=f"Done — {res['mode']}", state="complete", expanded=False)
        st.session_state["result"] = res

    res = st.session_state.get("result")
    if res:
        tools = res["tools"]
        df = results_frame(list(tools.cache.values()))
        rep = tools.report
        st.caption(f"Mode: {res['mode']}")
        k = st.columns(4)
        k[0].metric("Listings analysed", len(df))
        k[1].metric("Clean deals", len(rep["deals"]))
        k[2].metric("Speculative", int((df["Verdict"] == "speculative").sum()))
        k[3].metric("Value traps", int((df["Verdict"] == "value trap").sum()))
        st.info(rep["market_summary"])

        if df["Fair value"].notna().any():
            plot = df.dropna(subset=["Fair value"])
            hi = float(max(plot["Asking"].max(), plot["Fair value"].max())) * 1.05
            line = alt.Chart(pd.DataFrame({"x": [0, hi]})).mark_line(strokeDash=[4, 4], color="#999").encode(
                x="x:Q", y="x:Q")
            pts = alt.Chart(plot).mark_circle(size=90, opacity=0.85).encode(
                x=alt.X("Fair value:Q", title="Model fair value ($)", scale=alt.Scale(domain=[0, hi])),
                y=alt.Y("Asking:Q", title="Asking price ($)", scale=alt.Scale(domain=[0, hi])),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS),
                                                             range=list(VERDICT_COLORS.values()))),
                tooltip=["ID", "Address", "Verdict", alt.Tooltip("Asking:Q", format="$,.0f"),
                         alt.Tooltip("Fair value:Q", format="$,.0f"), alt.Tooltip("Discount:Q", format=".0%"),
                         "Red flags"])
            st.subheader("Asking price vs fair value")
            st.caption("Points below the dashed line are priced under the model's fair value.")
            st.altair_chart((line + pts).properties(height=380), width="stretch")

        st.subheader("Ranked deals")
        if rep["deals"]:
            order = [d["id"] for d in rep["deals"]]
            ranked = df.set_index("ID").loc[order].reset_index()
            st.dataframe(ranked[["ID", "Address", "Neighborhood", "Verdict", "Asking", "Fair value", "Low", "High",
                                 "Discount", "Beds", "Baths", "Sqft", "Year", "Confidence"]],
                         column_config={**MONEY_COLS, **PCT_COLS}, hide_index=True, width="stretch")
            for i, d in enumerate(rep["deals"], 1):
                with st.expander(f"{i}. {d['id']} — {d['headline']}"):
                    st.write(d["rationale"])
                    if d.get("next_steps"):
                        st.caption(f"Next steps: {d['next_steps']}")
                    st.code(tools.listings.at[d["id"], "description"], language=None, wrap_lines=True)
        else:
            st.write("No clean deals matched these criteria.")

        for verdict, title, note in [
            ("speculative", "Speculative", "Still cheap after pricing in their risks — worth a look with due diligence."),
            ("value trap", "Value traps", "Look cheap, but the disclosed problems explain the discount."),
        ]:
            sub = df[df["Verdict"] == verdict]
            if len(sub):
                st.subheader(title)
                st.caption(note)
                st.dataframe(sub[["ID", "Address", "Asking", "Fair value", "Discount", "Risk-adj. discount",
                                  "Red flags"]], column_config={**MONEY_COLS, **PCT_COLS}, hide_index=True,
                             width="stretch")

        with st.expander("All analysed listings"):
            st.dataframe(df, column_config={**MONEY_COLS, **PCT_COLS}, hide_index=True, width="stretch")
        with st.expander("Agent log"):
            st.code("\n".join(tools.log), language=None)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Report (HTML)", res["html"], "deal_report.html", "text/html", width="stretch")
        d2.download_button("Report (Markdown)", res["markdown"], "deal_report.md", "text/markdown",
                           width="stretch")
        d3.download_button("All results (CSV)", df.to_csv(index=False), "deal_results.csv", "text/csv",
                           width="stretch")
        with st.expander("Preview full report"):
            components.html(res["html"], height=900, scrolling=True)

# ---------------------------------------------------------- value one listing --
with tab_one:
    st.header("Value a single listing")
    st.caption("Paste any listing text. The parser extracts the facts, the model prices it, "
               "and red flags get a risk haircut.")
    desc = st.text_area("Listing description", EXAMPLE, height=140)
    defaults = services.neighborhood_defaults()
    c1, c2 = st.columns(2)
    hood1 = c1.selectbox("Neighborhood", services.neighborhoods(), key="hood1")
    ask = c2.number_input("Asking price ($)", min_value=10_000, value=265_000, step=5_000)
    with st.expander("Location details (optional)"):
        dflt = defaults.get(hood1, {"dist_center_km": 6.0, "school_rating": 6.5})
        e1, e2 = st.columns(2)
        dist = e1.number_input("Distance to centre (km)", 0.0, 50.0, float(dflt["dist_center_km"]), 0.5)
        school = e2.number_input("School rating (1–10)", 1.0, 10.0, float(dflt["school_rating"]), 0.5)

    if st.button("Estimate value", type="primary") and desc.strip():
        with st.spinner("Parsing and valuing…"):
            st.session_state["one"] = services.value_one(desc, int(ask), hood1, model, use_ai, dist, school)
    a = st.session_state.get("one")
    if a:
        f = a["features"]
        if a.get("verdict") == "insufficient data":
            st.warning(f"Couldn't find: {', '.join(a['missing'])}. Add them to the description and try again.")
        else:
            color = VERDICT_COLORS.get(a["verdict"], "#888")
            st.markdown(f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>{a['verdict'].title()}</div>",
                        unsafe_allow_html=True)
            k = st.columns(4)
            k[0].metric("Asking", money(a["asking_price"]))
            k[1].metric("Fair value", money(a["fair_value"]), f"{-a['discount_vs_fair']:.0%} asking vs fair",
                        delta_color="inverse")
            k[2].metric("80% range", f"{money(a['fair_low'])} – {money(a['fair_high'])}")
            k[3].metric("Risk-adjusted value", money(a["risk_adjusted_value"]),
                        f"-{a['risk_haircut']:.0%} haircut" if a["risk_haircut"] else None, delta_color="off")
            if f["red_flags"]:
                st.error("Red flags: " + ", ".join(f["red_flags"]))
            st.caption(f"Confidence: {a['confidence']} · parsed by: {a['parser']}"
                       + (f" · AI and rules disagreed on: {', '.join(a['parser_disagreements'])}"
                          if a["parser_disagreements"] else ""))
        st.subheader("What the parser extracted")
        st.dataframe(pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in f.items()]), hide_index=True,
                     width="stretch")

# ------------------------------------------------------------------ model -----
with tab_model:
    st.header("The price model")
    m = model.metrics
    k = st.columns(4)
    k[0].metric("Average error (MAPE)", f"{m['MAPE']:.1%}")
    k[1].metric("Simple $/sqft baseline", f"{m['baseline_ppsf_MAPE']:.1%}")
    k[2].metric("Within 10% of sale price", f"{m['within_10pct']:.0%}")
    k[3].metric("80% range coverage", f"{m['interval_80_coverage']:.0%}")
    st.write(f"Gradient-boosted trees on log price, tested on {m['test_rows']:,} past sales it never saw. "
             "Two extra quantile models, calibrated with conformal prediction, give the 80% fair-value range "
             "that the deal rules rely on.")
    imp = pd.DataFrame({"Feature": list(model.importances), "Importance": list(model.importances.values())})
    imp = imp[imp["Importance"] > 0].head(12)
    st.subheader("What drives price")
    st.altair_chart(alt.Chart(imp).mark_bar(color="#1f7a5c").encode(
        x=alt.X("Importance:Q", title="Permutation importance"), y=alt.Y("Feature:N", sort="-x", title=None)
    ).properties(height=320), width="stretch")
    st.caption("Estimates only — not financial advice. Trained on a synthetic market; "
               "see the README for using real data.")
