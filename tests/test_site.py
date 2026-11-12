"""Tests for the helpers behind the website's cards, map, compare and explanations."""
import math

import pandas as pd
import pytest

from dealfinder import data, services
from dealfinder.agent import DealTools, run_rule_agent
from dealfinder.model import PriceModel
from ui.components import card_html, gap_bar, house_svg


@pytest.fixture(scope="module")
def analysed():
    model = PriceModel().fit(data.historical_sales(n=1500, seed=3))
    listings = data.active_listings(n=30, seed=4)
    tools = DealTools(listings, model, use_llm_parser=False)
    run_rule_agent(tools, {})
    return listings, services.analyses_frame(list(tools.cache.values()), listings)


def test_coordinates_are_stable_and_near_centre(analysed):
    listings, frame = analysed
    again = services.add_coordinates(listings)
    assert frame["lat"].notna().all()
    assert (again.set_index("listing_id").loc[frame["id"], "lat"].to_numpy() == frame["lat"].to_numpy()).all()
    km = ((frame["lat"] - services.MAP_CENTER[0]) * 111).abs()
    assert km.max() < 25


def test_uploaded_coordinates_win_and_unknown_hoods_have_none():
    df = services.prepare_listings(pd.DataFrame({
        "Description": ["3 bed 2 bath 1500 sqft built 1990"] * 2, "Price": ["$300,000", "250000"],
        "Neighbourhood": ["Riverside", "Atlantis"], "Latitude": [51.5, None], "Lng": [-0.12, None]}))
    out = services.add_coordinates(df)
    assert out.loc[0, "lat"] == 51.5 and out.loc[0, "lon"] == -0.12
    assert math.isnan(out.loc[1, "lat"])


def test_explanations_and_labels(analysed):
    _, frame = analysed
    for _, r in frame.iterrows():
        assert r["verdict_label"] and "Asking" in r["explanation"]
        if r["red_flags"]:
            assert "allowing" in r["explanation"]


def test_compare_frame(analysed):
    _, frame = analysed
    cmp = services.compare_frame(frame.head(3))
    assert cmp.shape == (len(services.COMPARE_FIELDS), 3)
    assert cmp.loc["Asking vs estimate"].str.contains("below|above").all()


def test_visual_components(analysed):
    _, frame = analysed
    r = frame.iloc[0].to_dict()
    assert card_html(r).count("ds-gap-dot") == 1
    assert house_svg(beds=None, sqft=float("nan")).startswith("data:image/svg+xml;base64,")
    assert gap_bar(None, 1, 2, 1.5, "#000") == ""
