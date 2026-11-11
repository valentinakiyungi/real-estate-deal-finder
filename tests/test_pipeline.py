import pandas as pd
import pytest

from dealfinder import data
from dealfinder.agent import DealTools, run_rule_agent
from dealfinder.model import PriceModel
from dealfinder.parser import parse_rules


@pytest.mark.parametrize("text,expect", [
    ("3bd / 2.5BA, approx 1,650 SF, built 1988, 2-car garage, 7,000 sf lot",
     dict(beds=3, baths=2.5, sqft=1650, year_built=1988, garage_spaces=2, lot_sqft=7000)),
    ("three bedrooms / 1 full + 1 half bath, 1760 sqft lot, approx 2,110 SF, circa 1948",
     dict(beds=3, baths=1.5, sqft=2110, lot_sqft=1760, year_built=1948)),
    ("2BR | 1.2k sq. ft. of living space | lot 0.25 acres | needs TLC | cash only",
     dict(beds=2, sqft=1200, lot_sqft=10890, condition="poor")),
])
def test_rule_parser(text, expect):
    f = parse_rules(text).model_dump()
    for k, v in expect.items():
        assert f[k] == v, (k, f[k], v)


def test_red_flags():
    f = parse_rules("4bd, 2000 sqft, foundation issues disclosed, in a flood zone")
    assert {"foundation issues", "flood zone"} <= set(f.red_flags)


@pytest.fixture(scope="module")
def model():
    return PriceModel().fit(data.historical_sales(n=1500, seed=3))


def test_model_beats_noise(model):
    test = data.historical_sales(n=400, seed=99)
    p = model.predict(test)["fair_value"]
    mape = ((p - test["sale_price"]).abs() / test["sale_price"]).mean()
    assert mape < 0.12


def test_agent_never_ranks_red_flags(model):
    tools = DealTools(data.active_listings(n=40, seed=5), model, use_llm_parser=False)
    run_rule_agent(tools, {})
    for d in tools.report["deals"]:
        assert not tools.cache[d["id"]]["features"]["red_flags"]
