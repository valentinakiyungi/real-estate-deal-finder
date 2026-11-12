"""Gemini adapter tests against simulated HTTP responses (no network, no key needed)."""
import json

import pytest

from dealfinder import data, llm, parser
from dealfinder.agent import DealTools, run_llm_agent
from dealfinder.model import PriceModel


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEALFINDER_PROVIDER", raising=False)
    sent = []

    def install(responder):
        def fake_post(url, body, headers, retries=5):
            assert "generativelanguage.googleapis.com" in url and headers["x-goog-api-key"] == "test-key"
            sent.append(body)
            return responder(body, len(sent))
        monkeypatch.setattr(llm, "_post", fake_post)
        return sent
    return install


def _call(name, args, sig=None):
    part = {"functionCall": {"name": name, "args": args}}
    if sig:
        part["thoughtSignature"] = sig
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}]}


def test_schema_is_gemini_compatible():
    s = llm.gemini_schema(parser.BATCH_TOOL["input_schema"])
    text = json.dumps(s)
    for bad in ("$defs", "$ref", "anyOf", "title", "default", "additionalProperties"):
        assert bad not in text, bad
    item = s["properties"]["listings"]["items"]
    assert item["properties"]["beds"] == {"description": "Number of bedrooms", "minimum": 0, "maximum": 12,
                                          "type": "INTEGER", "nullable": True}
    assert item["properties"]["condition"]["enum"] == ["poor", "fair", "good", "excellent"]


def test_provider_selection(gemini):
    assert llm.provider() == "gemini" and llm.available()


def test_batch_parse_uses_one_request(gemini):
    items = [(f"L{i}", f"{i % 4 + 1} bed / 2 bath, {1000 + i * 10} sqft, built 1990") for i in range(12)]

    def respond(body, n):
        assert body["toolConfig"]["functionCallingConfig"] == {"mode": "ANY",
                                                               "allowedFunctionNames": ["record_listings"]}
        rows = [{"id": i, **parser.parse_rules(t).model_dump()} for i, t in items]
        rows[0]["beds"] = 9  # simulate one disagreement with the rule parser
        return _call("record_listings", {"listings": rows})

    sent = gemini(respond)
    out = parser.parse_many(items)
    assert len(sent) == 1
    assert out["L0"][1] == {"source": "llm", "disagreements": ["beds"]}
    assert out["L5"][1]["source"] == "llm" and out["L5"][0].sqft == 1050


def test_parse_falls_back_on_api_error(gemini, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("HTTP 429 (rate limit)")
    monkeypatch.setattr(llm, "_post", boom)
    f, meta = parser.parse("3 bed / 2 bath, 1500 sqft, built 1990")
    assert f.beds == 3 and meta["source"].startswith("rules (AI failed")


def test_agent_loop_replays_thought_signatures(gemini):
    model = PriceModel().fit(data.historical_sales(n=1200, seed=3))
    listings = data.active_listings(n=20, seed=8)

    def respond(body, n):
        tools = body.get("tools", [{}])[0].get("functionDeclarations", [])
        if [t["name"] for t in tools] == ["record_listings"]:  # parser batch inside analyze_listings
            text = body["contents"][0]["parts"][0]["text"]
            rows = []
            for lid, desc in zip(listings["listing_id"], listings["description"]):
                if f"[{lid}]" in text:
                    rows.append({"id": lid, **parser.parse_rules(desc).model_dump()})
            return _call("record_listings", {"listings": rows})
        turn = sum(1 for c in body["contents"] if c["role"] == "model")
        if turn == 0:
            return _call("search_listings", {"max_price": 10_000_000}, sig="SIG-A")
        if turn == 1:
            # the previous model turn must be replayed verbatim, signature included
            assert body["contents"][1]["parts"][0]["thoughtSignature"] == "SIG-A"
            fr = body["contents"][2]["parts"][0]["functionResponse"]
            assert fr["name"] == "search_listings" and len(fr["response"]["result"]) == 20
            return _call("analyze_listings", {"ids": list(listings["listing_id"])}, sig="SIG-B")
        results = body["contents"][-1]["parts"][0]["functionResponse"]["response"]["result"]
        deals = [r for r in results if r.get("verdict") in ("strong deal", "deal")]
        return _call("submit_report", {"market_summary": "ok", "deals":
                                       [{"id": d["id"], "headline": "h", "rationale": "r"} for d in deals]
                                       + [{"id": "MADE-UP", "headline": "x", "rationale": "y"}]})

    gemini(respond)
    tools = DealTools(listings, model)
    assert run_llm_agent(tools, {}) == "llm"
    assert all(d["id"] != "MADE-UP" for d in tools.report["deals"])  # invented ids dropped


def test_agent_falls_back_when_api_down(gemini, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("network error")
    monkeypatch.setattr(llm, "_post", boom)
    model = PriceModel().fit(data.historical_sales(n=1200, seed=3))
    tools = DealTools(data.active_listings(n=15, seed=8), model)
    note = run_llm_agent(tools, {"max_price": 500000})
    assert "built-in planner" in note and tools.report is not None
