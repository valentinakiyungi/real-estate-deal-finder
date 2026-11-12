"""GenAI listing parser: messy description -> structured, validated features.

Primary path: Claude with a forced tool call whose input schema is the Pydantic
model below, so the output is always schema-shaped JSON. A rule-based extractor
runs as (a) an offline fallback and (b) a cross-check; fields where the two
disagree are reported so the agent can lower its confidence.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from . import llm


class ListingFeatures(BaseModel):
    beds: int | None = Field(None, ge=0, le=12, description="Number of bedrooms")
    baths: float | None = Field(None, ge=0, le=10, description="Bathrooms; a half bath counts 0.5")
    sqft: int | None = Field(None, ge=200, le=20000, description="Interior LIVING area in sq ft (NOT the lot)")
    lot_sqft: int | None = Field(None, ge=0, le=2_000_000, description="Lot size in sq ft; convert acres x 43560")
    year_built: int | None = Field(None, ge=1800, le=2027)
    renovated: bool = Field(False, description="Substantially renovated/remodeled")
    condition: Literal["poor", "fair", "good", "excellent"] = Field(
        "good", description="poor=fixer/as-is/major work; fair=dated/cosmetic work; good=maintained; excellent=pristine/turnkey")
    garage_spaces: int = Field(0, ge=0, le=6)
    pool: bool = False
    view: bool = False
    red_flags: list[str] = Field(default_factory=list, description=(
        "Material risks a buyer must price in: foundation/structural, as-is or estate sale, cash-only/unfinanceable, "
        "flood zone, roof end-of-life, tenant-occupied, unpermitted work. Short phrases."))


SYSTEM = ("You extract structured facts from real-estate listing descriptions. Use only what the text states; "
          "leave a field null if it is genuinely absent. Do not confuse lot size with living area. "
          "Ignore marketing fluff ('won't last', 'motivated seller').")
TOOL = {"name": "record_listing", "description": "Record the structured features of one listing.",
        "input_schema": ListingFeatures.model_json_schema()}


# ------------------------------------------------------------- rule-based ---
_W = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_N = r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven)"
_RED = {
    "foundation issues": r"foundation", "cash only / unfinanceable": r"cash only|not qualify for financ",
    "flood zone": r"flood zone", "roof end of life": r"roof (?:at|near) end", "tenant occupied": r"tenant",
    "as-is / estate sale": r"as-is|estate sale", "unpermitted work": r"unpermitted",
}


def _num(s: str) -> float:
    return float(_W.get(s.lower(), s.replace(",", "")))


def parse_rules(text: str) -> ListingFeatures:
    t = text.lower()
    f: dict = {}
    # lot first, then strip it so it can't be mistaken for living area
    m = re.search(r"lot\s*(?:of\s*)?(\d*\.?\d+)\s*acres?", t) or re.search(r"(\d*\.?\d+)\s*acres?", t)
    if m:
        f["lot_sqft"] = int(float(m.group(1)) * 43560)
        t_wo_lot = t.replace(m.group(0), " ")
    else:
        m = re.search(r"([\d,]+)\s*(?:sq\.?\s*ft\.?|sqft|sf)\s*lot", t)
        if m:
            f["lot_sqft"] = int(m.group(1).replace(",", ""))
        t_wo_lot = t.replace(m.group(0), " ") if m else t
    m = re.search(r"(\d+(?:\.\d)?)k\s*sq\.?\s*ft", t_wo_lot) or re.search(r"([\d,]{3,})\s*(?:sq\.?\s*ft\.?|sqft|sf)\b", t_wo_lot)
    if m:
        v = m.group(1).replace(",", "")
        f["sqft"] = int(float(v) * 1000) if "k" in m.group(0).split()[0] else int(v)
    m = re.search(_N + r"\s*(?:bed(?:room)?s?|br|bdrm|bd)\b", t)
    if m:
        f["beds"] = int(_num(m.group(1)))
    m = re.search(r"(\d+)\s*full\s*\+\s*(\d+)\s*half", t)
    if m:
        f["baths"] = int(m.group(1)) + 0.5 * int(m.group(2))
    else:
        m = re.search(_N + r"\s*(?:bath(?:room)?s?|ba)\b", t)
        if m:
            f["baths"] = _num(m.group(1))
    m = re.search(r"(?:built|circa|yr blt|constructed in)\s*(?:in\s*)?((?:18|19|20)\d\d)", t)
    if m:
        f["year_built"] = int(m.group(1))
    m = re.search(r"(\d)\s*-?\s*car\s*gar", t) or re.search(r"garage\s*\((\d)\)", t)
    f["garage_spaces"] = int(m.group(1)) if m else 0
    f["pool"] = "pool" in t
    f["view"] = bool(re.search(r"\bviews?\b", t))
    f["renovated"] = bool(re.search(r"renovat|remodel|gut reno", t))
    if re.search(r"needs (?:tlc|major)|handyman|fixer", t):
        f["condition"] = "poor"
    elif re.search(r"updating needed|original kitchen|cosmetic", t):
        f["condition"] = "fair"
    elif re.search(r"immaculate|pristine|turnkey|designer", t):
        f["condition"] = "excellent"
    else:
        f["condition"] = "good"
    f["red_flags"] = [k for k, pat in _RED.items() if re.search(pat, t)]
    return ListingFeatures(**f)


# ------------------------------------------------------------------- LLM ---
class _BatchItem(ListingFeatures):
    id: str = Field(description="The listing id exactly as given")


class _Batch(BaseModel):
    listings: list[_BatchItem]


BATCH_TOOL = {"name": "record_listings", "description": "Record the structured features of every listing given.",
              "input_schema": _Batch.model_json_schema()}
BATCH_SIZE = 15  # listings per request; keeps free-tier request counts low


def parse_llm_batch(items: list[tuple[str, str]]) -> dict[str, ListingFeatures]:
    """One LLM request for several (id, description) pairs."""
    body = "\n\n".join(f"[{i}]\n{t}" for i, t in items)
    resp = llm.messages(
        system=SYSTEM, tools=[BATCH_TOOL], tool_choice={"type": "tool", "name": "record_listings"},
        messages=[{"role": "user", "content": f"Extract features for each of these {len(items)} listings. "
                                              f"Return one entry per listing id.\n\n{body}"}],
        max_tokens=400 * len(items) + 500)
    block = next(b for b in resp["content"] if b["type"] == "tool_use")
    out = {}
    for row in block["input"].get("listings", []):
        try:
            item = _BatchItem(**row)
        except Exception:  # one malformed row shouldn't sink the batch
            continue
        out[item.id] = ListingFeatures(**item.model_dump(exclude={"id"}))
    return out


def _merge(ai: ListingFeatures, rules: ListingFeatures) -> tuple[ListingFeatures, list[str]]:
    diffs = [k for k in ("beds", "baths", "sqft", "year_built", "garage_spaces", "condition")
             if getattr(ai, k) is not None and getattr(ai, k) != getattr(rules, k)]
    # fill any gaps the LLM left from the rule extractor
    merged = ai.model_copy(update={k: getattr(rules, k) for k in ("beds", "baths", "sqft", "lot_sqft", "year_built")
                                   if getattr(ai, k) is None})
    return merged, diffs


def parse_many(items: list[tuple[str, str]], use_llm: bool | None = None,
               on_progress=None) -> dict[str, tuple[ListingFeatures, dict]]:
    """Parse (id, description) pairs. Returns id -> (features, meta).

    meta["source"] is "llm" or "rules"; meta["disagreements"] lists fields where the
    LLM and the rule extractor disagree (used to lower confidence)."""
    if use_llm is None:
        use_llm = llm.available()
    results = {i: (parse_rules(t), {"source": "rules", "disagreements": []}) for i, t in items}
    if not use_llm:
        return results
    for start in range(0, len(items), BATCH_SIZE):
        chunk = items[start:start + BATCH_SIZE]
        if on_progress:
            on_progress(f"AI parsing listings {start + 1}–{start + len(chunk)} of {len(items)}")
        try:
            got = parse_llm_batch(chunk)
        except Exception as e:  # network / quota problems -> keep rule results for this chunk
            for i, _ in chunk:
                results[i] = (results[i][0], {"source": f"rules (AI failed: {str(e)[:80]})", "disagreements": []})
            continue
        for i, _ in chunk:
            if i in got:
                merged, diffs = _merge(got[i], results[i][0])
                results[i] = (merged, {"source": "llm", "disagreements": diffs})
    return results


def parse(text: str, use_llm: bool | None = None) -> tuple[ListingFeatures, dict]:
    """Parse a single description. Returns (features, meta)."""
    return parse_many([("listing", text)], use_llm)["listing"]
