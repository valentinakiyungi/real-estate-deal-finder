"""LLM client with two providers, standard library only.

Provider is picked automatically from the keys you have:

    GEMINI_API_KEY      -> Google Gemini (free tier works)       model: GEMINI_MODEL   (default gemini-flash-latest)
    ANTHROPIC_API_KEY   -> Anthropic Claude                       model: CLAUDE_MODEL   (default claude-sonnet-5-5)

Force one with DEALFINDER_PROVIDER=gemini|anthropic. Keys can live in a `.env`
file in the project folder (see .env.example).

The rest of the code speaks one message format (Anthropic-style: text /
tool_use / tool_result blocks). This module translates to and from Gemini's
format, including function declarations, forced tool calls and the "thought
signatures" Gemini 3 models require to be sent back on later turns.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ config ---
def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=value lines). Real environment variables win."""
    for p in [path] if path else [Path.cwd() / ".env", ROOT / ".env"]:
        if p and p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


load_env()


def _gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def provider() -> str | None:
    forced = os.environ.get("DEALFINDER_PROVIDER", "").lower()
    if forced in ("gemini", "anthropic"):
        return forced
    if _gemini_key():
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def available() -> bool:
    return provider() is not None


def model_name() -> str:
    if provider() == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    return os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")


def describe() -> str:
    p = provider()
    return f"{p} · {model_name()}" if p else "offline (no API key)"


# -------------------------------------------------------------------- http ---
class LLMError(RuntimeError):
    pass


def _post(url: str, body: dict, headers: dict, retries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(min(60, 4 * 2 ** attempt))  # free tiers rate-limit: back off and retry
                continue
            hint = " (rate limit: free tier allows only a few requests per minute)" if e.code == 429 else ""
            raise LLMError(f"HTTP {e.code}{hint}: {detail}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(f"network error: {e.reason}") from None
    raise LLMError("unreachable")


# --------------------------------------------------------------- public API ---
def messages(messages: list[dict], system: str = "", tools: list[dict] | None = None,
             tool_choice: dict | None = None, max_tokens: int = 2048, model: str | None = None) -> dict:
    """Send a conversation; returns {"content": [blocks]} in Anthropic block format."""
    p = provider()
    if p == "gemini":
        return _gemini(messages, system, tools, tool_choice, max_tokens, model or model_name())
    if p == "anthropic":
        return _anthropic(messages, system, tools, tool_choice, max_tokens, model or model_name())
    raise LLMError("No API key. Put GEMINI_API_KEY=... in a .env file in the project folder.")


# --------------------------------------------------------------- anthropic ---
def _anthropic(msgs, system, tools, tool_choice, max_tokens, model) -> dict:
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    return _post("https://api.anthropic.com/v1/messages", body,
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})


# ------------------------------------------------------------------ gemini ---
_KEEP = {"description", "enum", "required", "minimum", "maximum", "minItems", "maxItems", "format"}


def gemini_schema(schema: dict, defs: dict | None = None) -> dict:
    """JSON Schema (as produced by Pydantic) -> the OpenAPI subset Gemini accepts."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return gemini_schema(defs[schema["$ref"].split("/")[-1]], defs)
    nullable = False
    if "anyOf" in schema:  # Optional[X] -> X + nullable
        opts = [o for o in schema["anyOf"] if o.get("type") != "null"]
        nullable = len(opts) < len(schema["anyOf"])
        merged = {**{k: v for k, v in schema.items() if k != "anyOf"}, **opts[0]}
        out = gemini_schema(merged, defs)
        if nullable:
            out["nullable"] = True
        return out
    out = {k: v for k, v in schema.items() if k in _KEEP}
    t = schema.get("type") or ("string" if "enum" in schema else "object" if "properties" in schema else None)
    if t:
        out["type"] = t.upper()
    if "properties" in schema:
        out["properties"] = {k: gemini_schema(v, defs) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = gemini_schema(schema["items"], defs)
    return out


def to_gemini_contents(msgs: list[dict]) -> list[dict]:
    names: dict[str, str] = {}  # tool_use_id -> function name
    native_ids: set[str] = set()  # ids Gemini itself issued (echo them back)
    contents = []
    for m in msgs:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        if m["role"] == "assistant":
            for b in blocks:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
            raw = next((b["parts"] for b in blocks if b.get("type") == "_gemini_raw"), None)
            native_ids.update(p["functionCall"]["id"] for p in raw or [] if p.get("functionCall", {}).get("id"))
            if raw is None:  # built by us, not returned by Gemini
                raw = [{"text": b["text"]} if b["type"] == "text" else
                       {"functionCall": {"name": b["name"], "args": b["input"]}}
                       for b in blocks if b["type"] in ("text", "tool_use")]
            contents.append({"role": "model", "parts": raw})  # replayed verbatim -> keeps thought signatures
        else:
            parts = []
            for b in blocks:
                if b["type"] == "text":
                    parts.append({"text": b["text"]})
                elif b["type"] == "tool_result":
                    try:
                        payload = json.loads(b["content"])
                    except (TypeError, ValueError):
                        payload = b["content"]
                    fr = {"name": names.get(b["tool_use_id"], "tool"), "response": {"result": payload}}
                    if b["tool_use_id"] in native_ids:
                        fr["id"] = b["tool_use_id"]
                    parts.append({"functionResponse": fr})
            contents.append({"role": "user", "parts": parts})
    return contents


def from_gemini_response(resp: dict) -> dict:
    cands = resp.get("candidates") or []
    if not cands:
        raise LLMError(f"Gemini returned no candidates: {json.dumps(resp.get('promptFeedback', resp))[:300]}")
    parts = cands[0].get("content", {}).get("parts", [])
    blocks = []
    for p in parts:
        if "functionCall" in p:
            fc = p["functionCall"]
            blocks.append({"type": "tool_use", "id": fc.get("id") or uuid.uuid4().hex[:12],
                           "name": fc["name"], "input": fc.get("args", {})})
        elif p.get("text") and not p.get("thought"):
            blocks.append({"type": "text", "text": p["text"]})
    blocks.append({"type": "_gemini_raw", "parts": parts})
    return {"content": blocks, "stop_reason": cands[0].get("finishReason")}


def _gemini(msgs, system, tools, tool_choice, max_tokens, model) -> dict:
    body: dict = {"contents": to_gemini_contents(msgs),
                  "generationConfig": {"maxOutputTokens": max(max_tokens, 4096)}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t.get("description", ""),
             "parameters": gemini_schema(t["input_schema"])} for t in tools]}]
        cfg = {"mode": "AUTO"}
        if tool_choice and tool_choice.get("type") == "tool":
            cfg = {"mode": "ANY", "allowedFunctionNames": [tool_choice["name"]]}
        body["toolConfig"] = {"functionCallingConfig": cfg}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = _post(url, body, {"x-goog-api-key": _gemini_key()})
    return from_gemini_response(resp)
