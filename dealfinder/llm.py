"""Minimal Claude Messages API client using only the standard library.

Set ANTHROPIC_API_KEY to enable. Model can be overridden with DEALFINDER_MODEL.
"""
from __future__ import annotations

import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("DEALFINDER_MODEL", "claude-sonnet-5-5")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def messages(messages: list[dict], system: str = "", tools: list[dict] | None = None,
             tool_choice: dict | None = None, max_tokens: int = 2048, model: str | None = None) -> dict:
    body = {"model": model or DEFAULT_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(), method="POST",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())
