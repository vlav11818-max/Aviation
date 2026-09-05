"""Direct HTTP client for kie.ai's ``/claude/v1/messages`` endpoint.

LiteLLM's Anthropic transformer was failing on kie.ai's success
responses with ``KeyError: 'content'`` — its parser assumes the exact
Anthropic-Messages response shape and blows up on any variation.

This client bypasses LiteLLM entirely for ``kie/*`` calls:

* POSTs to ``{KIE_BASE_URL}/v1/messages`` with ``Authorization: Bearer``
  (kie.ai's actual header) and Anthropic's request body shape.
* Handles both possible response shapes tolerantly — the canonical
  Anthropic ``{"content": [{"type":"text","text":…}], "usage": {…}}``
  and an OpenAI-compatible ``{"choices": [{"message": …}]}`` fallback
  in case kie.ai returns that instead.
* Returns a normalised dict with the same shape the router already
  consumes from ``litellm.acompletion``, so nothing downstream cares.
* Extracts a helpful error message on non-2xx (kie.ai puts useful
  detail in the body).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class KieHTTPError(Exception):
    """Raised when kie.ai returns a non-2xx or an unrecognised body."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _resolve_messages_url(base_url: str) -> str:
    """``https://api.kie.ai/claude``       → …/claude/v1/messages
       ``https://api.kie.ai/claude/v1``    → …/claude/v1/messages
       ``https://api.kie.ai/claude/v1/messages`` → unchanged
    """
    base = (base_url or "").rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic wants ``system`` as a top-level field, not a message role."""
    system_parts: list[str] = []
    non_system: list[dict] = []
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(str(content))
        else:
            non_system.append({"role": role or "user", "content": content})
    return "\n\n".join(system_parts), non_system


async def kie_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One completion against kie.ai's Anthropic-Messages endpoint.

    Returns a LiteLLM-shaped dict:
    ``{"choices":[{"message":{"content":str}}], "usage": {...}, "model": str}``.
    """
    system, chat = _split_system(messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": chat,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        body["system"] = system

    url = _resolve_messages_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body, headers=headers)

    text_body = resp.text or ""
    if resp.status_code >= 400:
        # Try to pull a useful message out of the body.
        try:
            data = resp.json()
            msg = (
                (data.get("error") or {}).get("message")
                or data.get("message")
                or text_body[:400]
            )
        except Exception:
            msg = text_body[:400]
        raise KieHTTPError(
            f"kie.ai {resp.status_code}: {msg}",
            status_code=resp.status_code,
            body=text_body[:2000],
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise KieHTTPError(
            f"kie.ai returned non-JSON response: {exc}",
            status_code=resp.status_code,
            body=text_body[:2000],
        ) from exc

    text, tokens_in, tokens_out = _extract_text_and_usage(data)
    return {
        "id": data.get("id", ""),
        "model": data.get("model", model),
        "choices": [
            {
                "index": 0,
                "finish_reason": data.get("stop_reason") or "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    }


def _extract_text_and_usage(data: dict[str, Any]) -> tuple[str, int, int]:
    """Handle both Anthropic-shape and OpenAI-shape response bodies."""
    text = ""
    tokens_in = 0
    tokens_out = 0

    # Anthropic shape: {"content": [{"type":"text","text":"..."}], "usage": {...}}
    if isinstance(data.get("content"), list):
        for block in data["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text") or "")
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        return text, tokens_in, tokens_out

    # OpenAI shape: {"choices": [{"message": {"content": "..."}}], "usage": {...}}
    if isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        return text, tokens_in, tokens_out

    # Some proxies wrap the payload one level deeper.
    if isinstance(data.get("data"), dict):
        return _extract_text_and_usage(data["data"])

    keys = list(data.keys())[:8]
    raise KieHTTPError(
        f"kie.ai response has no 'content' or 'choices' field. Top-level keys: {keys}. "
        f"First 400 chars: {json.dumps(data, ensure_ascii=False)[:400]}",
        status_code=200,
        body=json.dumps(data, ensure_ascii=False)[:2000],
    )
