"""Thin wrapper around :func:`core.llm.call_llm` used by the aviation
agents.

Adds:

* JSON-mode robust parsing — pulls the first JSON object out of a
  code fence or a bare object even when the model wraps it with a
  short prose preamble.
* Retry with a "your last output was not valid JSON — try again"
  reminder when parsing fails.
* Cost accounting into the caller's :class:`~aviation.state.AviationJob`.
* A single ``llm_json`` and ``llm_text`` API used by every agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APIResponseError,
)
from core.llm import LLMResponse, call_llm
from aviation.state import AviationJob

logger = logging.getLogger(__name__)

# Retry policy for transient upstream errors (503 "resources tight" /
# 429 rate-limit / network blips / timeouts). The router raises them as
# APIResponseError / APIRateLimitError / APIConnectionError. Auth errors
# and 4xx client errors are not retried.
_TRANSIENT = (APIConnectionError, APIRateLimitError, APIResponseError)
_MAX_RETRIES = 5              # 6 total attempts
_BACKOFF_BASE = 2.0           # seconds
_BACKOFF_MAX = 45.0           # cap


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _extract_json(text: str) -> str:
    """Best-effort extract of the JSON payload from a model response.

    Handles: raw JSON, ```json fenced```, and JSON preceded/followed
    by prose. Returns the JSON substring, or the original text if
    no JSON-looking region is found (json.loads then decides).
    """
    if not text:
        return "{}"
    stripped = text.strip()
    fence = _JSON_FENCE.search(stripped)
    if fence:
        return fence.group(1).strip()
    # Find first { or [ and matching close.
    open_idx = min(
        (i for i in (stripped.find("{"), stripped.find("[")) if i >= 0),
        default=-1,
    )
    if open_idx < 0:
        return stripped
    opener = stripped[open_idx]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(open_idx, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return stripped[open_idx : i + 1]
    return stripped[open_idx:]


async def llm_text(
    *,
    job: AviationJob,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    agent: str = "writer",
) -> str:
    """One completion, tracked into ``job``, returns the raw text."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await _call(model, messages, temperature, max_tokens, json_mode=False)
    _track_cost(job, response, model)
    job.append_log(agent, f"{model} → {response.tokens_out} tokens out")
    return response.text


async def llm_json(
    *,
    job: AviationJob,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    agent: str = "planner",
    retries: int = 1,
) -> dict[str, Any] | list[Any]:
    """One completion that must return JSON.

    Wraps :func:`llm_text` with an extract + parse loop that retries
    with a stern reminder when the model returns non-JSON.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = await _call(
            model, messages, temperature, max_tokens, json_mode=True
        )
        _track_cost(job, response, model)
        payload = _extract_json(response.text)
        try:
            data = json.loads(payload)
            job.append_log(agent, f"{model} → JSON ok ({response.tokens_out} tokens)")
            return data
        except json.JSONDecodeError as exc:
            last_error = exc
            job.append_log(
                agent,
                f"{model} returned non-JSON on attempt {attempt + 1}: {exc}. "
                "Prompt reminder appended.",
                level="warn",
            )
            messages.append({"role": "assistant", "content": response.text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON — the parser "
                        f"error was: {exc}. Re-emit ONLY the JSON object, "
                        "starting with { or [, and nothing else."
                    ),
                }
            )
    # All retries exhausted — raise so the orchestrator can decide.
    raise ValueError(f"LLM did not return parseable JSON after {retries + 1} tries: {last_error}")


async def _call(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> LLMResponse:
    """Send one completion with retry on transient upstream errors.

    Retries on 503 ServiceUnavailable, 429 RateLimit, connection errors
    and timeouts. Non-retryable auth errors and 4xx client errors are
    surfaced immediately. Backs off exponentially (2 s → 4 s → 8 s → 16 s
    → 32 s, capped at 45 s) with a jitter to avoid a thundering herd.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except APIAuthError:
            # Bad key — no point retrying.
            raise
        except _TRANSIENT as exc:
            last_exc = exc
            # Honour Retry-After if the provider gave one.
            hint = getattr(exc, "retry_after", None)
            if hint is not None:
                wait = min(float(hint), _BACKOFF_MAX)
            else:
                wait = min(_BACKOFF_MAX, _BACKOFF_BASE * (2 ** attempt))
            # ±25 % jitter.
            wait *= 0.75 + 0.5 * random.random()
            if attempt >= _MAX_RETRIES:
                logger.error(
                    "LLM call failed on %s after %d attempts: %s",
                    model, attempt + 1, exc,
                )
                raise
            logger.warning(
                "Transient error on %s (attempt %d/%d, backing off %.1fs): %s",
                model, attempt + 1, _MAX_RETRIES + 1, wait, exc,
            )
            await asyncio.sleep(wait)
        except Exception as exc:
            logger.error("LLM call failed on %s: %s", model, exc)
            raise
    # Loop fell through (shouldn't happen — the last iteration re-raises).
    if last_exc is not None:
        raise last_exc
    raise APIError(f"Unexpected retry-loop exit for {model}", provider="", model=model)


def _track_cost(job: AviationJob, response: LLMResponse, model: str) -> None:
    job.tokens_in += response.tokens_in
    job.tokens_out += response.tokens_out
    # Rough cost from the settings pricing table.
    try:
        from core.settings import Settings

        pricing_table = Settings().api.pricing
    except Exception:
        return
    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
        candidates.append(model.rsplit("/", 1)[-1])
    for cand in candidates:
        p = pricing_table.get(cand)
        if p is not None:
            job.cost_usd += (
                response.tokens_in / 1_000_000
            ) * p.input + (response.tokens_out / 1_000_000) * p.output
            return
