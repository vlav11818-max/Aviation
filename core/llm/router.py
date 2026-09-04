"""LiteLLM-backed router — one entry point for every provider.

This module owns the actual network call. The high-level
:class:`~core.api_client.APIClient` sits on top and adds rate limiting,
cost tracking, a fallback pool, and event-bus notifications.

Model identifiers follow the LiteLLM convention:

===============================================  ===========================
Model ID                                         Provider / notes
===============================================  ===========================
``openrouter/anthropic/claude-3.5-sonnet``       OpenRouter — ``OPENROUTER_API_KEY``
``openai/gpt-4o`` (or bare ``gpt-4o``)           OpenAI                    — ``OPENAI_API_KEY``
``anthropic/claude-3-5-sonnet-latest``           Anthropic direct          — ``ANTHROPIC_API_KEY``
``gemini/gemini-1.5-pro``                        Google Gemini             — ``GEMINI_API_KEY``
``deepseek/deepseek-chat``                       DeepSeek                  — ``DEEPSEEK_API_KEY``
``kie/<any-model>``                              kie.ai — ``KIE_API_KEY`` + ``KIE_BASE_URL``
``custom/<any-model>``                           Any OpenAI-compatible base URL —
                                                  ``CUSTOM_API_KEY`` + ``CUSTOM_BASE_URL``
``mock/demo`` (also just ``mock``)               Offline deterministic mock
===============================================  ===========================

The router intentionally does not implement its own multi-provider
fallback — that is the ``APIClient``'s job, because it also owns rate
limiting and event emissions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APIResponseError,
)
from core.llm.mock_provider import mock_completion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Normalised response returned by :func:`call_llm`."""

    text: str
    tokens_in: int
    tokens_out: int
    model: str
    raw: dict = field(default_factory=dict)


# ── Provider prefix table ──────────────────────────────────────────────
#
# Maps a model-id prefix to (env_var_for_key, optional_base_url_env_var).
# The special ``mock`` prefix short-circuits the network entirely.
_PROVIDER_PREFIXES: dict[str, tuple[str, str | None]] = {
    "openrouter": ("OPENROUTER_API_KEY", None),
    "openai": ("OPENAI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "gemini": ("GEMINI_API_KEY", None),
    "google": ("GEMINI_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", None),
    "kie": ("KIE_API_KEY", "KIE_BASE_URL"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    "mock": (None, None),  # type: ignore[dict-item]
}


def is_mock_model(model: str) -> bool:
    """Return ``True`` if the given model ID resolves to the offline mock."""
    if not model:
        return False
    if os.environ.get("AVIATION_FORCE_MOCK", "").strip() == "1":
        return True
    prefix = model.split("/", 1)[0].lower()
    return prefix == "mock"


def _provider_of(model: str) -> str:
    """Return the LiteLLM provider prefix for a model id.

    A bare model id like ``gpt-4o`` or ``claude-3-5-sonnet-latest`` is
    routed to a well-known provider by pattern-matching the prefix; the
    LiteLLM defaults do the same, but we mirror the logic here for the
    rate-limiter key.
    """
    if "/" in model:
        return model.split("/", 1)[0].lower()
    lower = model.lower()
    if lower.startswith("gpt-") or lower.startswith("o1-") or lower.startswith("o3-"):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("gemini"):
        return "gemini"
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("qwen"):
        return "openai"  # qwen goes via OpenAI-compatible endpoint
    return "openai"


def _extract_key(model: str, explicit_key: str | None) -> tuple[str | None, str | None]:
    """Return (api_key, base_url) for the given model, honouring env vars.

    ``explicit_key`` takes precedence when non-empty. Otherwise the
    provider's environment variable is consulted. Returns ``(None, None)``
    for the mock provider.
    """
    provider = _provider_of(model)
    if provider == "mock":
        return None, None
    key_env, base_env = _PROVIDER_PREFIXES.get(provider, ("", None))
    if explicit_key:
        api_key: str | None = explicit_key
    else:
        api_key = os.environ.get(key_env) if key_env else None
    base_url = os.environ.get(base_env) if base_env else None
    return api_key, base_url


def _classify_error(exc: Exception) -> APIError:
    """Translate a LiteLLM exception into one of our typed exceptions."""
    # Import lazily so tests can run without LiteLLM misbehaviour on import.
    try:
        from litellm import exceptions as le  # type: ignore
    except Exception:  # pragma: no cover - defensive
        le = None  # type: ignore

    if le is not None:
        if isinstance(exc, le.AuthenticationError):
            return APIAuthError(str(exc), provider=getattr(exc, "llm_provider", ""), model=getattr(exc, "model", ""))
        if isinstance(exc, le.RateLimitError):
            retry_after = getattr(exc, "retry_after", None)
            return APIRateLimitError(str(exc), provider=getattr(exc, "llm_provider", ""), model=getattr(exc, "model", ""), retry_after=retry_after)
        if isinstance(exc, (le.APIConnectionError, le.Timeout)):
            return APIConnectionError(str(exc), provider=getattr(exc, "llm_provider", ""), model=getattr(exc, "model", ""))
        if isinstance(exc, le.APIError):
            status = getattr(exc, "status_code", None)
            return APIResponseError(str(exc), provider=getattr(exc, "llm_provider", ""), model=getattr(exc, "model", ""), status_code=status)
    return APIResponseError(str(exc), provider="", model="", status_code=None)


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    json_mode: bool = False,
    api_key: str | None = None,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> LLMResponse:
    """Send one chat completion through LiteLLM (or the mock).

    Args:
        model: LiteLLM-style model id. See module docstring.
        messages: Chat messages in OpenAI format.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        json_mode: If True, request a JSON-formatted response
            (``response_format={"type": "json_object"}``). Silently
            ignored by providers that do not support it.
        api_key: Explicit key; falls back to env var for the provider.
        timeout: Wall-clock timeout for one call in seconds.
        extra_headers: Extra HTTP headers (e.g. OpenRouter attribution).

    Returns:
        A :class:`LLMResponse`.

    Raises:
        APIAuthError / APIRateLimitError / APIConnectionError / APIResponseError
        or bare ``APIError`` on unclassified LiteLLM exceptions.
    """
    if is_mock_model(model):
        raw = mock_completion(messages, max_tokens=max_tokens)
        return LLMResponse(
            text=raw["choices"][0]["message"]["content"],
            tokens_in=int(raw["usage"]["prompt_tokens"]),
            tokens_out=int(raw["usage"]["completion_tokens"]),
            model=raw["model"],
            raw=raw,
        )

    key, base_url = _extract_key(model, api_key)
    if not key:
        raise APIAuthError(
            f"No API key configured for model '{model}'. Set the provider's "
            f"env var (e.g. OPENROUTER_API_KEY) or pass api_key explicitly.",
            provider=_provider_of(model),
            model=model,
        )

    # Import lazily so an environment without LiteLLM (e.g. a unit test
    # that only exercises the mock branch) does not fail to import this module.
    import litellm  # type: ignore

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": key,
        "timeout": timeout,
        "num_retries": 0,  # our APIClient owns retry
    }
    if base_url:
        kwargs["api_base"] = base_url
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as exc:
        raise _classify_error(exc) from exc

    # LiteLLM returns a ModelResponse; access via dict-like or attribute.
    try:
        choice = resp["choices"][0]
        text = choice["message"]["content"] or ""
        usage = resp.get("usage", {}) or {}
    except Exception as exc:  # pragma: no cover - defensive
        raise APIResponseError(
            f"Malformed LiteLLM response for {model}: {exc}",
            provider=_provider_of(model),
            model=model,
            status_code=None,
        ) from exc

    return LLMResponse(
        text=text,
        tokens_in=int(usage.get("prompt_tokens", 0) or 0),
        tokens_out=int(usage.get("completion_tokens", 0) or 0),
        model=getattr(resp, "model", model),
        raw=dict(resp) if hasattr(resp, "__iter__") else {},
    )


def route_call(  # pragma: no cover - convenience sync wrapper
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> LLMResponse:
    """Synchronous helper for scripts and tests."""
    return asyncio.get_event_loop().run_until_complete(
        call_llm(model, messages, **kwargs)
    )
