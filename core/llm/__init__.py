"""LiteLLM-backed LLM router for the aviation content factory.

Public surface:
    ``call_llm(model, messages, temperature, max_tokens, json_mode)`` ->
    :class:`LLMResponse` — one shot at one model, with retry.

Model IDs follow the LiteLLM convention (see :func:`route_call`):
    ``openrouter/anthropic/claude-3.5-sonnet``
    ``openai/gpt-4o`` (or bare ``gpt-4o``)
    ``anthropic/claude-3-5-sonnet-latest``
    ``gemini/gemini-1.5-pro``
    ``deepseek/deepseek-chat``
    ``custom/<any-model>``  — uses CUSTOM_BASE_URL + CUSTOM_API_KEY
    ``kie/<any-model>``     — uses KIE_BASE_URL + KIE_API_KEY
    ``mock/demo``           — deterministic offline provider (see ``mock_provider``)

The high-level :class:`APIClient` in ``core.api_client`` wraps this with
the rate limiter, fallback pool, and event-bus emissions.
"""

from __future__ import annotations

from core.llm.router import LLMResponse, call_llm, is_mock_model, route_call

__all__ = ["LLMResponse", "call_llm", "is_mock_model", "route_call"]
