"""Unified LLM API client (LiteLLM-backed).

This is a thin façade over :mod:`core.llm.router`:

    ``APIClient.send(messages, temperature, max_tokens, model=None) -> str``

Responsibilities layered on top of the raw router call:

* **Model routing** — chooses the effective LiteLLM model id from the
  ``APIConfig`` (see :func:`_effective_model_id`), so the legacy call
  sites that only know ``primary_provider`` / ``primary_model`` keep
  working unchanged.
* **Retry with back-off** — via ``tenacity``, honouring
  ``retry_after`` hints on rate-limit errors.
* **Fallback pool** — tries each fallback provider in order after the
  primary is exhausted, emitting ``API_FALLBACK`` events for the UI.
* **Rate limiting** — one semaphore per provider (see
  :class:`~core.rate_limiter.RateLimiter`).
* **Cost tracking** — updates cumulative token counters and emits
  ``COST_UPDATE`` events. Uses the pricing table on ``settings.api.pricing``.

The old adapter-per-wire-format machinery
(``OpenAIAdapter`` / ``AnthropicAdapter`` / ``GoogleAdapter``) is gone;
LiteLLM handles every provider through one call.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
)

from core.events import EventBus, EventType
from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APIResponseError,
)
from core.llm import LLMResponse, call_llm, is_mock_model
from core.rate_limiter import RateLimiter
from core.settings import Settings
from models.config import APIConfig, APIProvider, FallbackPoolEntry, PROVIDER_CONFIG

logger = logging.getLogger(__name__)


# Provider prefix used by LiteLLM for each APIProvider enum value.
# ``openai`` keeps the raw model id (no prefix); everything else prefixes.
_PROVIDER_PREFIX: dict[str, str] = {
    "openrouter": "openrouter/",
    "openai": "",
    "anthropic": "anthropic/",
    "google": "gemini/",
    "deepseek": "deepseek/",
    "qwen": "openrouter/",  # Qwen is easiest via OpenRouter; keep as reasonable default.
}


def _effective_model_id(provider: APIProvider | str, model: str) -> str:
    """Return the LiteLLM-shaped model id for a (provider, model) pair.

    Rules:

    * If ``model`` already contains a "/" prefix that matches a known
      provider (``openrouter/…``, ``anthropic/…``, ``gemini/…``,
      ``deepseek/…``, ``kie/…``, ``custom/…``, ``mock/…``) it is
      returned unchanged.
    * For legacy call sites that pass ``primary_provider=openrouter`` +
      ``primary_model='anthropic/claude-3.5-sonnet'`` the OpenRouter
      prefix is added: ``openrouter/anthropic/claude-3.5-sonnet``.
    * For ``primary_provider=openai`` the raw model id
      (``gpt-4o``, ``gpt-4-turbo`` …) is used as-is.
    """
    m = (model or "").strip()
    if not m:
        return m
    known = ("openrouter/", "anthropic/", "gemini/", "google/", "deepseek/",
             "kie/", "custom/", "mock/", "openai/")
    if any(m.startswith(p) for p in known):
        return m
    if isinstance(provider, APIProvider):
        provider_name = provider.value
    else:
        provider_name = str(provider).lower()
    prefix = _PROVIDER_PREFIX.get(provider_name, "")
    return f"{prefix}{m}" if prefix else m


class APIClient:
    """LiteLLM-backed API client with retry, fallback pool, rate limiting, and cost tracking."""

    def __init__(
        self,
        api_config: APIConfig,
        settings: Settings,
        event_bus: EventBus,
    ) -> None:
        self._config = api_config
        self._settings = settings
        self._event_bus = event_bus

        # Cumulative cost tracking.
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._total_cost_usd: float = 0.0

        # One rate limiter per provider we might use.
        self._limiters: dict[str, RateLimiter] = {}
        self._limiters[self._provider_name()] = self._create_rate_limiter(self._provider_name())

        pool_summary = ", ".join(
            f"{e.provider.value}/{e.model}" for e in self._config.get_effective_fallback_pool()
        ) or "none"
        logger.info(
            "APIClient initialised (LiteLLM): primary=%s/%s, fallback_pool=[%s]",
            self._provider_name(),
            self._config.primary_model,
            pool_summary,
        )

    # ── public interface ───────────────────────────────────────────

    async def send(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a prompt and return the generated text.

        Args:
            messages: Chat messages (OpenAI format).
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            model: Optional override. When ``None`` the primary model is
                used, translated to a LiteLLM id via
                :func:`_effective_model_id`. Overrides may be either raw
                (``gpt-4o-mini``, evaluated via the primary provider's
                prefix) or fully-qualified (``anthropic/claude-3-5-haiku-latest``).
            json_mode: If True, requests a JSON-formatted response.

        Returns:
            The generated text.
        """
        primary_provider = self._provider_name()
        effective_model = _effective_model_id(
            self._config.primary_provider, model or self._config.primary_model,
        )
        capped = self._cap_max_tokens(max_tokens, primary_provider)

        # Primary try (with retries).
        try:
            response = await self._send_with_retry(
                model=effective_model,
                api_key=self._config.api_key,
                provider_name=primary_provider,
                messages=messages,
                temperature=temperature,
                max_tokens=capped,
                json_mode=json_mode,
            )
            self._track_cost(response, effective_model)
            return response.text
        except APIAuthError:
            raise
        except APIError as primary_err:
            logger.warning("APIClient primary %s exhausted: %s", effective_model, primary_err)
            self._event_bus.emit(
                EventType.API_ERROR,
                provider=primary_provider,
                model=effective_model,
                error=str(primary_err),
            )
            pool = self._config.get_effective_fallback_pool()
            if pool and self._config.auto_fallback:
                return await self._try_fallback_pool(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    primary_error=primary_err,
                )
            raise

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    def reset_cost_tracking(self) -> None:
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_cost_usd = 0.0

    async def reconfigure(self, api_config: APIConfig) -> None:
        """Swap in a new APIConfig (e.g. after the user changed provider in the UI)."""
        await self.close()
        self._config = api_config
        self._limiters = {}
        self._limiters[self._provider_name()] = self._create_rate_limiter(self._provider_name())
        logger.info(
            "APIClient reconfigured: primary=%s/%s",
            self._provider_name(),
            self._config.primary_model,
        )

    async def close(self) -> None:
        """No-op — LiteLLM manages its own connection pool."""
        return None

    def close_sync(self) -> None:  # pragma: no cover - kept for legacy callers
        return None

    # ── internals ──────────────────────────────────────────────────

    def _provider_name(self) -> str:
        return self._config.primary_provider.value

    def _limiter_for(self, provider_name: str) -> RateLimiter:
        limiter = self._limiters.get(provider_name)
        if limiter is None:
            limiter = self._create_rate_limiter(provider_name)
            self._limiters[provider_name] = limiter
        return limiter

    async def _send_with_retry(
        self,
        *,
        model: str,
        api_key: str,
        provider_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        retry_cfg = self._settings.retry
        max_retries = max(1, int(self._config.max_retries))

        def wait_fn(retry_state: Any) -> float:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(exc, APIRateLimitError) and exc.retry_after is not None:
                return min(float(exc.retry_after), retry_cfg.max_delay_seconds)
            attempt = retry_state.attempt_number
            wait = retry_cfg.initial_delay_seconds * (retry_cfg.exponential_base ** (attempt - 1))
            return min(wait, retry_cfg.max_delay_seconds)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                wait=wait_fn,
                retry=retry_if_exception_type(
                    (APIConnectionError, APIRateLimitError, APIResponseError)
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._send_once(
                        model=model,
                        api_key=api_key,
                        provider_name=provider_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
        except RetryError as exc:
            last = exc.last_attempt.exception()
            if last is not None:
                raise last from exc
            raise APIError(
                f"All {max_retries} retries exhausted for {model}",
                provider=provider_name,
                model=model,
            ) from exc

        # Unreachable.
        raise APIError(f"Unexpected retry exit for {model}", provider=provider_name, model=model)

    async def _send_once(
        self,
        *,
        model: str,
        api_key: str,
        provider_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        limiter = self._limiter_for(provider_name)
        await limiter.acquire()
        try:
            return await call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                api_key=api_key or None,
                timeout=float(self._config.timeout),
                extra_headers=self._extra_headers(provider_name),
            )
        except APIRateLimitError as exc:
            limiter.report_rate_limit(exc.retry_after)
            raise
        except APIResponseError as exc:
            # Do not retry non-transient 4xx client errors.
            if exc.status_code is not None and 400 <= exc.status_code < 500:
                raise APIError(
                    f"Non-retryable client error ({exc.status_code}) from {provider_name}: {exc.message}",
                    provider=provider_name,
                    model=model,
                ) from exc
            raise
        finally:
            limiter.release()

    async def _try_fallback_pool(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        primary_error: APIError,
    ) -> str:
        pool: list[FallbackPoolEntry] = self._config.get_effective_fallback_pool()
        errors: list[str] = [
            f"Primary ({self._provider_name()}/{self._config.primary_model}): {primary_error}"
        ]

        for entry in pool:
            provider_name = entry.provider.value
            if provider_name == self._provider_name():
                continue
            if not entry.api_key and not is_mock_model(entry.model):
                # No key configured — silently skip.
                continue
            effective = _effective_model_id(entry.provider, entry.model)
            capped = self._cap_max_tokens(max_tokens, provider_name)

            self._event_bus.emit(
                EventType.API_FALLBACK,
                primary_provider=self._provider_name(),
                primary_model=self._config.primary_model,
                fallback_provider=provider_name,
                fallback_model=entry.model,
                reason=str(primary_error),
            )

            try:
                response = await self._send_with_retry(
                    model=effective,
                    api_key=entry.api_key,
                    provider_name=provider_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=capped,
                    json_mode=json_mode,
                )
                self._track_cost(response, effective)
                logger.info("APIClient fallback %s succeeded", effective)
                return response.text
            except APIAuthError as exc:
                errors.append(f"{provider_name}/{entry.model}: auth — {exc}")
                continue
            except APIError as exc:
                self._event_bus.emit(
                    EventType.API_ERROR,
                    provider=provider_name,
                    model=entry.model,
                    error=str(exc),
                )
                errors.append(f"{provider_name}/{entry.model}: {exc}")
                continue

        total_tried = 1 + len(pool)
        raise APIError(
            f"All {total_tried} providers failed. " + "; ".join(errors),
            provider="all",
            model="",
        )

    # ── cost tracking ──────────────────────────────────────────────

    def _track_cost(self, response: LLMResponse, model: str) -> None:
        self._total_tokens_in += response.tokens_in
        self._total_tokens_out += response.tokens_out
        cost = self._calculate_cost(response.tokens_in, response.tokens_out, model)
        self._total_cost_usd += cost
        self._event_bus.emit(
            EventType.COST_UPDATE,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=cost,
            total_tokens_in=self._total_tokens_in,
            total_tokens_out=self._total_tokens_out,
            total_cost_usd=self._total_cost_usd,
            model=model,
        )

    def _calculate_cost(self, tokens_in: int, tokens_out: int, model: str) -> float:
        pricing_table = self._settings.api.pricing
        # Try full id first, then the bare part after the last "/".
        candidates = [model]
        if "/" in model:
            candidates.append(model.split("/", 1)[1])
            candidates.append(model.rsplit("/", 1)[-1])
        for candidate in candidates:
            pricing = pricing_table.get(candidate)
            if pricing is not None:
                return (tokens_in / 1_000_000) * pricing.input + (tokens_out / 1_000_000) * pricing.output
        logger.debug("APIClient: no pricing entry for '%s', cost=0.0", model)
        return 0.0

    # ── misc ───────────────────────────────────────────────────────

    _PROVIDER_MAX_TOKENS: dict[str, int] = {
        "deepseek": 8192,
        "qwen": 8192,
    }

    @staticmethod
    def _cap_max_tokens(max_tokens: int, provider_name: str) -> int:
        limit = APIClient._PROVIDER_MAX_TOKENS.get(provider_name)
        if limit is not None and max_tokens > limit:
            return limit
        return max_tokens

    def _create_rate_limiter(self, provider_name: str) -> RateLimiter:
        rate_limits = self._settings.api.rate_limits
        max_concurrent = getattr(rate_limits, provider_name, 3)
        return RateLimiter(provider=provider_name, max_concurrent=max_concurrent)

    @staticmethod
    def _extra_headers(provider_name: str) -> dict[str, str] | None:
        if provider_name == "openrouter":
            import os
            return {
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://aviation-content-factory.local"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Aviation Content Factory"),
            }
        return None


# Legacy re-export so downstream code that imported PROVIDER_CONFIG from
# ``core.api_client`` (via the old adapter layer) keeps working.
__all__ = ["APIClient", "PROVIDER_CONFIG"]
