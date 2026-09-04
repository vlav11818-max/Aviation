"""Unified API client for all 6 LLM providers.

Selects the appropriate adapter based on the provider's wire format,
applies retry with exponential back-off (via ``tenacity``), integrates
rate limiting, falls back through a pool of alternative providers on
persistent failure, tracks cumulative token costs, and emits events
for the GUI.

FIX: Replaced single-fallback with **fallback pool**.  When the primary
provider exhausts retries, the client iterates through all pool entries
in order.  The first successful response is returned.  If all providers
fail, a summary error is raised.

Typical usage::

    client = APIClient(
        api_config=api_config,
        settings=settings,
        event_bus=event_bus,
    )
    text = await client.send(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=4096,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.api_adapters.anthropic_adapter import AnthropicAdapter
from core.api_adapters.google_adapter import GoogleAdapter
from core.api_adapters import APIResponse
from core.api_adapters.openai_adapter import OpenAIAdapter
from core.events import EventBus, EventType
from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APIResponseError,
)
from core.rate_limiter import RateLimiter
from core.settings import Settings
from models.config import (
    APIConfig,
    APIFormat,
    APIProvider,
    FallbackPoolEntry,
    PROVIDER_CONFIG,
)

logger = logging.getLogger(__name__)


def _wait_respect_retry_after(
    retry_state: Any,
    *,
    multiplier: float,
    max_wait: float,
    exp_base: float,
) -> float:
    """Custom wait function that uses ``retry_after`` from rate-limit errors.

    If the last exception is an ``APIRateLimitError`` with a non-None
    ``retry_after`` value, that value is used as the wait duration
    (capped at ``max_wait``).  Otherwise falls back to standard
    exponential backoff.

    Args:
        retry_state: Tenacity retry state object.
        multiplier: Base multiplier for exponential backoff.
        max_wait: Maximum wait time in seconds.
        exp_base: Exponential base.

    Returns:
        Seconds to wait before the next retry.
    """
    # Check if the last exception carries a retry_after hint.
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, APIRateLimitError) and exc.retry_after is not None:
        return min(float(exc.retry_after), max_wait)

    # Fall back to exponential backoff.
    attempt = retry_state.attempt_number
    wait = multiplier * (exp_base ** (attempt - 1))
    return min(wait, max_wait)


class APIClient:
    """Unified API client that works with all 6 providers.

    Responsibilities:

    - Selects the correct adapter based on provider format.
    - Retries transient errors with exponential back-off.
    - Falls back through a pool of alternative providers on failure.
    - Integrates per-provider rate limiting.
    - Tracks cumulative token usage and emits ``COST_UPDATE`` events.
    - Emits ``API_ERROR`` and ``API_FALLBACK`` events for the GUI.

    Args:
        api_config: Provider, model, keys, and resilience settings.
        settings: Application-wide settings (retry, rate-limit config).
        event_bus: Event bus for GUI communication.
    """

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

        # Build primary adapter and rate limiter.
        self._primary_adapter = self._create_adapter(
            provider=self._config.primary_provider,
            api_key=self._config.api_key,
            model=self._config.primary_model,
        )
        self._primary_limiter = self._create_rate_limiter(
            self._config.primary_provider
        )

        # Build fallback pool adapters.
        self._pool_entries: list[_PoolAdapterEntry] = []
        self._build_fallback_pool()

        pool_summary = ", ".join(
            f"{e.provider_name}/{e.model}" for e in self._pool_entries
        ) or "none"
        logger.info(
            "APIClient initialised: primary=%s/%s, fallback_pool=[%s]",
            self._config.primary_provider.value,
            self._config.primary_model,
            pool_summary,
        )

    # ── public interface ────────────────────────────────────────────

    async def send(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        """Send a prompt and return the generated text.

        Retries transient errors on the primary provider.  If all retries
        are exhausted, iterates through the fallback pool in order until
        one succeeds or all fail.

        Args:
            messages: Chat messages in standard format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Optional model override.  If ``None``, uses the
                primary model from the API config.  Use this to send
                evaluation prompts to a cheaper/lighter model.

        Returns:
            The generated text string.

        Raises:
            APIAuthError: If authentication fails on all providers.
            APIError: If all retry and fallback attempts are exhausted.
        """
        effective_model = model or self._config.primary_model

        # Cap max_tokens to the provider's known limit.
        capped_max_tokens = self._cap_max_tokens(
            max_tokens, self._config.primary_provider.value
        )

        # Try primary provider with retries.
        try:
            response = await self._send_with_retry(
                adapter=self._primary_adapter,
                limiter=self._primary_limiter,
                messages=messages,
                model=effective_model,
                temperature=temperature,
                max_tokens=capped_max_tokens,
                provider_name=self._config.primary_provider.value,
            )
            self._track_cost(response, effective_model)
            return response.text
        except APIAuthError:
            # Auth errors are not retryable and should not fall back.
            raise
        except APIError as primary_err:
            logger.warning(
                "APIClient primary provider %s/%s exhausted retries: %s",
                self._config.primary_provider.value,
                self._config.primary_model,
                primary_err,
            )
            self._event_bus.emit(
                EventType.API_ERROR,
                provider=self._config.primary_provider.value,
                model=effective_model,
                error=str(primary_err),
            )

            # Try fallback pool.
            if self._pool_entries and self._config.auto_fallback:
                return await self._try_fallback_pool(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    primary_error=primary_err,
                )

            raise

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input tokens across all requests."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output tokens across all requests."""
        return self._total_tokens_out

    @property
    def total_cost_usd(self) -> float:
        """Cumulative estimated cost in USD across all requests."""
        return self._total_cost_usd

    def reset_cost_tracking(self) -> None:
        """Reset cumulative token and cost counters to zero."""
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_cost_usd = 0.0
        logger.debug("APIClient cost tracking reset")

    async def reconfigure(self, api_config: APIConfig) -> None:
        """Reconfigure the client with a new API configuration.

        Closes existing adapter sessions and rebuilds primary and
        fallback pool adapters from the new config.  This allows the
        GUI to change provider/model at runtime without recreating the
        entire client.

        Cost tracking counters are **not** reset — call
        ``reset_cost_tracking()`` separately if needed.

        Args:
            api_config: The new API configuration from the GUI.
        """
        # Close existing sessions first.
        await self.close()

        self._config = api_config

        # Rebuild primary adapter and rate limiter.
        self._primary_adapter = self._create_adapter(
            provider=self._config.primary_provider,
            api_key=self._config.api_key,
            model=self._config.primary_model,
        )
        self._primary_limiter = self._create_rate_limiter(
            self._config.primary_provider
        )

        # Rebuild fallback pool.
        self._pool_entries = []
        self._build_fallback_pool()

        pool_summary = ", ".join(
            f"{e.provider_name}/{e.model}" for e in self._pool_entries
        ) or "none"
        logger.info(
            "APIClient reconfigured: primary=%s/%s, fallback_pool=[%s]",
            self._config.primary_provider.value,
            self._config.primary_model,
            pool_summary,
        )

    async def close(self) -> None:
        """Close all underlying adapter HTTP sessions.

        Should be called during application shutdown to release
        connection pool resources gracefully.
        """
        # Close primary.
        if hasattr(self._primary_adapter, "close"):
            try:
                await self._primary_adapter.close()
            except Exception as exc:
                logger.warning("APIClient: error closing primary adapter: %s", exc)

        # Close all pool adapters.
        for entry in self._pool_entries:
            if hasattr(entry.adapter, "close"):
                try:
                    await entry.adapter.close()
                except Exception as exc:
                    logger.warning(
                        "APIClient: error closing pool adapter %s: %s",
                        entry.provider_name,
                        exc,
                    )

        logger.debug("APIClient sessions closed")

    def close_sync(self) -> None:
        """Synchronous wrapper for close() — for use in finally blocks."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except RuntimeError:
            # No event loop — create a temporary one.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.close())
            finally:
                loop.close()

    # ── fallback pool construction ──────────────────────────────────

    def _build_fallback_pool(self) -> None:
        """Build adapter+limiter pairs for each fallback pool entry.

        Skips entries that match the primary provider (no point in
        falling back to the same provider) and entries without an API
        key.
        """
        pool = self._config.get_effective_fallback_pool()
        primary_provider = self._config.primary_provider.value

        for entry in pool:
            provider_name = entry.provider.value

            # Skip if same as primary — no point falling back to self.
            if provider_name == primary_provider:
                logger.debug(
                    "Skipping pool entry %s (same as primary)",
                    provider_name,
                )
                continue

            # Skip if no API key.
            if not entry.api_key:
                logger.debug(
                    "Skipping pool entry %s (no API key)",
                    provider_name,
                )
                continue

            try:
                adapter = self._create_adapter(
                    provider=entry.provider,
                    api_key=entry.api_key,
                    model=entry.model,
                )
                limiter = self._create_rate_limiter(entry.provider)

                self._pool_entries.append(
                    _PoolAdapterEntry(
                        provider_name=provider_name,
                        model=entry.model,
                        adapter=adapter,
                        limiter=limiter,
                    )
                )
                logger.debug(
                    "Fallback pool: added %s/%s",
                    provider_name,
                    entry.model,
                )
            except APIError as exc:
                logger.warning(
                    "Failed to create adapter for pool entry %s: %s",
                    provider_name,
                    exc,
                )

    # ── retry logic ─────────────────────────────────────────────────

    async def _send_with_retry(
        self,
        adapter: OpenAIAdapter | AnthropicAdapter | GoogleAdapter,
        limiter: RateLimiter,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        provider_name: str,
    ) -> APIResponse:
        """Send a request with exponential-backoff retry.

        Retries on ``APIConnectionError``, ``APIRateLimitError``, and
        ``APIResponseError`` (status >= 500).  Does **not** retry on
        ``APIAuthError``.

        Args:
            adapter: The provider adapter to use.
            limiter: The rate limiter for this provider.
            messages: Chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Max tokens.
            provider_name: Name string for logging.

        Returns:
            The ``APIResponse`` from a successful call.

        Raises:
            APIAuthError: Immediately (not retried).
            APIError: If all retries fail.
        """
        retry_cfg = self._settings.retry
        max_retries = self._config.max_retries

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max(max_retries, 1)),
                wait=lambda rs: _wait_respect_retry_after(
                    rs,
                    multiplier=retry_cfg.initial_delay_seconds,
                    max_wait=retry_cfg.max_delay_seconds,
                    exp_base=retry_cfg.exponential_base,
                ),
                retry=retry_if_exception_type(
                    (APIConnectionError, APIRateLimitError, APIResponseError)
                ),
                reraise=True,
            ):
                with attempt:
                    attempt_num = attempt.retry_state.attempt_number
                    logger.debug(
                        "APIClient attempt %d/%d for %s/%s",
                        attempt_num,
                        max_retries,
                        provider_name,
                        model,
                    )
                    return await self._send_once(
                        adapter=adapter,
                        limiter=limiter,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

        except RetryError as exc:
            # Unwrap the last exception from the retry chain.
            last_exc = exc.last_attempt.exception()
            if last_exc is not None:
                raise last_exc from exc
            raise APIError(
                f"All {max_retries} retries exhausted for {provider_name}/{model}",
                provider=provider_name,
                model=model,
            ) from exc

        # Unreachable, but keeps the type checker happy.
        raise APIError(  # pragma: no cover
            f"Unexpected retry exit for {provider_name}/{model}",
            provider=provider_name,
            model=model,
        )

    async def _send_once(
        self,
        adapter: OpenAIAdapter | AnthropicAdapter | GoogleAdapter,
        limiter: RateLimiter,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> APIResponse:
        """Send a single request through the rate limiter.

        On rate-limit errors the limiter is notified before re-raising.

        Args:
            adapter: The provider adapter.
            limiter: The rate limiter.
            messages: Chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Max tokens.

        Returns:
            ``APIResponse`` on success.

        Raises:
            APIRateLimitError: Notifies the limiter then re-raises.
            APIAuthError: Re-raised immediately.
            APIConnectionError: Re-raised for retry.
            APIResponseError: Re-raised for retry (if server error).
        """
        await limiter.acquire()
        try:
            return await adapter.send(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIRateLimitError as exc:
            limiter.report_rate_limit(exc.retry_after)
            raise
        except APIResponseError as exc:
            # Only retry server errors (5xx).  Client errors (4xx other
            # than 401/403/429) are not retryable.
            if exc.status_code is not None and 400 <= exc.status_code < 500:
                raise APIError(
                    f"Non-retryable client error ({exc.status_code}) from "
                    f"{limiter.provider}: {exc.message}",
                    provider=limiter.provider,
                    model=model,
                ) from exc
            raise
        finally:
            limiter.release()

    # ── fallback pool iteration ─────────────────────────────────────

    async def _try_fallback_pool(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        primary_error: APIError,
    ) -> str:
        """Iterate through the fallback pool until one provider succeeds.

        Each pool entry is tried with full retry logic.  Auth errors
        on a pool entry cause it to be skipped (not the whole pool).

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Max tokens.
            primary_error: The error from the primary provider.

        Returns:
            Generated text from the first successful fallback.

        Raises:
            APIError: If all pool entries also fail.
        """
        errors: list[str] = [
            f"Primary ({self._config.primary_provider.value}/"
            f"{self._config.primary_model}): {primary_error}"
        ]

        for entry in self._pool_entries:
            # Cap max_tokens for this provider.
            capped = self._cap_max_tokens(max_tokens, entry.provider_name)

            logger.info(
                "APIClient fallback pool: trying %s/%s",
                entry.provider_name,
                entry.model,
            )
            self._event_bus.emit(
                EventType.API_FALLBACK,
                primary_provider=self._config.primary_provider.value,
                primary_model=self._config.primary_model,
                fallback_provider=entry.provider_name,
                fallback_model=entry.model,
                reason=str(primary_error),
            )

            try:
                response = await self._send_with_retry(
                    adapter=entry.adapter,
                    limiter=entry.limiter,
                    messages=messages,
                    model=entry.model,
                    temperature=temperature,
                    max_tokens=capped,
                    provider_name=entry.provider_name,
                )
                self._track_cost(response, entry.model)
                logger.info(
                    "APIClient fallback pool: %s/%s succeeded",
                    entry.provider_name,
                    entry.model,
                )
                return response.text

            except APIAuthError as exc:
                error_msg = (
                    f"{entry.provider_name}/{entry.model}: "
                    f"Auth error — {exc}"
                )
                logger.warning("APIClient pool: %s", error_msg)
                errors.append(error_msg)
                # Auth error on one provider — skip it, try next.
                continue

            except APIError as exc:
                error_msg = (
                    f"{entry.provider_name}/{entry.model}: {exc}"
                )
                logger.warning("APIClient pool: %s", error_msg)
                self._event_bus.emit(
                    EventType.API_ERROR,
                    provider=entry.provider_name,
                    model=entry.model,
                    error=str(exc),
                )
                errors.append(error_msg)
                continue

        # All pool entries exhausted.
        total_tried = 1 + len(self._pool_entries)  # primary + pool
        summary = "; ".join(errors)
        raise APIError(
            f"All {total_tried} providers failed. {summary}",
            provider="all",
            model="",
        )

    # ── cost tracking ───────────────────────────────────────────────

    def _track_cost(self, response: APIResponse, model: str) -> None:
        """Update cumulative counters and emit a COST_UPDATE event.

        Args:
            response: The API response with token counts.
            model: The model used (for pricing lookup).
        """
        self._total_tokens_in += response.tokens_in
        self._total_tokens_out += response.tokens_out

        cost = self._calculate_cost(
            response.tokens_in, response.tokens_out, model
        )
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

    def _calculate_cost(
        self, tokens_in: int, tokens_out: int, model: str
    ) -> float:
        """Calculate the cost of a single request in USD.

        Looks up the model in the settings pricing table.  Returns 0.0
        if the model is not found (with a warning).

        Args:
            tokens_in: Input token count.
            tokens_out: Output token count.
            model: Model identifier.

        Returns:
            Estimated cost in USD.
        """
        pricing = self._settings.api.pricing.get(model)
        if pricing is None:
            logger.warning(
                "APIClient: no pricing entry for model '%s', cost=0.0",
                model,
            )
            return 0.0

        # Pricing is per 1M tokens.
        cost_in = (tokens_in / 1_000_000) * pricing.input
        cost_out = (tokens_out / 1_000_000) * pricing.output
        return cost_in + cost_out

    # ── adapter factory ─────────────────────────────────────────────

    # Per-provider maximum output tokens.  If a caller requests more
    # than this, the value is silently capped to avoid 400 errors.
    _PROVIDER_MAX_TOKENS: dict[str, int] = {
        "deepseek": 8192,
        "qwen": 8192,
        # OpenRouter, OpenAI, Anthropic, Google — no hard cap needed
        # (their limits are high enough for story generation).
    }

    @staticmethod
    def _cap_max_tokens(max_tokens: int, provider_name: str) -> int:
        """Cap max_tokens to the provider's known limit.

        Args:
            max_tokens: Requested max tokens.
            provider_name: Provider name string.

        Returns:
            Capped max_tokens value.
        """
        limit = APIClient._PROVIDER_MAX_TOKENS.get(provider_name)
        if limit is not None and max_tokens > limit:
            logger.debug(
                "APIClient: capping max_tokens %d → %d for provider %s",
                max_tokens,
                limit,
                provider_name,
            )
            return limit
        return max_tokens

    def _create_adapter(
        self,
        provider: APIProvider,
        api_key: str,
        model: str,
    ) -> OpenAIAdapter | AnthropicAdapter | GoogleAdapter:
        """Instantiate the correct adapter for a given provider.

        Args:
            provider: The API provider enum value.
            api_key: API key for the provider.
            model: Model identifier (unused in construction, logged).

        Returns:
            An adapter instance.

        Raises:
            APIError: If the provider format is unknown.
        """
        provider_name = provider.value
        provider_meta = PROVIDER_CONFIG.get(provider_name, {})
        api_format = provider_meta.get("format")
        base_url = provider_meta.get("base_url", "")
        timeout = self._config.timeout

        logger.debug(
            "APIClient creating adapter for %s (format=%s, base_url=%s)",
            provider_name,
            api_format,
            base_url,
        )

        if api_format in (APIFormat.OPENAI_COMPATIBLE, APIFormat.OPENAI_NATIVE):
            extra_headers = self._get_extra_headers(provider)
            return OpenAIAdapter(
                base_url=base_url,
                api_key=api_key,
                provider=provider_name,
                timeout_seconds=timeout,
                extra_headers=extra_headers,
            )

        if api_format == APIFormat.ANTHROPIC_NATIVE:
            return AnthropicAdapter(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )

        if api_format == APIFormat.GOOGLE_NATIVE:
            return GoogleAdapter(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )

        raise APIError(
            f"Unknown API format '{api_format}' for provider '{provider_name}'",
            provider=provider_name,
            model=model,
        )

    def _create_rate_limiter(self, provider: APIProvider) -> RateLimiter:
        """Create a rate limiter for the given provider.

        Reads the per-provider max concurrent setting from
        ``settings.api.rate_limits``.

        Args:
            provider: The API provider.

        Returns:
            A ``RateLimiter`` instance.
        """
        rate_limits = self._settings.api.rate_limits
        provider_name = provider.value
        max_concurrent = getattr(rate_limits, provider_name, 3)
        return RateLimiter(
            provider=provider_name,
            max_concurrent=max_concurrent,
        )

    @staticmethod
    def _get_extra_headers(provider: APIProvider) -> dict[str, str]:
        """Return any extra headers needed for OpenAI-compatible providers.

        OpenRouter requires an ``HTTP-Referer`` and ``X-Title`` header.
        Other providers use the standard ``Authorization: Bearer`` which
        the ``OpenAIAdapter`` already handles.

        Args:
            provider: The API provider.

        Returns:
            Extra headers dict (may be empty).
        """
        if provider == APIProvider.OPENROUTER:
            return {
                "HTTP-Referer": "https://ai-story-generator.local",
                "X-Title": "AI Story Generator Pro",
            }
        return {}


# ── Internal dataclass for pool entries ─────────────────────────────────


class _PoolAdapterEntry:
    """Holds a pre-built adapter + limiter for one fallback pool entry.

    This is an internal class — not part of the public API.

    Attributes:
        provider_name: Provider string (e.g. ``"openai"``).
        model: Model identifier.
        adapter: The provider adapter instance.
        limiter: The rate limiter instance.
    """

    __slots__ = ("provider_name", "model", "adapter", "limiter")

    def __init__(
        self,
        provider_name: str,
        model: str,
        adapter: OpenAIAdapter | AnthropicAdapter | GoogleAdapter,
        limiter: RateLimiter,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.adapter = adapter
        self.limiter = limiter
