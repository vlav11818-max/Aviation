"""OpenAI-compatible API adapter.

Handles all providers that speak the OpenAI chat-completions wire format:
OpenAI (native), OpenRouter, DeepSeek, and Qwen.  Differences in base URL
and authentication headers are handled transparently.

The adapter uses ``aiohttp`` directly (not the ``openai`` SDK) so that
timeout, header, and error handling are uniform across all providers.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from core.api_adapters import APIResponse
from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
)

logger = logging.getLogger(__name__)


# ── Response dataclass ──────────────────────────────────────────────────



# ── OpenAI adapter ──────────────────────────────────────────────────────


class OpenAIAdapter:
    """Adapter for OpenAI-compatible chat-completion APIs.

    Works with OpenAI native (``openai_native``), OpenRouter, DeepSeek,
    and Qwen (``openai_compatible``).  The caller is responsible for
    supplying the correct ``base_url`` and ``api_key`` for the target
    provider.

    Args:
        base_url: Provider API base URL (e.g. ``https://api.openai.com/v1``).
        api_key: Bearer token / API key.
        provider: Provider name string used in log messages and exceptions.
        timeout_seconds: Per-request timeout in seconds.
        extra_headers: Optional additional headers merged into every request.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider: str,
        timeout_seconds: int = 120,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider = provider
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._extra_headers = extra_headers or {}
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, creating it lazily.

        The session is reused across requests to enable HTTP connection
        pooling and avoid per-request TCP/TLS overhead.

        Returns:
            The shared ``aiohttp.ClientSession``.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session.

        Should be called when the adapter is no longer needed (e.g. on
        application shutdown) to release connection pool resources.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── public ──────────────────────────────────────────────────────

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> APIResponse:
        """Send a chat-completion request and return a normalised response.

        Args:
            messages: OpenAI-format messages list
                (``[{"role": "user", "content": "…"}, …]``).
            model: Model identifier (e.g. ``"gpt-4o"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            An ``APIResponse`` with text, token counts, and model used.

        Raises:
            APIAuthError: On 401 / 403.
            APIRateLimitError: On 429.
            APIResponseError: On other non-2xx status or malformed body.
            APIConnectionError: On network / timeout errors.
        """
        url = f"{self._base_url}/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(messages, model, temperature, max_tokens)

        logger.debug(
            "OpenAIAdapter [%s] sending request to %s, model=%s, temp=%.2f, max_tokens=%d",
            self._provider,
            url,
            model,
            temperature,
            max_tokens,
        )

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.json(content_type=None)
                self._check_status(resp.status, body)
                return self._parse_response(body)

        except (APIAuthError, APIRateLimitError, APIResponseError):
            raise
        except aiohttp.ClientError as exc:
            logger.error(
                "OpenAIAdapter [%s] connection error: %s", self._provider, exc
            )
            raise APIConnectionError(
                f"Connection error for {self._provider}: {exc}",
                provider=self._provider,
                model=model,
            ) from exc
        except TimeoutError as exc:
            logger.error(
                "OpenAIAdapter [%s] request timed out after %ss",
                self._provider,
                self._timeout.total,
            )
            raise APIConnectionError(
                f"Request to {self._provider} timed out after {self._timeout.total}s",
                provider=self._provider,
                model=model,
            ) from exc

    # ── private helpers ─────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Construct request headers including auth and extras.

        Returns:
            Headers dict ready for the HTTP request.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._extra_headers)
        return headers

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the JSON request body.

        Args:
            messages: Chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.

        Returns:
            Dict suitable for JSON serialisation.
        """
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _check_status(self, status: int, body: dict[str, Any]) -> None:
        """Raise typed exceptions for non-2xx HTTP status codes.

        Args:
            status: HTTP status code.
            body: Decoded JSON response body.

        Raises:
            APIAuthError: On 401 / 403.
            APIRateLimitError: On 429.
            APIResponseError: On any other non-2xx.
        """
        if 200 <= status < 300:
            return

        error_msg = self._extract_error_message(body)

        if status in (401, 403):
            logger.error(
                "OpenAIAdapter [%s] auth error (%d): %s",
                self._provider,
                status,
                error_msg,
            )
            raise APIAuthError(
                f"Authentication failed for {self._provider}: {error_msg}",
                provider=self._provider,
            )

        if status == 429:
            retry_after = self._extract_retry_after(body)
            logger.warning(
                "OpenAIAdapter [%s] rate limited (429), retry_after=%s",
                self._provider,
                retry_after,
            )
            raise APIRateLimitError(
                f"Rate limited by {self._provider}: {error_msg}",
                retry_after=retry_after,
                provider=self._provider,
            )

        logger.error(
            "OpenAIAdapter [%s] API error (%d): %s",
            self._provider,
            status,
            error_msg,
        )
        raise APIResponseError(
            f"API error from {self._provider} ({status}): {error_msg}",
            status_code=status,
            response_body=str(body),
            provider=self._provider,
        )

    def _parse_response(self, body: dict[str, Any]) -> APIResponse:
        """Extract text, token counts, and model from the response body.

        Args:
            body: Decoded JSON body from a successful (2xx) response.

        Returns:
            Normalised ``APIResponse``.

        Raises:
            APIResponseError: If the body lacks expected fields.
        """
        try:
            choices = body.get("choices", [])
            if not choices:
                raise APIResponseError(
                    f"No choices in response from {self._provider}",
                    response_body=str(body),
                    provider=self._provider,
                )

            message = choices[0].get("message", {})
            text = message.get("content", "")
            if not text:
                raise APIResponseError(
                    f"Empty content in response from {self._provider}",
                    response_body=str(body),
                    provider=self._provider,
                )

            usage = body.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            model_used = body.get("model", "")

            logger.debug(
                "OpenAIAdapter [%s] response: model=%s, tokens_in=%d, tokens_out=%d",
                self._provider,
                model_used,
                tokens_in,
                tokens_out,
            )

            return APIResponse(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model_used,
                raw=body,
            )

        except APIResponseError:
            raise
        except Exception as exc:
            logger.error(
                "OpenAIAdapter [%s] failed to parse response: %s",
                self._provider,
                exc,
            )
            raise APIResponseError(
                f"Failed to parse response from {self._provider}: {exc}",
                response_body=str(body),
                provider=self._provider,
            ) from exc

    def _extract_error_message(self, body: dict[str, Any]) -> str:
        """Pull a human-readable error message from an error response body.

        Args:
            body: Decoded JSON error body.

        Returns:
            Error message string, or ``"unknown error"`` as fallback.
        """
        error = body.get("error", {})
        if isinstance(error, dict):
            return error.get("message", str(error) or "unknown error")
        return str(error) or "unknown error"

    def _extract_retry_after(self, body: dict[str, Any]) -> float | None:
        """Attempt to extract a retry-after hint from a 429 response.

        Args:
            body: Decoded JSON body from a 429 response.

        Returns:
            Seconds to wait, or ``None`` if not available.
        """
        error = body.get("error", {})
        if isinstance(error, dict):
            metadata = error.get("metadata", {})
            if isinstance(metadata, dict):
                raw = metadata.get("retry_after")
                if raw is not None:
                    try:
                        return float(raw)
                    except (ValueError, TypeError):
                        pass
        return None
