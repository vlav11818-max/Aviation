"""Anthropic Messages API adapter.

Handles the Anthropic-native wire format which differs from OpenAI in
several ways: ``x-api-key`` header instead of ``Authorization: Bearer``,
separate ``system`` parameter, content blocks response structure, and
different error payloads.

Uses ``aiohttp`` directly for consistency with the other adapters.
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

# Anthropic requires an API version header.
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API.

    Maps the standard ``[{"role": …, "content": …}]`` message list into
    Anthropic's format, which separates the ``system`` message from the
    conversation turns.

    Args:
        base_url: Anthropic API base URL
            (e.g. ``https://api.anthropic.com/v1``).
        api_key: Anthropic API key (sent via ``x-api-key`` header).
        timeout_seconds: Per-request timeout in seconds.
    """

    _PROVIDER = "anthropic"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
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
        """Send a message to the Anthropic Messages API.

        Args:
            messages: Standard chat messages.  A message with
                ``role="system"`` is extracted and passed as the
                top-level ``system`` parameter.
            model: Model identifier
                (e.g. ``"claude-3-5-sonnet-20241022"``).
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
        url = f"{self._base_url}/messages"
        headers = self._build_headers()
        system_text, conversation = self._split_system(messages)
        payload = self._build_payload(
            system_text, conversation, model, temperature, max_tokens
        )

        logger.debug(
            "AnthropicAdapter sending request to %s, model=%s, temp=%.2f, max_tokens=%d",
            url,
            model,
            temperature,
            max_tokens,
        )

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.json(content_type=None)
                self._check_status(resp.status, body, model)
                return self._parse_response(body)

        except (APIAuthError, APIRateLimitError, APIResponseError):
            raise
        except aiohttp.ClientError as exc:
            logger.error("AnthropicAdapter connection error: %s", exc)
            raise APIConnectionError(
                f"Connection error for {self._PROVIDER}: {exc}",
                provider=self._PROVIDER,
                model=model,
            ) from exc
        except TimeoutError as exc:
            logger.error(
                "AnthropicAdapter request timed out after %ss",
                self._timeout.total,
            )
            raise APIConnectionError(
                f"Request to {self._PROVIDER} timed out after {self._timeout.total}s",
                provider=self._PROVIDER,
                model=model,
            ) from exc

    # ── private helpers ─────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Construct request headers with ``x-api-key`` authentication.

        Returns:
            Headers dict.
        """
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _split_system(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Separate system messages from conversation turns.

        Anthropic's API takes ``system`` as a top-level parameter, not as
        a message with ``role="system"``.

        Args:
            messages: Standard chat message list.

        Returns:
            A tuple of ``(system_text, conversation)`` where
            ``system_text`` is the concatenation of all system messages
            and ``conversation`` contains only user/assistant turns.
        """
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                conversation.append(msg)

        return "\n\n".join(system_parts), conversation

    def _build_payload(
        self,
        system_text: str,
        conversation: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the Anthropic Messages API request body.

        Args:
            system_text: Combined system prompt text.
            conversation: User/assistant message turns.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.

        Returns:
            Dict suitable for JSON serialisation.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            payload["system"] = system_text
        return payload

    def _check_status(
        self, status: int, body: dict[str, Any], model: str
    ) -> None:
        """Raise typed exceptions for non-2xx responses.

        Args:
            status: HTTP status code.
            body: Decoded JSON response body.
            model: Model used in the request (for exception context).

        Raises:
            APIAuthError: On 401 / 403.
            APIRateLimitError: On 429.
            APIResponseError: On other non-2xx.
        """
        if 200 <= status < 300:
            return

        error_msg = self._extract_error_message(body)

        if status in (401, 403):
            logger.error(
                "AnthropicAdapter auth error (%d): %s", status, error_msg
            )
            raise APIAuthError(
                f"Authentication failed for {self._PROVIDER}: {error_msg}",
                provider=self._PROVIDER,
                model=model,
            )

        if status == 429:
            retry_after = self._extract_retry_after(body)
            logger.warning(
                "AnthropicAdapter rate limited (429), retry_after=%s",
                retry_after,
            )
            raise APIRateLimitError(
                f"Rate limited by {self._PROVIDER}: {error_msg}",
                retry_after=retry_after,
                provider=self._PROVIDER,
                model=model,
            )

        logger.error(
            "AnthropicAdapter API error (%d): %s", status, error_msg
        )
        raise APIResponseError(
            f"API error from {self._PROVIDER} ({status}): {error_msg}",
            status_code=status,
            response_body=str(body),
            provider=self._PROVIDER,
            model=model,
        )

    def _parse_response(self, body: dict[str, Any]) -> APIResponse:
        """Extract text from Anthropic's content-blocks response.

        Anthropic returns ``content: [{type: "text", text: "…"}, …]``.
        This method concatenates all ``text`` blocks.

        Args:
            body: Decoded JSON body from a successful response.

        Returns:
            Normalised ``APIResponse``.

        Raises:
            APIResponseError: If content blocks are missing or empty.
        """
        try:
            content_blocks = body.get("content", [])
            if not content_blocks:
                raise APIResponseError(
                    f"No content blocks in response from {self._PROVIDER}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                )

            text_parts: list[str] = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            text = "".join(text_parts)
            if not text:
                raise APIResponseError(
                    f"Empty text in content blocks from {self._PROVIDER}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                )

            usage = body.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            model_used = body.get("model", "")

            logger.debug(
                "AnthropicAdapter response: model=%s, tokens_in=%d, tokens_out=%d",
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
                "AnthropicAdapter failed to parse response: %s", exc
            )
            raise APIResponseError(
                f"Failed to parse response from {self._PROVIDER}: {exc}",
                response_body=str(body),
                provider=self._PROVIDER,
            ) from exc

    def _extract_error_message(self, body: dict[str, Any]) -> str:
        """Pull a human-readable message from an Anthropic error body.

        Anthropic error format: ``{"type": "error", "error": {"type": …, "message": …}}``.

        Args:
            body: Decoded JSON error body.

        Returns:
            Error message string.
        """
        error = body.get("error", {})
        if isinstance(error, dict):
            return error.get("message", str(error) or "unknown error")
        return str(error) or "unknown error"

    def _extract_retry_after(self, body: dict[str, Any]) -> float | None:
        """Attempt to extract a retry-after hint from a 429 response.

        Anthropic may include ``retry_after`` in the error metadata or
        in the response headers.  Since we only have the body here,
        we attempt a best-effort extraction.

        Args:
            body: Decoded JSON body from a 429 response.

        Returns:
            Seconds to wait, or ``None`` if not available.
        """
        error = body.get("error", {})
        if isinstance(error, dict):
            raw = error.get("retry_after")
            if raw is not None:
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    pass
        return None
