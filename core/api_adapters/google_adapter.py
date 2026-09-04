"""Google Gemini API adapter.

Handles the Google ``generateContent`` wire format which uses a completely
different request/response shape from OpenAI or Anthropic.  Key
differences:

- Model is part of the URL, not the request body.
- Messages are ``contents: [{role: "user", parts: [{text: "…"}]}]``.
- System instructions are a separate top-level field.
- Responses use ``candidates[0].content.parts[0].text``.
- API key is sent as a query parameter, not a header.
- Safety settings must be explicitly configured to avoid over-blocking.

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

# Safety categories to set to BLOCK_NONE so creative content is not
# filtered.  These map to the HarmCategory enum values in the Gemini API.
_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


class GoogleAdapter:
    """Adapter for the Google Gemini ``generateContent`` API.

    Translates the standard message list into Gemini's ``contents``
    format, passes the API key as a query parameter, and parses the
    ``candidates`` response.

    Args:
        base_url: Gemini API base URL
            (e.g. ``https://generativelanguage.googleapis.com/v1beta``).
        api_key: Google API key (passed as ``?key=…`` query parameter).
        timeout_seconds: Per-request timeout in seconds.
    """

    _PROVIDER = "google"

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
        """Send a ``generateContent`` request to the Gemini API.

        Args:
            messages: Standard chat messages.  System messages are
                extracted into ``system_instruction``.
            model: Model identifier (e.g. ``"gemini-1.5-pro"``).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            An ``APIResponse`` with text, token counts, and model used.

        Raises:
            APIAuthError: On 401 / 403.
            APIRateLimitError: On 429.
            APIResponseError: On other non-2xx status or malformed body.
            APIConnectionError: On network / timeout errors.
        """
        url = self._build_url(model)
        headers = {"Content-Type": "application/json"}
        system_text, contents = self._convert_messages(messages)
        payload = self._build_payload(
            system_text, contents, temperature, max_tokens
        )

        logger.debug(
            "GoogleAdapter sending request to %s, model=%s, temp=%.2f, max_tokens=%d",
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
                return self._parse_response(body, model)

        except (APIAuthError, APIRateLimitError, APIResponseError):
            raise
        except aiohttp.ClientError as exc:
            logger.error("GoogleAdapter connection error: %s", exc)
            raise APIConnectionError(
                f"Connection error for {self._PROVIDER}: {exc}",
                provider=self._PROVIDER,
                model=model,
            ) from exc
        except TimeoutError as exc:
            logger.error(
                "GoogleAdapter request timed out after %ss",
                self._timeout.total,
            )
            raise APIConnectionError(
                f"Request to {self._PROVIDER} timed out after {self._timeout.total}s",
                provider=self._PROVIDER,
                model=model,
            ) from exc

    # ── private helpers ─────────────────────────────────────────────

    def _build_url(self, model: str) -> str:
        """Construct the generateContent endpoint URL.

        The model is embedded in the path, and the API key is a query
        parameter.

        Args:
            model: Model identifier.

        Returns:
            Full URL string.
        """
        return (
            f"{self._base_url}/models/{model}:generateContent"
            f"?key={self._api_key}"
        )

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert standard chat messages into Gemini ``contents`` format.

        Gemini uses ``role="user"`` and ``role="model"`` (not
        ``"assistant"``).  System messages are separated out as
        ``system_instruction``.

        Args:
            messages: Standard chat message list.

        Returns:
            A tuple of ``(system_text, contents)`` ready for the
            Gemini payload.
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            # Gemini uses "model" where OpenAI uses "assistant".
            gemini_role = "model" if role == "assistant" else "user"

            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        return "\n\n".join(system_parts), contents

    def _build_payload(
        self,
        system_text: str,
        contents: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the Gemini ``generateContent`` request body.

        Args:
            system_text: Combined system instruction text.
            contents: Gemini-format conversation turns.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Dict suitable for JSON serialisation.
        """
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
            "safetySettings": _SAFETY_SETTINGS,
        }
        if system_text:
            payload["system_instruction"] = {
                "parts": [{"text": system_text}],
            }
        return payload

    def _check_status(
        self, status: int, body: dict[str, Any], model: str
    ) -> None:
        """Raise typed exceptions for non-2xx responses.

        Args:
            status: HTTP status code.
            body: Decoded JSON response body.
            model: Model used in the request.

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
                "GoogleAdapter auth error (%d): %s", status, error_msg
            )
            raise APIAuthError(
                f"Authentication failed for {self._PROVIDER}: {error_msg}",
                provider=self._PROVIDER,
                model=model,
            )

        if status == 429:
            retry_after = self._extract_retry_after(body)
            logger.warning(
                "GoogleAdapter rate limited (429), retry_after=%s",
                retry_after,
            )
            raise APIRateLimitError(
                f"Rate limited by {self._PROVIDER}: {error_msg}",
                retry_after=retry_after,
                provider=self._PROVIDER,
                model=model,
            )

        logger.error(
            "GoogleAdapter API error (%d): %s", status, error_msg
        )
        raise APIResponseError(
            f"API error from {self._PROVIDER} ({status}): {error_msg}",
            status_code=status,
            response_body=str(body),
            provider=self._PROVIDER,
            model=model,
        )

    def _parse_response(
        self, body: dict[str, Any], model: str
    ) -> APIResponse:
        """Extract text from Gemini's ``candidates`` response structure.

        Expected path: ``candidates[0].content.parts[0].text``.

        Also checks for a ``blockReason`` in ``promptFeedback`` which
        indicates content was blocked by safety filters.

        Args:
            body: Decoded JSON body from a successful response.
            model: Model identifier (used in the returned APIResponse).

        Returns:
            Normalised ``APIResponse``.

        Raises:
            APIResponseError: If candidates are missing, blocked, or empty.
        """
        try:
            # Check for safety blocking.
            prompt_feedback = body.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                logger.warning(
                    "GoogleAdapter content blocked: %s", block_reason
                )
                raise APIResponseError(
                    f"Content blocked by {self._PROVIDER} safety filter: {block_reason}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                    model=model,
                )

            candidates = body.get("candidates", [])
            if not candidates:
                raise APIResponseError(
                    f"No candidates in response from {self._PROVIDER}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                    model=model,
                )

            candidate = candidates[0]

            # Check candidate-level finish reason for safety stops.
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "SAFETY":
                safety_ratings = candidate.get("safetyRatings", [])
                logger.warning(
                    "GoogleAdapter candidate blocked by safety: %s",
                    safety_ratings,
                )
                raise APIResponseError(
                    f"Response blocked by {self._PROVIDER} safety filter "
                    f"(finishReason=SAFETY)",
                    response_body=str(body),
                    provider=self._PROVIDER,
                    model=model,
                )

            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise APIResponseError(
                    f"No content parts in response from {self._PROVIDER}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                    model=model,
                )

            # Concatenate all text parts.
            text_parts: list[str] = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])

            text = "".join(text_parts)
            if not text:
                raise APIResponseError(
                    f"Empty text in response from {self._PROVIDER}",
                    response_body=str(body),
                    provider=self._PROVIDER,
                    model=model,
                )

            # Token counts from usageMetadata.
            usage = body.get("usageMetadata", {})
            tokens_in = usage.get("promptTokenCount", 0)
            tokens_out = usage.get("candidatesTokenCount", 0)

            logger.debug(
                "GoogleAdapter response: model=%s, tokens_in=%d, tokens_out=%d",
                model,
                tokens_in,
                tokens_out,
            )

            return APIResponse(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model,
                raw=body,
            )

        except APIResponseError:
            raise
        except Exception as exc:
            logger.error(
                "GoogleAdapter failed to parse response: %s", exc
            )
            raise APIResponseError(
                f"Failed to parse response from {self._PROVIDER}: {exc}",
                response_body=str(body),
                provider=self._PROVIDER,
                model=model,
            ) from exc

    def _extract_error_message(self, body: dict[str, Any]) -> str:
        """Pull a human-readable error message from a Gemini error body.

        Gemini error format: ``{"error": {"code": …, "message": …, "status": …}}``.

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

        Gemini may include retry information in error details.

        Args:
            body: Decoded JSON body from a 429 response.

        Returns:
            Seconds to wait, or ``None`` if not available.
        """
        error = body.get("error", {})
        if isinstance(error, dict):
            details = error.get("details", [])
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, dict):
                        retry_delay = detail.get("retryDelay")
                        if retry_delay is not None:
                            return self._parse_duration(retry_delay)
        return None

    @staticmethod
    def _parse_duration(value: str | float | int) -> float | None:
        """Parse a duration value that may be seconds or a string like ``"30s"``.

        Args:
            value: Duration value from the API.

        Returns:
            Seconds as float, or ``None`` on parse failure.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().rstrip("s")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None
