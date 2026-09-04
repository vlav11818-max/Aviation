"""API adapter package.

Provides provider-specific adapters that normalise different LLM API
wire formats into a common ``APIResponse`` dataclass.

Adapters
--------
- ``OpenAIAdapter`` -- OpenAI-native, OpenRouter, DeepSeek, Qwen
  (any ``openai_compatible`` or ``openai_native`` provider).
- ``AnthropicAdapter`` -- Anthropic Messages API.
- ``GoogleAdapter`` -- Google Gemini ``generateContent`` API.

Each adapter exposes a single async ``send()`` method returning an
``APIResponse``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class APIResponse:
    """Normalised response returned by every adapter.

    Attributes:
        text: The generated text content.
        tokens_in: Number of input tokens consumed (from usage metadata).
        tokens_out: Number of output tokens generated (from usage metadata).
        model: Model identifier reported by the API.
        raw: Raw response dict for debugging (optional).
    """

    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)
