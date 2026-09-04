"""API-configuration models used by the LiteLLM-backed client.

The generic story-generator enums (Tone / Perspective / Register / ... /
StructureType) and ``GenerationConfig`` were removed as part of the
dead-code cleanup — the aviation factory carries its own ``JobSettings``
in :mod:`aviation.state`.

What remains is the API-provider layer that ``core.api_client`` still
uses:

* :class:`APIProvider` — enum for the six historic direct-API providers.
* :class:`APIFormat` — wire-format tag (informational only now that
  LiteLLM does the actual routing).
* :data:`PROVIDER_CONFIG` — reference metadata (base URLs, known models).
* :class:`FallbackPoolEntry` and :class:`APIConfig` — carried into
  ``APIClient`` per request; the model IDs use the LiteLLM convention
  (see :mod:`core.llm.router`).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Provider enum & metadata ──────────────────────────────────────────

class APIProvider(str, Enum):
    """Supported API providers (historic naming — LiteLLM covers the wire)."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class APIFormat(str, Enum):
    """Wire format used by each provider (informational only)."""

    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI_NATIVE = "openai_native"
    ANTHROPIC_NATIVE = "anthropic_native"
    GOOGLE_NATIVE = "google_native"


PROVIDER_CONFIG: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "format": APIFormat.OPENAI_COMPATIBLE,
        "models": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "deepseek/deepseek-chat",
            "qwen/qwen-max",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "format": APIFormat.OPENAI_NATIVE,
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "format": APIFormat.ANTHROPIC_NATIVE,
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "format": APIFormat.GOOGLE_NATIVE,
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "format": APIFormat.OPENAI_COMPATIBLE,
        "models": ["deepseek-chat", "deepseek-coder"],
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "format": APIFormat.OPENAI_COMPATIBLE,
        "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
    },
}


# ── Language list (kept for the Streamlit sidebar dropdown) ───────────

LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "English", "flag": "🇬🇧", "native": "English"},
    "ru": {"name": "Russian", "flag": "🇷🇺", "native": "Русский"},
    "de": {"name": "German", "flag": "🇩🇪", "native": "Deutsch"},
    "fr": {"name": "French", "flag": "🇫🇷", "native": "Français"},
    "pt": {"name": "Portuguese", "flag": "🇵🇹", "native": "Português"},
    "it": {"name": "Italian", "flag": "🇮🇹", "native": "Italiano"},
    "pl": {"name": "Polish", "flag": "🇵🇱", "native": "Polski"},
    "uk": {"name": "Ukrainian", "flag": "🇺🇦", "native": "Українська"},
    "ro": {"name": "Romanian", "flag": "🇷🇴", "native": "Română"},
    "tr": {"name": "Turkish", "flag": "🇹🇷", "native": "Türkçe"},
    "da": {"name": "Danish", "flag": "🇩🇰", "native": "Dansk"},
}
SUPPORTED_LANGUAGE_CODES: list[str] = list(LANGUAGES.keys())


# ── APIConfig + fallback pool ─────────────────────────────────────────

class FallbackPoolEntry(BaseModel):
    """One entry in the fallback provider pool tried after the primary fails."""

    provider: APIProvider = Field(description="API provider.")
    model: str = Field(default="", description="Model identifier.")
    api_key: str = Field(default="", description="API key.")


class APIConfig(BaseModel):
    """API provider configuration for a single generation run."""

    primary_provider: APIProvider = Field(default=APIProvider.OPENROUTER)
    primary_model: str = Field(default="openrouter/anthropic/claude-3.5-sonnet")
    api_key: str = Field(default="")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    auto_fallback: bool = Field(default=True)

    # Legacy single-fallback fields kept for backward compatibility with
    # any saved job payload that still uses them.
    fallback_provider: Optional[APIProvider] = Field(default=APIProvider.OPENAI)
    fallback_model: str = Field(default="gpt-4o")
    fallback_api_key: str = Field(default="")

    fallback_pool: list[FallbackPoolEntry] = Field(default_factory=list)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=120, ge=10, le=600)

    def get_effective_fallback_pool(self) -> list[FallbackPoolEntry]:
        """Return the effective fallback pool (v2 first, legacy second)."""
        if self.fallback_pool:
            return self.fallback_pool
        if (
            self.auto_fallback
            and self.fallback_provider is not None
            and self.fallback_api_key
        ):
            return [
                FallbackPoolEntry(
                    provider=self.fallback_provider,
                    model=self.fallback_model,
                    api_key=self.fallback_api_key,
                )
            ]
        return []
