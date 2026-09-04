"""Configuration data models for generation and API settings.

``GenerationConfig`` captures all creative/structural parameters for a
single story generation run.  ``APIConfig`` captures the provider, model,
keys, and resilience settings for the API layer.

FIX: Added ``FallbackPoolEntry`` and ``fallback_pool`` field to
``APIConfig`` for automatic multi-provider fallback.  All providers
with API keys are tried in order when the primary fails.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Enums for finite option sets ────────────────────────────────────────


class Tone(str, Enum):
    """Available story tone presets."""

    DRAMATIC_CINEMATIC = "dramatic_cinematic"
    SUSPENSEFUL = "suspenseful"
    WARM_EMOTIONAL = "warm_emotional"
    DARK_GOTHIC = "dark_gothic"
    WHIMSICAL = "whimsical"
    INSPIRATIONAL = "inspirational"


class Perspective(str, Enum):
    """Narration perspective options."""

    FIRST_PERSON = "first_person"
    SECOND_PERSON = "second_person"
    THIRD_PERSON = "third_person"
    OMNISCIENT = "omniscient"


class Register(str, Enum):
    """Language register options."""

    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    LITERARY = "literary"
    POETIC = "poetic"


class Pacing(str, Enum):
    """Story pacing options."""

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class Audience(str, Enum):
    """Target audience options."""

    CHILDREN = "children"
    YOUNG_ADULT = "young_adult"
    ALL_AGES = "all_ages"
    MATURE = "mature"


class DialogDensity(str, Enum):
    """Dialog density levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StructureType(str, Enum):
    """Available story structure types."""

    THREE_ACT = "three_act"
    HERO_JOURNEY = "hero_journey"
    IN_MEDIAS_RES = "in_medias_res"
    EPISODIC = "episodic"
    CIRCULAR = "circular"


class APIProvider(str, Enum):
    """Supported API providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class APIFormat(str, Enum):
    """Wire format used by each provider."""

    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI_NATIVE = "openai_native"
    ANTHROPIC_NATIVE = "anthropic_native"
    GOOGLE_NATIVE = "google_native"


# ── Provider metadata ───────────────────────────────────────────────────

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
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
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


# ── Supported languages ─────────────────────────────────────────────────

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


# ── Data models ─────────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    """All creative and structural parameters for one generation run.

    Captures tone, perspective, register, pacing, audience, dialog density,
    target word count, structure type, genres, voiceover flags, and quality
    thresholds.
    """

    tone: Tone = Field(
        default=Tone.DRAMATIC_CINEMATIC,
        description="Story tone preset.",
    )
    perspective: Perspective = Field(
        default=Perspective.THIRD_PERSON,
        description="Narration perspective.",
    )
    register: Register = Field(
        default=Register.CONVERSATIONAL,
        description="Language register.",
    )
    pacing: Pacing = Field(
        default=Pacing.MEDIUM,
        description="Story pacing.",
    )
    audience: Audience = Field(
        default=Audience.ALL_AGES,
        description="Target audience.",
    )
    dialog_density: DialogDensity = Field(
        default=DialogDensity.MEDIUM,
        description="Density of dialog in the story.",
    )
    target_words: int = Field(
        default=3000,
        ge=500,
        le=10000,
        description="Target word count for the generated story.",
    )
    structure: StructureType = Field(
        default=StructureType.THREE_ACT,
        description="Story structure template.",
    )
    genres: list[str] = Field(
        default_factory=lambda: ["fantasy", "mystery"],
        description="Selected genre tags.",
    )
    min_score: float = Field(
        default=9.0,
        ge=0.0,
        le=10.0,
        description="Minimum acceptable evaluation score (0–10).",
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum evaluation/revision attempts before accepting.",
    )
    voiceover_optimized: bool = Field(
        default=True,
        description="Whether to optimize text for voiceover/TTS.",
    )
    avoid_complex_sentences: bool = Field(
        default=True,
        description="Simplify sentence structure for clarity.",
    )
    pause_markers: bool = Field(
        default=True,
        description="Include punctuation-based pause markers.",
    )
    no_headers: bool = Field(
        default=True,
        description="Suppress chapter/section headers in output.",
    )
    no_meta_comments: bool = Field(
        default=True,
        description="Remove AI meta-commentary from output.",
    )


class FallbackPoolEntry(BaseModel):
    """A single entry in the fallback provider pool.

    Represents one provider that can be used as a fallback when the
    primary provider fails.  Entries are tried in list order.

    Attributes:
        provider: The API provider enum value.
        model: Default model to use for this provider.
        api_key: API key for this provider.
    """

    provider: APIProvider = Field(description="API provider.")
    model: str = Field(default="", description="Model identifier.")
    api_key: str = Field(default="", description="API key.")


class APIConfig(BaseModel):
    """API provider configuration for a generation run.

    Holds provider, model, API key, base URL, fallback settings, and
    resilience parameters (retries, timeout).

    The ``fallback_pool`` is an ordered list of alternative providers
    that are tried automatically when the primary provider fails.  It
    is populated from all providers that have an API key configured.
    The legacy ``fallback_provider`` / ``fallback_model`` /
    ``fallback_api_key`` fields are still supported for backward
    compatibility — if ``fallback_pool`` is empty but the legacy
    fields are set, a single-entry pool is built from them.
    """

    primary_provider: APIProvider = Field(
        default=APIProvider.OPENROUTER,
        description="Primary API provider.",
    )
    primary_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model identifier for the primary provider.",
    )
    api_key: str = Field(
        default="",
        description="API key for the primary provider.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the primary provider API.",
    )
    auto_fallback: bool = Field(
        default=True,
        description="Enable automatic fallback on failure.",
    )
    # ── Legacy single-fallback fields (backward compatibility) ───────
    fallback_provider: APIProvider | None = Field(
        default=APIProvider.OPENAI,
        description="Legacy: fallback API provider (None to disable).",
    )
    fallback_model: str = Field(
        default="gpt-4o",
        description="Legacy: model identifier for the fallback provider.",
    )
    fallback_api_key: str = Field(
        default="",
        description="Legacy: API key for the fallback provider.",
    )
    # ── Fallback pool (new) ──────────────────────────────────────────
    fallback_pool: list[FallbackPoolEntry] = Field(
        default_factory=list,
        description=(
            "Ordered list of fallback providers.  Each entry is tried "
            "in order when the primary fails.  Built automatically from "
            "all providers with API keys, excluding the primary."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts per API call.",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout in seconds for a single API call.",
    )

    def get_effective_fallback_pool(self) -> list[FallbackPoolEntry]:
        """Return the effective fallback pool.

        If ``fallback_pool`` is populated, returns it as-is.  Otherwise,
        if the legacy ``fallback_provider`` is set with a key, builds a
        single-entry pool for backward compatibility.

        Returns:
            Ordered list of fallback pool entries (may be empty).
        """
        if self.fallback_pool:
            return self.fallback_pool

        # Legacy compatibility: build a single-entry pool.
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
